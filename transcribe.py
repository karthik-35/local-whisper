#!/usr/bin/env python3
"""Local, resumable MLX Whisper transcription. No audio/transcript network I/O."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import wave

ROOT = Path(__file__).resolve().parent
REPO = "mlx-community/whisper-large-v3-turbo"
SR = 16000
VERSION = "1.0"


def save_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def load_json(path):
    return json.loads(Path(path).read_text())


def stamp(seconds, srt=False):
    ms = max(0, round(seconds * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02}{',' if srt else '.'}{ms:03}"


def exports(base, data):
    """Write all four formats atomically; transcript contents never go to stdout."""
    segments = data["segments"]
    texts = [s["text"].strip() for s in segments]
    formats = {"txt": "\n".join(texts) + "\n", "srt": "", "vtt": "WEBVTT\n\n"}
    for i, seg in enumerate(segments, 1):
        # Arrow is reserved in subtitle syntax.
        text = seg["text"].strip().replace("-->", "→")
        formats["srt"] += f"{i}\n{stamp(seg['start'], True)} --> {stamp(seg['end'], True)}\n{text}\n\n"
        formats["vtt"] += f"{stamp(seg['start'])} --> {stamp(seg['end'])}\n{text}\n\n"
    paths = {}
    for ext, text in formats.items():
        p = Path(str(base) + "." + ext)
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(p)
        paths[ext] = str(p)
    p = Path(str(base) + ".json")
    save_json(p, data)
    paths["json"] = str(p)
    return paths


def validate_exports(paths, segments, duration):
    assert segments, "No transcription segments were produced."
    chars = sum(c.isalnum() for s in segments for c in s["text"])
    assert chars > 0, "Transcript contains no letters or numbers."
    assert all(0 <= s["start"] < s["end"] <= duration + .001 for s in segments)
    assert all(a["start"] <= b["start"] for a, b in zip(segments, segments[1:]))
    for ext, path in paths.items():
        assert Path(path).stat().st_size > 0, f"Empty {ext} export"
    assert len(load_json(paths["json"])["segments"]) == len(segments)
    assert Path(paths["srt"]).read_text().count(" --> ") == len(segments)
    assert Path(paths["vtt"]).read_text().startswith("WEBVTT\n")
    assert Path(paths["vtt"]).read_text().count(" --> ") == len(segments)
    assert "\ufffd" not in Path(paths["txt"]).read_text(), "Invalid text encoding"
    return {"segments": len(segments), "alphanumeric_characters": chars,
            "last_subtitle_end_seconds": segments[-1]["end"],
            "files": {ext: {"path": p, "bytes": Path(p).stat().st_size} for ext, p in paths.items()}}


def worker(request_file):
    """Isolated GPU process: a crash cannot corrupt previous chunk checkpoints."""
    import mlx.core as mx
    import mlx_whisper
    import numpy as np
    req = load_json(request_file)
    if not mx.metal.is_available():
        raise RuntimeError("MLX Metal GPU is unavailable; CPU fallback is disabled.")
    mx.set_default_device(mx.gpu)
    mx.random.seed(0)
    if req.get("test_wav"):
        audio = req["test_wav"]
    else:
        full = np.memmap(req["pcm"], dtype="<f4", mode="r")
        audio = np.array(full[round(req["audio_start"]*SR):round(req["audio_end"]*SR)])
    start = time.monotonic()
    result = mlx_whisper.transcribe(
        audio, path_or_hf_repo=req["model"], language=req["language"],
        task="transcribe", verbose=None, fp16=True,
        condition_on_previous_text=False, word_timestamps=True,
        temperature=(0.0, 0.2, 0.4, 0.6), compression_ratio_threshold=2.4,
        logprob_threshold=-1.0, no_speech_threshold=0.6,
        hallucination_silence_threshold=2.0,
    )
    result["elapsed_seconds"] = round(time.monotonic()-start, 3)
    result["device"] = str(mx.default_device())
    save_json(req["result"], result)


def infer(work, pcm, model, start, end, duration, language, test_wav=None):
    audio_start = max(0, start - 3) if not test_wav else start
    audio_end = min(duration, end + 3) if not test_wav else end
    name = f"{round(start*1000):010}-{round(end*1000):010}" + ("-test" if test_wav else "")
    target = work / (name + ".raw.json")
    reqfile = work / (name + ".request.json")
    req = {"pcm": str(pcm), "model": model, "audio_start": audio_start,
           "audio_end": audio_end, "core_start": start, "core_end": end,
           "language": language, "result": str(target), "test_wav": test_wav}
    if target.exists() and reqfile.exists() and load_json(reqfile) == req:
        return load_json(target), req
    save_json(reqfile, req)
    with (work / (name + ".log")).open("w") as log:
        try:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", str(reqfile)],
                           stdout=log, stderr=log, check=True,
                           timeout=max(240, (audio_end-audio_start)*4))
        except (subprocess.SubprocessError, OSError) as exc:
            # Do not print logs: some dependency errors could include decoded text.
            raise RuntimeError(f"Inference failed at {stamp(start)}–{stamp(end)}; see {log.name}") from exc
    return load_json(target), req


def merge_result(raw, req, duration, chunk_id):
    """Assign overlap words by timestamp midpoint; never remove repeated content."""
    output = []
    offset = req["audio_start"]
    lo, hi = req["core_start"], req["core_end"]
    for original in raw.get("segments", []):
        if not original.get("text", "").strip():
            continue
        seg = {k: v for k, v in original.items() if k not in ("tokens", "words", "seek", "id")}
        words = []
        for original_word in original.get("words", []):
            word = dict(original_word)
            word["start"] = round(max(0, offset + float(word["start"])), 3)
            word["end"] = round(min(duration, offset + float(word["end"])), 3)
            mid = (word["start"]+word["end"])/2
            if lo <= mid < hi or (hi == duration and mid == hi):
                words.append(word)
        if original.get("words"):
            if not words:
                continue
            seg.update(text="".join(w["word"] for w in words), words=words,
                       start=words[0]["start"], end=words[-1]["end"])
        else:
            seg["start"] = round(max(0, offset+float(original["start"])), 3)
            seg["end"] = round(min(duration, offset+float(original["end"])), 3)
            if not lo <= (seg["start"]+seg["end"])/2 < hi:
                continue
        if seg["end"] <= seg["start"]:
            seg["end"] = min(duration, seg["start"]+.02)
        if seg["end"] > seg["start"]:
            seg["chunk_id"] = chunk_id
            output.append(seg)
    return output


def quiet_ranges(pcm):
    import numpy as np
    # Only used for boundary selection and gap reporting, never to discard audio.
    block = 800  # 50 milliseconds
    result = []
    opening = None
    for i in range(0, len(pcm), SR*30):
        part = np.asarray(pcm[i:min(len(pcm),i+SR*30)])
        for j in range(0, len(part), block):
            a = part[j:j+block]
            quiet = float(np.sqrt(np.mean(a*a))) < 10**(-45/20)
            t = (i+j)/SR
            if quiet and opening is None:
                opening = t
            elif not quiet and opening is not None:
                if t-opening >= .5:
                    result.append([round(opening,3),round(t,3)])
                opening = None
    if opening is not None and len(pcm)/SR-opening >= .5:
        result.append([round(opening,3),len(pcm)/SR])
    return result


def boundaries(duration, quiet, chunk_seconds):
    cuts = [0.0]
    while duration-cuts[-1] > chunk_seconds+30:
        ideal = cuts[-1]+chunk_seconds
        candidates = [(max(a,ideal-20)+min(b,ideal+20))/2 for a,b in quiet
                      if b >= ideal-20 and a <= ideal+20]
        cuts.append(round(min(candidates,key=lambda t:abs(t-ideal)) if candidates else ideal,3))
    cuts.append(duration)
    return list(zip(cuts,cuts[1:]))


def audit(segments, duration, quiet, failures, chunks):
    flags = []
    previous_texts = []
    for s in segments:
        reasons = []
        if s.get("avg_logprob", 0) < -1:
            reasons.append("low_model_confidence")
        if s.get("compression_ratio", 0) > 2.4:
            reasons.append("possible_repetition")
        if s.get("no_speech_prob", 0) > .6:
            reasons.append("model_uncertain_whether_speech")
        words = s.get("words", [])
        if any(w.get("probability",1)<.2 for w in words):
            reasons.append("one_or_more_low_confidence_words")
        normalized = re.sub(r"\W+", "", s["text"]).casefold()
        if len(normalized)>10 and normalized in previous_texts[-4:]:
            reasons.append("repeated_nearby_phrase_review_audio")
        previous_texts.append(normalized)
        if reasons:
            flags.append({"start":s["start"],"end":s["end"],"reasons":reasons})
    gaps = []
    cursor = 0.0
    for seg in segments + [{"start":duration,"end":duration}]:
        if seg["start"]-cursor >= 8:
            start,end = cursor,seg["start"]
            quiet_s = sum(max(0,min(end,b)-max(start,a)) for a,b in quiet)
            gaps.append({"start":round(start,3),"end":round(end,3),
                         "seconds":round(end-start,3),"quiet_audio_fraction":round(quiet_s/(end-start),3),
                         "status":"mostly_low_level_audio" if quiet_s/(end-start)>.9 else "review_untranscribed_audio"})
        cursor = max(cursor,seg["end"])
    overlaps = [{"start":b["start"],"end":a["end"],"reason":"overlap_at_chunk_boundary_review"}
                for a,b in zip(segments,segments[1:]) if a["chunk_id"]!=b["chunk_id"] and a["end"]>b["start"]+.1]
    return {"full_audio_processed":not failures,
            "processed_audio_seconds":round(sum(c['end']-c['start'] for c in chunks),3),
            "decoded_audio_duration_seconds":duration,"failed_ranges":failures,
            "suspicious_segments":flags,"gaps_at_least_8_seconds":gaps,"boundary_overlaps":overlaps,
            "notes":["Every decoded sample is assigned to a processing section; overlapping context is also transcribed.",
                     "Words in overlapping context are assigned by their timestamp midpoint. Raw results are retained for review.",
                     "Low-level audio is measured below -45 dBFS, not proof of silence. Gaps are not filled with invented text.",
                     "Full processing and valid exports do not guarantee perfect word recognition. Review flagged ranges."]}


def choose_file():
    r = subprocess.run(["osascript","-e",'POSIX path of (choose file with prompt "Choose a recording for local Whisper transcription")'],
                       capture_output=True,text=True)
    if r.returncode:
        raise RuntimeError("File picker unavailable or canceled. Run ~/local-whisper/transcribe '/absolute/path/to/audio'.")
    return Path(r.stdout.strip()).expanduser().resolve()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024), b""):
            h.update(b)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio",nargs="?",help="Audio path; omit to open the macOS file picker")
    parser.add_argument("--language",help="Language code such as en or te; default: automatic for each section")
    parser.add_argument("--output-dir",type=Path,default=Path.home()/"Documents"/"WhisperTranscripts")
    parser.add_argument("--chunk-seconds",type=int,default=300)
    parser.add_argument("--audio-stream",type=int,default=0,help="Audio stream index (default 0)")
    args = parser.parse_args()
    if not 60 <= args.chunk_seconds <= 900:
        parser.error("--chunk-seconds must be between 60 and 900")
    os.umask(0o077)
    source = Path(args.audio).expanduser().resolve() if args.audio else choose_file()
    if not source.is_file():
        raise RuntimeError(f"Recording does not exist: {source}")
    model = load_json(ROOT/"model-cache.json")
    assert model["repo"] == REPO and (Path(model["path"])/"weights.safetensors").is_file(), "Model is not cached."
    outdir = args.output_dir.expanduser().resolve()
    outdir.mkdir(parents=True,exist_ok=True)
    print("Inspecting recording and verifying source fingerprint…",flush=True)
    fingerprint = sha256(source)
    config = {"version":VERSION,"sha256":fingerprint,"language":args.language,
              "chunk_seconds":args.chunk_seconds,"audio_stream":args.audio_stream,"revision":model["revision"]}
    key = hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:16]
    work = outdir/".whisper-work"/key
    work.mkdir(parents=True,exist_ok=True)
    lock = (work/"run.lock").open("w")
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("This recording is already being transcribed by another process.")
    probe_r = subprocess.run(["ffprobe","-v","error","-show_format","-show_streams","-of","json",str(source)],
                             capture_output=True,text=True,check=True)
    probe = json.loads(probe_r.stdout)
    streams = [s for s in probe["streams"] if s["codec_type"]=="audio"]
    if not streams or not 0<=args.audio_stream<len(streams):
        raise RuntimeError("Selected audio stream does not exist.")
    stream = streams[args.audio_stream]
    if len(streams)>1:
        print(f"NOTICE: {len(streams)} audio streams; using audio stream {args.audio_stream}. See probe.json.",flush=True)
    print(json.dumps({"source":str(source),"codec":stream.get("codec_name"),
                      "sample_rate":stream.get("sample_rate"),"channels":stream.get("channels"),
                      "container":probe["format"].get("format_name"),"probe_duration":probe["format"].get("duration")}),flush=True)
    save_json(work/"probe.json",probe)
    pcm = work/"audio-16k-mono.f32"
    if not pcm.exists():
        temp = pcm.with_name(pcm.name+".tmp")
        print("Decoding to a separate 16 kHz mono working copy; source stays unchanged…",flush=True)
        with (work/"ffmpeg.log").open("w") as log:
            subprocess.run(["ffmpeg","-nostdin","-hide_banner","-v","error","-xerror","-y","-i",str(source),
                            "-map",f"0:a:{args.audio_stream}","-vn","-ac","1","-ar",str(SR),"-c:a","pcm_f32le",
                            "-f","f32le",str(temp)],stderr=log,stdout=subprocess.DEVNULL,check=True)
        temp.replace(pcm)
    import numpy as np
    if pcm.stat().st_size == 0 or pcm.stat().st_size%4:
        raise RuntimeError("Decoded audio is empty or malformed.")
    samples = np.memmap(pcm,dtype="<f4",mode="r")
    duration = len(samples)/SR
    print(f"Decoded duration: {stamp(duration)} ({duration:.3f} seconds).",flush=True)
    quiet = quiet_ranges(samples)
    save_json(work/"low-level-audio-ranges.json",quiet)
    # Choose a non-silent 60-second test, while keeping absolute timestamps.
    test_start = 0.0
    if quiet and quiet[0][0]==0 and quiet[0][1]>5:
        test_start = min(max(0,quiet[0][1]-.5),max(0,duration-60))
    test_wav = work/"test-60s.wav"
    candidates = list(dict.fromkeys([test_start]+[round(max(0,duration-60)*f,3) for f in (.1,.5,.8)]))
    for test_start in candidates:
        test_end = min(duration,test_start+60)
        with wave.open(str(test_wav),"wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
            audio = np.asarray(samples[round(test_start*SR):round(test_end*SR)])
            w.writeframes((np.clip(audio,-1,1)*32767).astype("<i2").tobytes())
        print(f"Testing {stamp(test_start)}–{stamp(test_end)} on the MLX GPU…",flush=True)
        raw,req = infer(work,pcm,model["path"],test_start,test_end,duration,args.language,str(test_wav))
        test_segments = merge_result(raw,req,duration,"test")
        if not any(c.isalnum() for s in test_segments for c in s["text"]):
            print("No readable speech in this test segment; trying another position.",flush=True)
            continue
        for i,s in enumerate(test_segments): s["id"] = i
        test_paths = exports(work/"test-60s",{"segments":test_segments,"language":raw.get("language"),"model":REPO})
        test_validation = validate_exports(test_paths,test_segments,duration)
        save_json(work/"test-validation.json",test_validation)
        test_wav.unlink(missing_ok=True)
        break
    else:
        raise RuntimeError(f"No readable speech in the test positions. Check the recording or language. Test artifacts: {work}")
    print(f"Test passed: {len(test_segments)} readable segments; language={raw.get('language')}; {raw['elapsed_seconds']:.1f} seconds inference. All four test exports validated.",flush=True)
    if (work/"output-base.json").exists():
        base = Path(load_json(work/"output-base.json")["base"])
    else:
        stem = source.stem[:120]
        base = outdir/stem
        if any(Path(str(base)+"."+e).exists() for e in ("txt","srt","vtt","json","quality.json")):
            base = outdir/(stem+"-"+key)
        save_json(work/"output-base.json",{"base":str(base)})
    chunks,failed,segments = [],[],[]
    plan = boundaries(duration,quiet,args.chunk_seconds)
    save_json(work/"plan.json",{"config":config,"source":str(source),"duration":duration,"sections":plan})

    def process(start,end,chunk_id):
        try:
            raw,req = infer(work,pcm,model["path"],start,end,duration,args.language)
            merged = merge_result(raw,req,duration,chunk_id)
            chunks.append({"id":chunk_id,"start":start,"end":end,"language":raw.get("language"),
                           "elapsed_seconds":raw["elapsed_seconds"],"raw_result":req["result"],"device":raw["device"]})
            segments.extend(merged)
        except RuntimeError as exc:
            if end-start>60:
                middle = round((start+end)/2,3)
                print(f"Retrying {stamp(start)}–{stamp(end)} as smaller sections.",flush=True)
                process(start,middle,chunk_id+"a"); process(middle,end,chunk_id+"b")
            else:
                failed.append({"start":start,"end":end,"error":str(exc)})
                print(str(exc),flush=True)

    started = time.monotonic()
    for i,(start,end) in enumerate(plan,1):
        print(f"Section {i}/{len(plan)}: {stamp(start)}–{stamp(end)}",flush=True)
        process(start,end,str(i))
        save_json(work/"progress.json",{"completed_sections":i,"total_sections":len(plan),
                                      "processed_through_seconds":end,"failed_ranges":failed})
    segments.sort(key=lambda s:(s["start"],s["end"]))
    for i,s in enumerate(segments): s["id"]=i
    quality = audit(segments,duration,quiet,failed,chunks)
    if sha256(source)!=fingerprint:
        raise RuntimeError("Source changed during transcription. Outputs will not be finalized.")
    probed_duration = float(stream.get("duration") or probe["format"].get("duration") or duration)
    quality["probe_duration_seconds"]=probed_duration
    quality["decode_duration_difference_seconds"]=round(duration-probed_duration,3)
    quality["source_sha256_verified_unchanged"]=True
    if abs(duration-probed_duration)>max(2,probed_duration*.001):
        quality["duration_warning"]="Decoded and probed durations differ; check the input and chosen stream."
        quality["full_audio_processed"]=False
    data = {"text":"\n".join(s["text"].strip() for s in segments),"segments":segments,
            "language":next(iter(set(c['language'] for c in chunks))) if len(set(c['language'] for c in chunks))==1 else "multiple",
            "source":str(source),"source_sha256":fingerprint,"duration":duration,"model":REPO,
            "model_revision":model["revision"],"framework":"mlx-whisper",
            "versions":{p:importlib.metadata.version(p) for p in ("mlx-whisper","mlx","numpy")},
            "created_at":datetime.now(timezone.utc).isoformat(),"chunks":chunks,"quality":quality}
    paths = exports(base,data)
    quality["export_validation"]=validate_exports(paths,segments,duration)
    quality["run_elapsed_seconds"]=round(time.monotonic()-started,3)
    save_json(Path(str(base)+".quality.json"),quality)
    save_json(work/"completed.json",{"paths":paths,"quality":str(base)+".quality.json","full_audio_processed":quality["full_audio_processed"]})
    print(json.dumps({"full_audio_processed":quality["full_audio_processed"],"duration_seconds":duration,
                      "segments":len(segments),"suspicious_segments":len(quality["suspicious_segments"]),
                      "review_gaps":sum(g["status"]=="review_untranscribed_audio" for g in quality["gaps_at_least_8_seconds"]),
                      "paths":paths,"quality_report":str(base)+".quality.json"},indent=2),flush=True)
    return 0 if quality["full_audio_processed"] else 2


if __name__ == "__main__":
    if len(sys.argv)==3 and sys.argv[1]=="--worker":
        worker(sys.argv[2])
    else:
        try:
            sys.exit(main())
        except KeyboardInterrupt:
            print("Interrupted. Checkpoints are saved; rerun the same command to resume.",file=sys.stderr)
            sys.exit(130)
        except Exception as exc:
            print(f"ERROR: {exc}",file=sys.stderr)
            sys.exit(1)
