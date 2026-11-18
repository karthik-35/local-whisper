# Local Whisper on this Mac

This installation uses the official [`mlx-whisper` implementation from Apple's MLX examples](https://github.com/ml-explore/mlx-examples/tree/main/whisper) and exactly [`mlx-community/whisper-large-v3-turbo`](https://huggingface.co/mlx-community/whisper-large-v3-turbo). It uses the M5 GPU through MLX Metal. There are no paid APIs.

## Start another transcription

Open Terminal and run:

```sh
~/local-whisper/transcribe
```

A macOS file picker opens. Select a recording. A 60-second test runs first; the complete recording follows automatically after the test exports validate.

If the picker is unavailable or canceled, provide the absolute path:

```sh
~/local-whisper/transcribe "/absolute/path/to/recording.m4a"
```

In Finder, select the recording and press Option–Command–C to copy its absolute path. Keep quotation marks around paths containing spaces.

The command automatically detects the spoken language for each processing section and uses the transcription task, preserving the spoken language. To specify one language explicitly:

```sh
~/local-whisper/transcribe "/absolute/path/to/recording.m4a" --language en
```

Use `te`, `hi`, `es`, etc. as appropriate. A section containing several languages may need review because Whisper selects one language per section.

## Results

Files are saved in `~/Documents/WhisperTranscripts` using the recording's base name:

- `.txt`: readable text, one recognized segment per line.
- `.srt`: numbered subtitles with original-recording timestamps.
- `.vtt`: WebVTT subtitles with original-recording timestamps.
- `.json`: text, word/segment timestamps, probabilities, model revision, processing sections, and quality information.
- `.quality.json`: export verification, processed duration, failed ranges, uncertain speech, repeated phrases, and untranscribed intervals of at least eight seconds.

An existing recording's exports are preserved when a different run would use the same name; a run identifier is added. Repeating the same command resumes its checkpoints and regenerates its own exports. Output paths are printed at completion.

`full_audio_processed: true` means every decoded sample was included in successfully processed sections and decoded duration agrees with the audio duration reported by FFprobe. It does not guarantee perfect recognition. Review `suspicious_segments`, `boundary_overlaps`, and gaps marked `review_untranscribed_audio` in the quality report. Those reports contain timestamps rather than guesses for missing speech. Subtitle timestamps end at the last recognized speech; silence at the end does not get a fabricated subtitle.

## Local processing and recovery

The `transcribe` launcher denies network access to the entire transcription process using the macOS sandbox. It also sets Hugging Face offline mode and disables telemetry. Model downloads are completed separately before use. Neither recordings nor transcripts are sent to external services. Do not bypass the launcher when transcribing private material.

The original recording is opened for reading only. Its SHA-256 is checked before and after processing. FFmpeg creates a separate mono, 16 kHz working waveform required by Whisper; the source is not converted in place, normalized, denoised, or shortened. The full waveform is decoded with FFmpeg error checking.

Processing uses approximately five-minute sections, preferring quiet boundaries, with three seconds of overlapping context on each side. Word timestamp midpoints assign the overlapping words to their owning section. All raw section results are retained for checking boundary ambiguities. Context prompting is disabled to reduce repetition loops; Whisper's confidence, repetition, no-speech, and hallucination-silence checks remain enabled. No recognized repeated text is silently deleted by the export script.

Each section runs in an isolated GPU process. If it crashes or exceeds its time limit, the script retries progressively smaller sections. Unrecoverable sections are recorded as failures; partial exports are marked incomplete and the command exits with status 2. Checkpoints are kept privately in `~/Documents/WhisperTranscripts/.whisper-work/`. The temporary test WAV is removed after the test succeeds. The working waveform and raw results remain available for resuming and auditing.

Keep the Mac connected to power and its lid open. `caffeinate` prevents idle system sleep during the command; it cannot prevent shutdown or every lid-closed sleep condition. If interrupted, run the same command with the same input and options to resume. Do not run two copies on the same recording simultaneously.

## Common errors

- **Picker canceled / file not found:** pass a quoted absolute path. For macOS protected folders, grant Terminal or the launching app access when macOS requests it, or choose a readable copy; keep the original.
- **No readable text in the test:** the opening minute may contain silence, music, or very quiet speech. The script tries other test positions automatically. Verify the recording has audible speech and optionally specify `--language` if detection is wrong.
- **FFmpeg error:** see the run's `ffmpeg.log` under `.whisper-work`. A damaged or unsupported input can stop decoding. The source remains unchanged. Confirm `ffmpeg -version` and `ffprobe -version` work.
- **Metal unavailable / memory error:** this setup requires native Apple Silicon Python and a working MLX Metal backend. Close memory-heavy applications and rerun. For smaller sections use `--chunk-seconds 120`; changing options starts separate checkpoints. CPU fallback is intentionally disabled so a GPU failure is visible.
- **Multiple audio streams:** the default is audio stream 0. Inspect the saved `probe.json`, then use `--audio-stream 1` to select another stream if needed. Each run processes the selected stream completely.
- **Missing cached weights:** run `~/local-whisper/.venv/bin/python ~/local-whisper/download_model.py` with internet access, then rerun the transcription command. This download script handles model files only.
- **Dependency issues:** use `~/local-whisper/.venv/bin/python -m pip check`. Installed versions are recorded in `requirements.lock.txt`. The virtual environment is dedicated to this project; other Python environments are not changed.
- **An apparent gap or repeated phrase:** review the indicated original-file timestamps in the quality report and raw section results. Quiet audio is an energy measurement, not proof that nobody spoke. Nothing is invented to fill a gap.

## Installation recorded on this Mac

macOS 26.6.2; Apple M5; 24 GB unified memory; Homebrew Python 3.13.5; FFmpeg 8.1.2; `mlx-whisper` 0.4.3; MLX 0.32.4. The dedicated environment is `~/local-whisper/.venv`. Model weights occupy about 1.61 GB in the Hugging Face cache; `model-cache.json` records the pinned local snapshot. Expect roughly 384 MB of working waveform per 100 minutes plus raw results and exports.

To inspect options or run focused timestamp/coverage checks:

```sh
~/local-whisper/transcribe --help
cd ~/local-whisper
.venv/bin/python -m unittest -v test_pipeline.py
```
