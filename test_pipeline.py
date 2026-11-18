"""Focused checks for timestamp math, overlap ownership, and gap reporting."""
import unittest
from transcribe import audit, boundaries, merge_result, stamp


class PipelineTests(unittest.TestCase):
    def test_timestamps_keep_hours_and_milliseconds(self):
        self.assertEqual(stamp(5811.123, True), "01:36:51,123")
        self.assertEqual(stamp(59.9998), "00:01:00.000")

    def test_overlap_words_belong_to_exactly_one_section(self):
        raw = {"segments": [{"start": 0, "end": 6, "text": " alpha beta gamma", "words": [
            {"start": 1, "end": 2, "word": " alpha"},
            {"start": 2.5, "end": 3.5, "word": " beta"},
            {"start": 4, "end": 5, "word": " gamma"}]}]}
        req = {"audio_start": 297, "core_start": 0, "core_end": 300}
        left = merge_result(raw, req, 600, "1")
        req = {"audio_start": 297, "core_start": 300, "core_end": 600}
        right = merge_result(raw, req, 600, "2")
        self.assertEqual("".join(s["text"] for s in left+right), " alpha beta gamma")
        self.assertEqual(right[0]["words"][0]["start"], 299.5)

    def test_plan_has_no_unprocessed_ranges(self):
        plan = boundaries(5811, [[298, 300], [604, 606]], 300)
        self.assertEqual(plan[0][0], 0)
        self.assertEqual(plan[-1][1], 5811)
        self.assertTrue(all(a[1] == b[0] for a,b in zip(plan, plan[1:])))
        self.assertAlmostEqual(sum(b-a for a,b in plan),5811)

    def test_gap_and_failure_are_not_called_silence_or_success(self):
        segments = [{"start":0,"end":2,"text":"speech","chunk_id":"1"}]
        result = audit(segments,20,[],[{"start":10,"end":20}], [{"start":0,"end":10}])
        self.assertFalse(result["full_audio_processed"])
        self.assertEqual(result["gaps_at_least_8_seconds"][0]["status"],"review_untranscribed_audio")


if __name__ == "__main__":
    unittest.main()
