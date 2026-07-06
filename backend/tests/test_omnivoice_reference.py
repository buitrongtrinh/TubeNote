import unittest

from backend.services.dubbing.engines.omnivoice import (
    _segment_text,
    pick_reference_window,
)


def _raw(text: str, start: float, duration: float) -> dict:
    """Entry gốc (subtitle chưa dịch / regenerate path) — có key ``text``."""
    return {"text": text, "start": start, "duration": duration}


def _merged(text_tts: str, start: float, duration: float) -> dict:
    """Entry đã merge cho TTS (luồng dub chính) — chỉ có ``text_tts``, không
    có ``text`` (đúng cấu trúc thật của merge_segments())."""
    return {"text_tts": text_tts, "source_indices": [0], "source_texts": [text_tts], "start": start, "duration": duration}


LONG_SENTENCE = "This is a long enough sentence to pass the letter count threshold easily"


class SegmentTextTests(unittest.TestCase):
    def test_prefers_text_over_translated_fields(self):
        """Reference audio được cắt từ audio NGUỒN (ngôn ngữ gốc) nên transcript
        phải ưu tiên 'text' (gốc), không phải bản dịch text_tts/text_vi —
        dùng nhầm bản dịch sẽ khớp sai ngôn ngữ với audio thật."""
        item = {"text_tts": "bản dịch", "text": "original version"}
        self.assertEqual(_segment_text(item), "original version")

    def test_falls_back_to_text_tts_when_no_text(self):
        item = {"text_tts": "tts version"}
        self.assertEqual(_segment_text(item), "tts version")

    def test_falls_back_to_source_texts_list(self):
        item = {"source_texts": ["a", "b"]}
        self.assertEqual(_segment_text(item), "a b")

    def test_missing_everything_is_empty_string(self):
        self.assertEqual(_segment_text({"start": 0.0, "duration": 1.0}), "")


class PickReferenceWindowTests(unittest.TestCase):
    def test_regression_merged_data_tts_entries_are_found(self):
        """Bug thật đã gặp: data_tts (luồng dub chính) chỉ có text_tts, không
        có 'text' -> trước khi sửa, hàm luôn báo lỗi bất kể độ dài video."""
        data = [_merged(LONG_SENTENCE, 0.0, 5.0)]
        start, duration, transcript = pick_reference_window(data)
        self.assertEqual(start, 0.0)
        self.assertEqual(duration, 5.0)
        self.assertEqual(transcript, LONG_SENTENCE)

    def test_raw_text_field_still_works(self):
        data = [_raw(LONG_SENTENCE, 2.0, 6.0)]
        start, duration, transcript = pick_reference_window(data)
        self.assertEqual(start, 2.0)
        self.assertEqual(duration, 6.0)

    def test_picks_window_closest_to_seven_seconds(self):
        data = [
            _raw(LONG_SENTENCE, 0.0, 3.0),
            _raw(LONG_SENTENCE, 10.0, 7.0),  # duration from here == 7.0, ideal
        ]
        start, duration, _ = pick_reference_window(data)
        self.assertEqual(start, 10.0)
        self.assertAlmostEqual(duration, 7.0)

    def test_too_short_and_no_text_raises(self):
        data = [_raw("hi", 0.0, 1.0)]
        with self.assertRaises(ValueError):
            pick_reference_window(data)

    def test_all_segments_over_ten_seconds_uses_truncated_fallback(self):
        """sentence_max_words=0 -> segments có thể dài hơn 10s; không còn tổ
        hợp [3,10]s nào sạch -> fallback cắt bớt thay vì báo lỗi."""
        long_text = " ".join(["word"] * 40)  # plenty of letters
        data = [_raw(long_text, 5.0, 18.0)]
        start, duration, transcript = pick_reference_window(data)
        self.assertEqual(start, 5.0)
        self.assertAlmostEqual(duration, 7.0)
        self.assertLess(len(transcript), len(long_text))
        self.assertTrue(transcript)

    def test_fallback_skips_segments_still_too_short(self):
        data = [
            _raw("too short", 0.0, 1.0),
            _raw(" ".join(["word"] * 40), 5.0, 15.0),
        ]
        start, duration, _ = pick_reference_window(data)
        self.assertEqual(start, 5.0)
        self.assertAlmostEqual(duration, 7.0)


if __name__ == "__main__":
    unittest.main()
