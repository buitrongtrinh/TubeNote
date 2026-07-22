"""Dựng lại câu từ cue caption (manual sub) qua pseudo-words.

Cue caption chia theo dòng hiển thị, không theo câu — kiểm tra việc chẻ cue
thành pseudo word-timestamps rồi đi qua cùng bộ máy ``_split_into_entries``
với đường Whisper cho ra câu hoàn chỉnh đúng ranh giới dấu câu/khoảng lặng.
"""
import unittest

from backend.services.youtube.transcript_yt_dlp import (
    _clean_transcript,
    _cues_to_pseudo_words,
)


def _event(text: str, start_ms: int, duration_ms: int) -> dict:
    return {
        "tStartMs": start_ms,
        "dDurationMs": duration_ms,
        "segs": [{"utf8": text}],
    }


class CuesToPseudoWordsTests(unittest.TestCase):
    def test_words_are_contiguous_within_a_cue(self):
        words = _cues_to_pseudo_words([
            {"text": "hello there friend", "start": 1.0, "duration": 3.0},
        ])
        self.assertEqual([w["word"] for w in words], ["hello", "there", "friend"])
        self.assertEqual(words[0]["start"], 1.0)
        self.assertEqual(words[-1]["end"], 4.0)
        # nối đuôi: end từ trước == start từ sau -> gap 0, không sinh điểm cắt giả
        for prev, cur in zip(words, words[1:]):
            self.assertEqual(prev["end"], cur["start"])

    def test_gap_between_cues_is_preserved(self):
        words = _cues_to_pseudo_words([
            {"text": "first cue", "start": 0.0, "duration": 2.0},
            {"text": "second cue", "start": 5.0, "duration": 2.0},
        ])
        self.assertEqual(words[1]["end"], 2.0)     # hết cue 1
        self.assertEqual(words[2]["start"], 5.0)   # cue 2 giữ nguyên mốc thật

    def test_longer_words_get_more_time(self):
        words = _cues_to_pseudo_words([
            {"text": "a extraordinarily", "start": 0.0, "duration": 2.0},
        ])
        short = words[0]["end"] - words[0]["start"]
        long = words[1]["end"] - words[1]["start"]
        self.assertLess(short, long)


class CleanTranscriptReflowTests(unittest.TestCase):
    def test_fragmented_cues_of_one_sentence_merge(self):
        # 1 câu bị bổ thành 3 cue sát nhau (gap < caption pause alpha 1.0s)
        trans = _clean_transcript([
            _event("We made this film", 0, 1500),
            _event("because we were talking", 1600, 1500),
            _event("about architecture.", 3200, 1500),
        ])
        texts = [e.text for e in trans.segments]
        self.assertEqual(
            texts,
            ["We made this film because we were talking about architecture."],
        )

    def test_mid_cue_punctuation_splits_sentences(self):
        # dấu chấm nằm GIỮA cue -> vẫn cắt đúng ranh giới câu
        trans = _clean_transcript([
            _event("and that is done. Now we", 0, 3000),
            _event("start the next part here.", 3100, 2500),
        ])
        texts = [e.text for e in trans.segments]
        self.assertEqual(
            texts,
            ["and that is done.", "Now we start the next part here."],
        )

    def test_long_silence_between_cues_splits(self):
        # im lặng dài (>= 1.0s) giữa 2 cue không có dấu kết câu -> vẫn tách
        trans = _clean_transcript([
            _event("first thought trails off", 0, 2000),
            _event("second thought starts anew", 5000, 2000),
        ])
        self.assertEqual(len(trans.segments), 2)
        self.assertEqual(trans.segments[1].start, 5.0)

    def test_empty_events_return_none(self):
        self.assertIsNone(_clean_transcript([]))
        self.assertIsNone(_clean_transcript([{"tStartMs": 0, "dDurationMs": 100}]))


if __name__ == "__main__":
    unittest.main()
