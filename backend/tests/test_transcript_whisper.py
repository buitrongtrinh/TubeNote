import unittest
from types import SimpleNamespace

from backend.services.youtube.transcript_whisper import (
    _faster_segments_to_transcript_with_progress,
    _find_balanced_cut,
    _looks_like_number_continuation,
    _merge_short_ranges,
    _split_into_entries,
)


def _w(word: str, start: float, end: float) -> dict:
    return {"word": word, "start": start, "end": end}


def _contiguous_words(n: int) -> list[dict]:
    """N từ liên tiếp không có khoảng nghỉ (end của từ i == start của từ i+1),
    dùng để test riêng logic max_words/split_long_run mà không bị outer loop
    tách theo gap trước."""
    return [_w(f"w{i}", i * 0.3, (i + 1) * 0.3) for i in range(n)]


class SplitIntoEntriesTests(unittest.TestCase):
    def test_empty_words_returns_empty(self):
        self.assertEqual(_split_into_entries([], 16, 0.02), [])

    def test_single_sentence_stays_one_entry(self):
        words = [_w("Hello", 0.0, 0.3), _w("world.", 0.3, 0.6)]
        entries = _split_into_entries(words, 16, 0.02)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].text, "Hello world.")
        self.assertEqual(entries[0].start, 0.0)
        self.assertAlmostEqual(entries[0].duration, 0.6)

    def test_sentence_end_punctuation_splits_even_without_gap(self):
        words = [
            _w("Hello", 0.0, 0.3),
            _w("there.", 0.3, 0.6),
            _w("How", 0.6, 0.8),
            _w("are", 0.8, 0.9),
            _w("you?", 0.9, 1.1),
        ]
        entries = _split_into_entries(words, 16, 0.02)
        self.assertEqual([e.text for e in entries], ["Hello there.", "How are you?"])
        self.assertEqual(entries[1].start, 0.6)
        self.assertAlmostEqual(entries[1].duration, 0.5)

    def test_pause_gap_splits_without_any_punctuation(self):
        words = [_w("Yes", 0.0, 0.2), _w("okay", 0.5, 0.7)]
        entries = _split_into_entries(words, 16, 0.02)
        self.assertEqual([e.text for e in entries], ["Yes", "okay"])
        self.assertEqual(entries[0].start, 0.0)
        self.assertAlmostEqual(entries[0].duration, 0.2)
        self.assertEqual(entries[1].start, 0.5)

    def test_long_run_no_punctuation_hard_cuts_at_max_words(self):
        words = _contiguous_words(10)
        entries = _split_into_entries(words, 6, 0.02)
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0].text, "w0 w1 w2 w3 w4 w5")
        self.assertEqual(entries[1].text, "w6 w7 w8 w9")

    def test_hard_cut_avoids_splitting_a_number_fragment(self):
        words = _contiguous_words(10)
        words[6]["word"] = ",000"  # simulates "4,000" tokenized as "4" + ",000"
        entries = _split_into_entries(words, 6, 0.02)
        self.assertEqual(len(entries), 2)
        # boundary nudged past the number fragment instead of landing on it
        self.assertTrue(entries[0].text.endswith(",000"))
        self.assertNotEqual(entries[1].text[:1], ",")

    def test_long_run_prefers_balanced_comma_split(self):
        words = _contiguous_words(14)
        words[4]["word"] = "w4,"
        words[9]["word"] = "w9,"
        entries = _split_into_entries(words, 8, 0.02)
        self.assertEqual(len(entries), 3)
        self.assertEqual(entries[0].text, "w0 w1 w2 w3 w4,")
        self.assertEqual(entries[1].text, "w5 w6 w7 w8 w9,")
        self.assertEqual(entries[2].text, "w10 w11 w12 w13")

    def test_conjunction_used_as_fallback_split_point(self):
        words = _contiguous_words(10)
        words[5]["word"] = "because"
        entries = _split_into_entries(words, 6, 0.02)
        self.assertEqual(len(entries), 2)
        self.assertTrue(entries[0].text.endswith("because"))

    def test_max_words_zero_disables_word_count_splitting(self):
        words = _contiguous_words(30)
        words[10]["word"] = "w10,"  # a comma is present but should be ignored
        entries = _split_into_entries(words, 0, 0.02)
        self.assertEqual(len(entries), 1)
        self.assertEqual(len(entries[0].text.split()), 30)

    def test_negative_max_words_also_disables_splitting(self):
        words = _contiguous_words(20)
        entries = _split_into_entries(words, -1, 0.02)
        self.assertEqual(len(entries), 1)


class MergeShortRangesTests(unittest.TestCase):
    def test_isolated_short_range_merges_into_nearer_previous_neighbor(self):
        words = [
            _w("Alpha", 0.0, 0.3), _w("beta", 0.3, 0.6), _w("gamma.", 0.6, 0.9),
            _w("Solo", 1.0, 1.2),   # gap to previous = 0.1 (closer)
            _w("Next", 1.5, 1.8), _w("door", 1.8, 2.1), _w("closes.", 2.1, 2.4),
            # gap Solo -> Next = 0.3 (farther)
        ]
        entries = _split_into_entries(words, 16, 0.02, min_words=2)
        self.assertEqual(
            [e.text for e in entries],
            ["Alpha beta gamma. Solo", "Next door closes."],
        )

    def test_isolated_short_range_merges_into_nearer_next_neighbor(self):
        words = [
            _w("Alpha", 0.0, 0.3), _w("beta", 0.3, 0.6), _w("gamma.", 0.6, 0.9),
            _w("Solo", 1.2, 1.4),   # gap to previous = 0.3 (farther)
            _w("Next", 1.5, 1.8), _w("door", 1.8, 2.1), _w("closes.", 2.1, 2.4),
            # gap Solo -> Next = 0.1 (closer)
        ]
        entries = _split_into_entries(words, 16, 0.02, min_words=2)
        self.assertEqual(
            [e.text for e in entries],
            ["Alpha beta gamma.", "Solo Next door closes."],
        )

    def test_leading_short_range_merges_forward_when_no_previous_neighbor(self):
        words = [
            _w("First.", 0.0, 0.3),
            _w("Yes", 1.0, 1.2),
            _w("Second", 1.25, 1.5), _w("thing", 1.5, 1.8), _w("here.", 1.8, 2.1),
        ]
        entries = _split_into_entries(words, 16, 0.02, min_words=2)
        self.assertEqual([e.text for e in entries], ["First. Yes", "Second thing here."])

    def test_min_words_one_disables_merging(self):
        words = [_w("Yes", 0.0, 0.2), _w("okay", 0.5, 0.7)]
        entries = _split_into_entries(words, 16, 0.02, min_words=1)
        self.assertEqual([e.text for e in entries], ["Yes", "okay"])

    def test_single_range_returned_unchanged(self):
        self.assertEqual(_merge_short_ranges([], [(0, 1)], 2), [(0, 1)])

    def test_no_ranges_returned_unchanged(self):
        self.assertEqual(_merge_short_ranges([], [], 2), [])


class FindBalancedCutTests(unittest.TestCase):
    def test_no_candidates_returns_negative_one(self):
        self.assertEqual(_find_balanced_cut([], 0, 10), -1)

    def test_picks_candidate_closest_to_middle(self):
        self.assertEqual(_find_balanced_cut([2, 5, 8], 0, 10), 5)


class NumberContinuationTests(unittest.TestCase):
    def test_detects_comma_digit_fragment(self):
        self.assertTrue(_looks_like_number_continuation({"word": ",000"}))
        self.assertTrue(_looks_like_number_continuation({"word": ".5kg"}))

    def test_ignores_normal_words(self):
        self.assertFalse(_looks_like_number_continuation({"word": "tokens,"}))
        self.assertFalse(_looks_like_number_continuation({"word": "4,000"}))


class FasterSegmentsToTranscriptTests(unittest.TestCase):
    def _word(self, word: str, start: float, end: float):
        return SimpleNamespace(word=word, start=start, end=end)

    def _segment(self, text: str, start: float, end: float, words=None):
        return SimpleNamespace(text=text, start=start, end=end, words=words)

    def test_segment_with_words_gets_split_and_tightened(self):
        segments = [
            self._segment(
                "Hello there. How are you?",
                -0.2,
                1.3,
                words=[
                    self._word("Hello", 0.0, 0.3),
                    self._word("there.", 0.3, 0.6),
                    self._word("How", 0.6, 0.8),
                    self._word("are", 0.8, 0.9),
                    self._word("you?", 0.9, 1.1),
                ],
            )
        ]
        trans = _faster_segments_to_transcript_with_progress(
            segments, duration=1.3, sentence_max_words=16, sentence_pause_alpha=0.02,
        )
        texts = [s.text for s in trans.segments]
        self.assertEqual(texts, ["Hello there.", "How are you?"])
        # boundaries tightened to real word edges, not the raw (padded) segment start/end
        self.assertEqual(trans.segments[0].start, 0.0)

    def test_segment_without_words_falls_back_to_whole_segment(self):
        segments = [
            self._segment("No word timestamps here.", 0.0, 2.0, words=None),
        ]
        trans = _faster_segments_to_transcript_with_progress(
            segments, duration=2.0, sentence_max_words=16, sentence_pause_alpha=0.02,
        )
        self.assertEqual(len(trans.segments), 1)
        self.assertEqual(trans.segments[0].text, "No word timestamps here.")
        self.assertEqual(trans.segments[0].start, 0.0)
        self.assertAlmostEqual(trans.segments[0].duration, 2.0)

    def test_mixed_segments_interleave_in_order(self):
        segments = [
            self._segment(
                "First one.",
                0.0,
                1.0,
                words=[self._word("First", 0.0, 0.4), self._word("one.", 0.4, 0.8)],
            ),
            self._segment("Fallback segment text.", 1.5, 3.0, words=None),
            self._segment(
                "Last one.",
                3.5,
                4.5,
                words=[self._word("Last", 3.5, 3.9), self._word("one.", 3.9, 4.2)],
            ),
        ]
        trans = _faster_segments_to_transcript_with_progress(
            segments, duration=4.5, sentence_max_words=16, sentence_pause_alpha=0.02,
        )
        texts = [s.text for s in trans.segments]
        self.assertEqual(texts, ["First one.", "Fallback segment text.", "Last one."])

    def test_no_segments_returns_none(self):
        trans = _faster_segments_to_transcript_with_progress([], duration=0.0)
        self.assertIsNone(trans)


if __name__ == "__main__":
    unittest.main()
