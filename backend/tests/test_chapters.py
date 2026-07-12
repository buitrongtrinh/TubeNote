from __future__ import annotations

import unittest

from backend.services.dubbing.generate_prompts import (
    CHAPTER_PROMPT_HEADER,
    build_chapter_translation_prompt,
    parse_chapter_translation_response,
)
from backend.services.video.chapters import (
    chapters_from_video_info,
    parse_description_chapters,
)
from backend.services.video.vtt import build_chapter_vtt
from backend.pipeline.dubbing import _apply_chapter_titles, _chapter_titles_for_dubbing


class ChapterMetadataTests(unittest.TestCase):
    def test_prefers_structured_ytdlp_chapters_and_infers_end(self):
        chapters, source = chapters_from_video_info({
            "duration": 90,
            "description": "00:00 Ignored\n00:30 Also ignored",
            "chapters": [
                {"start_time": 0, "end_time": 28, "title": "Intro"},
                {"start_time": 30, "title": "Main topic"},
            ],
        })
        self.assertEqual(source, "yt-dlp")
        self.assertEqual(chapters[0]["start"], 0.0)
        self.assertEqual(chapters[0]["end"], 30.0)
        self.assertEqual(chapters[1]["end"], 90.0)

    def test_parses_timestamped_description_only_when_it_is_a_real_chapter_list(self):
        chapters = parse_description_chapters(
            "Chapters\n00:00 Intro\n01:05 Prompt Engineering\n02:30 Conclusion",
            180,
        )
        self.assertEqual([chapter["title"] for chapter in chapters], [
            "Intro", "Prompt Engineering", "Conclusion",
        ])
        self.assertEqual(chapters[1]["start"], 65.0)
        self.assertEqual(chapters[1]["end"], 150.0)

    def test_ignores_incidental_description_timestamps(self):
        chapters = parse_description_chapters(
            "We mention 01:20 during the explanation.\n02:00 is another example.",
            180,
        )
        self.assertEqual(chapters, [])


class ChapterTranslationTests(unittest.TestCase):
    def setUp(self):
        self.metadata = {
            "title": "A video",
            "channel": "Channel",
            "chapters": [
                {"index": 1, "start": 0, "end": 30, "title": "Intro", "title_vi": None},
                {"index": 2, "start": 30, "end": 60, "title": "Main topic", "title_vi": None},
            ],
        }

    def test_prompt_uses_titles_without_timestamps(self):
        prompt = build_chapter_translation_prompt(self.metadata)
        self.assertIn(CHAPTER_PROMPT_HEADER, prompt)
        self.assertIn("1. Intro", prompt)
        self.assertNotIn("00:00", prompt)

    def test_parser_accepts_exact_numbered_titles(self):
        result = parse_chapter_translation_response(
            f"{CHAPTER_PROMPT_HEADER}\n1. Giới thiệu\n2. Nội dung chính",
            self.metadata,
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["titles"], ["Giới thiệu", "Nội dung chính"])

    def test_parser_rejects_missing_or_reordered_lines(self):
        missing = parse_chapter_translation_response(
            f"{CHAPTER_PROMPT_HEADER}\n1. Giới thiệu",
            self.metadata,
        )
        reordered = parse_chapter_translation_response(
            f"{CHAPTER_PROMPT_HEADER}\n2. Giới thiệu\n1. Nội dung chính",
            self.metadata,
        )
        self.assertFalse(missing["ok"])
        self.assertFalse(reordered["ok"])

    def test_vtt_uses_translated_title_and_normalized_ranges(self):
        vtt = build_chapter_vtt([
            {"start": 0, "end": 30, "title_vi": "Giới thiệu"},
            {"start": 30, "end": 60.5, "title_vi": "Nội dung chính"},
        ])
        self.assertIn("00:00:00.000 --> 00:00:30.000", vtt)
        self.assertIn("Nội dung chính", vtt)

    def test_dubbing_requires_complete_titles_and_only_updates_title_vi(self):
        with self.assertRaises(ValueError):
            _chapter_titles_for_dubbing(self.metadata, ["Giới thiệu"])

        titles = _chapter_titles_for_dubbing(
            self.metadata,
            ["Giới thiệu", "Nội dung chính"],
        )
        _apply_chapter_titles(self.metadata, titles)
        self.assertEqual(self.metadata["chapters"][0]["start"], 0)
        self.assertEqual(self.metadata["chapters"][1]["end"], 60)
        self.assertEqual(self.metadata["chapters"][1]["title_vi"], "Nội dung chính")


if __name__ == "__main__":
    unittest.main()
