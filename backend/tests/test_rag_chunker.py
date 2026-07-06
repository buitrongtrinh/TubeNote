import unittest

from backend.services.rag.chunker import chunk_subtitle_segments


class RagChunkerTests(unittest.TestCase):
    def test_subtitle_chunks_keep_timestamps_and_overlap(self):
        segments = [
            {
                "index": index,
                "start": index * 2.0,
                "end": index * 2.0 + 2.0,
                "text": f"đoạn {index} " + ("x" * 120),
            }
            for index in range(6)
        ]

        docs = chunk_subtitle_segments(
            segments,
            video_id="aaaaaaaaaaa",
            source="subtitles",
            target_chunk_chars=300,
            max_chunk_chars=420,
            overlap_segments=2,
        )

        self.assertGreaterEqual(len(docs), 2)
        self.assertEqual(docs[0].metadata["segment_start"], 0)
        self.assertEqual(docs[0].metadata["segment_end"], 2)
        self.assertEqual(docs[1].metadata["segment_start"], 1)
        self.assertEqual(docs[1].metadata["segment_end"], 3)
        self.assertEqual(docs[0].metadata["start"], 0.0)
        self.assertEqual(docs[0].metadata["end"], 6.0)
        self.assertEqual(docs[0].metadata["rag_version"], "subtitle-v1")
        self.assertTrue(docs[0].page_content.startswith("[00:00-00:06]"))


if __name__ == "__main__":
    unittest.main()
