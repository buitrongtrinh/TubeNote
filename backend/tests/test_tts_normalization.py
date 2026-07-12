import json
import tempfile
import unittest
from pathlib import Path

from backend.services.dubbing.duration_budget import (
    count_spoken_units,
    estimate_expansion_units,
    natural_duration_seconds,
)
from backend.services.dubbing.generate_prompts import build_batches
from backend.services.dubbing.text_normalizer import (
    apply_pronunciation_map,
    canonicalize_text,
    normalize_for_engine,
    number_to_vietnamese,
)
from backend.services.dubbing.translation_prepare import (
    prepare_translations_for_tts,
    renormalize_segments,
)
from backend.services.dubbing.audio_fit import active_range_samples, fit_to_slot
from backend.services.dubbing.common import save_tts_timings_to_file


class TtsNormalizationTests(unittest.TestCase):
    def test_canonicalize_text_removes_wrapping_quotes(self):
        self.assertEqual(
            canonicalize_text('  “GPT-4”  và  \'RAG\'  là "thuật ngữ".  '),
            "GPT-4 và RAG là thuật ngữ.",
        )
        self.assertEqual(canonicalize_text("don't đổi apostrophe trong từ"), "don't đổi apostrophe trong từ")

    def test_all_caps_acronyms_are_spelled_letter_by_letter(self):
        text, rules = normalize_for_engine("ChatGPT và OpenAI dùng LLM.", "omnivoice")
        self.assertEqual(text, "ChatGPT và OpenAI dùng eo eo em.")
        self.assertEqual(rules, ["acronym"])

    def test_glossary_overrides_win_over_acronym_spelling(self):
        text, rules = normalize_for_engine(
            "VRAM và RAM đọc thành từ.",
            "supertonic",
            glossary={"VRAM": "vi ram", "RAM": "ram"},
        )
        self.assertEqual(text, "vi ram và ram đọc thành từ.")
        self.assertEqual(rules, ["glossary"])

    def test_cli_flags_are_spoken(self):
        text, rules = normalize_for_engine("Chạy --version để kiểm tra.", "supertonic")
        self.assertEqual(text, "Chạy trừ trừ version để kiểm tra.")
        self.assertEqual(rules, ["cli_flag"])

    def test_single_dash_short_flag_spells_each_letter(self):
        text, rules = normalize_for_engine("docker ps -a để xem.", "supertonic")
        self.assertEqual(text, "docker ps trừ ei để xem.")
        self.assertEqual(rules, ["cli_flag"])

    def test_single_dash_combined_flags_spell_all_letters(self):
        text, rules = normalize_for_engine("ls -la liệt kê file.", "supertonic")
        self.assertEqual(text, "ls trừ eo ei liệt kê file.")
        self.assertEqual(rules, ["cli_flag"])

    def test_single_dash_flag_before_punctuation_still_spoken(self):
        """Flag đứng ngay trước dấu câu (không có khoảng trắng) vẫn phải được
        đọc — regression: lookahead (?!\\S) cũ làm rule không khớp trường hợp
        này, để nguyên "-d." không đọc."""
        text, rules = normalize_for_engine("thêm cờ -d.", "supertonic")
        self.assertEqual(text, "thêm cờ trừ đi.")
        self.assertEqual(rules, ["cli_flag"])

    def test_indexed_symbol_reads_letter_and_digit(self):
        text, _ = normalize_for_engine("Ô A1 chứa kết quả.", "supertonic")
        self.assertEqual(text, "Ô a một chứa kết quả.")

    def test_leading_slash_path_reads_slash_as_siet(self):
        text, rules = normalize_for_engine("Chép vào /app/src rồi chạy.", "supertonic")
        self.assertEqual(text, "Chép vào siệt app siệt src rồi chạy.")
        self.assertIn("path", rules)

    def test_dotted_name_reads_dot_as_cham(self):
        text, rules = normalize_for_engine("Mở file abc.cde và example.com.", "supertonic")
        self.assertEqual(text, "Mở file abc chấm cde và example chấm com.")
        self.assertIn("path", rules)

    def test_path_with_extension_reads_both_separators(self):
        text, _ = normalize_for_engine("Sửa /app/config.yaml đi.", "supertonic")
        self.assertEqual(text, "Sửa siệt app siệt config chấm yaml đi.")

    def test_lone_slash_between_words_is_kept(self):
        # "nam/nữ", "và/hoặc" nghĩa là "hoặc", không phải path → giữ nguyên.
        text, rules = normalize_for_engine("Chọn nam/nữ tùy ý.", "supertonic")
        self.assertEqual(text, "Chọn nam/nữ tùy ý.")
        self.assertNotIn("path", rules)

    def test_decimal_dot_stays_a_number_not_cham(self):
        text, rules = normalize_for_engine("Số pi là 3.14 nhé.", "supertonic")
        self.assertEqual(text, "Số pi là ba phẩy một bốn nhé.")
        self.assertNotIn("path", rules)

    def test_url_scheme_is_left_untouched(self):
        text, rules = normalize_for_engine("Xem tại https://example.com trang chủ.", "supertonic")
        self.assertEqual(text, "Xem tại https://example.com trang chủ.")
        self.assertNotIn("path", rules)

    def test_tagged_version_reads_name_and_number(self):
        text, rules = normalize_for_engine("Cài node:22-alpine để build.", "supertonic")
        self.assertEqual(text, "Cài node hai mươi hai alpine để build.")
        self.assertIn("tagged_version", rules)

    def test_physical_units_are_spoken(self):
        text, _ = normalize_for_engine("Nặng 5 kg và chạy 10 km.", "supertonic")
        self.assertEqual(text, "Nặng năm ki lô gam và chạy mười ki lô mét.")

    def test_pronunciation_map_overrides_whole_terms_case_insensitively(self):
        text, mapping = apply_pronunciation_map(
            "RAG dùng RAGFlow, sau đó rag được chạy lại.",
            {"RAG": "Rác"},
        )
        self.assertEqual(text, "Rác dùng RAGFlow, sau đó Rác được chạy lại.")
        self.assertEqual(mapping, {"RAG": "Rác"})

    def test_omnivoice_normalizes_contextual_numbers(self):
        text, rules = normalize_for_engine(
            "GPT-4 chạy trên RTX 4090 với 16 GB VRAM.",
            "omnivoice",
        )
        self.assertEqual(
            text,
            "ji pi ti bốn chạy trên a ti ek bốn không chín không "
            "với mười sáu gi ga bai vi a ei em.",
        )
        self.assertIn("model_code", rules)
        self.assertIn("measurement", rules)
        self.assertIn("acronym", rules)

    def test_math_is_spoken_in_vietnamese(self):
        text, _ = normalize_for_engine("f(x) = W1 * x + 3.14?", "omnivoice")
        self.assertEqual(
            text,
            "ép của ích bằng đúp liu một nhân ích cộng ba phẩy một bốn?",
        )

    def test_mixed_case_terms_stay_verbatim(self):
        text, _ = normalize_for_engine("ChatGPT, OpenAI, C++ và LLM.", "supertonic")
        self.assertEqual(
            text,
            "ChatGPT, OpenAI, C++ và eo eo em.",
        )

    def test_vietnamese_cardinal_numbers(self):
        self.assertEqual(number_to_vietnamese(2024), "hai nghìn không trăm hai mươi tư")
        self.assertEqual(number_to_vietnamese(1250), "một nghìn hai trăm năm mươi")

    def test_validation_expands_acronyms_in_tts_only(self):
        segments = prepare_translations_for_tts(
            "[batch_1]\n1. GPT-4 hỗ trợ CLI và API.",
            "batch_1",
            engine="supertonic",
        )
        self.assertEqual(segments[0]["vi"], "GPT-4 hỗ trợ CLI và API.")
        self.assertEqual(segments[0]["tts"], "ji pi ti bốn hỗ trợ si eo ai và ei pi ai.")
        self.assertIn("acronym", segments[0]["normalization"]["applied_rules"])
        self.assertEqual(segments[0]["normalization"]["warnings"], [])
        self.assertIsNone(segments[0]["normalization"]["budget"])

    def test_validation_keeps_display_text_clean_while_tts_expands(self):
        segments = prepare_translations_for_tts(
            "[batch_1]\n1. RAG giúp tìm đúng ngữ cảnh.",
            "batch_1",
            engine="omnivoice",
        )
        self.assertEqual(segments[0]["vi"], "RAG giúp tìm đúng ngữ cảnh.")
        self.assertEqual(segments[0]["tts"], "a ei ji giúp tìm đúng ngữ cảnh.")
        self.assertNotIn("pronunciation_map", segments[0])

    def test_builtin_glossary_reads_word_acronyms_as_words(self):
        segments = prepare_translations_for_tts(
            "[batch_1]\n1. RAM và NASA vẫn đọc thành từ.",
            "batch_1",
            engine="supertonic",
        )
        self.assertEqual(segments[0]["tts"], "ram và na sa vẫn đọc thành từ.")
        self.assertIn("glossary", segments[0]["normalization"]["applied_rules"])

    def test_dubbing_renormalization_expands_acronyms(self):
        segments = renormalize_segments(
            [{"vi": "CLI gọi API để chạy GPT."}],
            "omnivoice",
            budgets=None,
        )
        self.assertEqual(segments[0]["tts"], "si eo ai gọi ei pi ai để chạy ji pi ti.")
        self.assertEqual(segments[0]["normalization"]["applied_rules"], ["acronym"])
        self.assertEqual(segments[0]["normalization"]["warnings"], [])

    def test_dubbing_renormalization_keeps_user_pronunciation_map_for_regenerate_paths(self):
        segments = renormalize_segments(
            [{"vi": "Kircer vẫn giữ nguyên trong phụ đề."}],
            "supertonic",
            pronunciation_map={"Kircer": "kơ sờ"},
        )
        self.assertEqual(segments[0]["vi"], "Kircer vẫn giữ nguyên trong phụ đề.")
        self.assertEqual(segments[0]["tts"], "kơ sờ vẫn giữ nguyên trong phụ đề.")
        self.assertEqual(segments[0]["normalization"]["applied_rules"], ["pronunciation_map"])

    def test_omnivoice_budget_allows_under_max_without_warning(self):
        segment = renormalize_segments(
            [{"vi": "Đây là một câu dài hơn mục tiêu nhưng vẫn chưa vượt quá tối đa."}],
            "omnivoice",
            budgets=[18],
        )[0]
        metadata = segment["normalization"]
        self.assertEqual(segment["tts"], "Đây là một câu dài hơn mục tiêu nhưng vẫn chưa vượt quá tối đa.")
        self.assertEqual(metadata["min_units"], 9)
        self.assertEqual(metadata["target_units"], 14)
        self.assertEqual(metadata["base_max_units"], 15)
        self.assertEqual(metadata["tolerance_units"], 6)
        self.assertEqual(metadata["max_units"], 21)
        self.assertEqual(metadata["errors"], [])
        self.assertEqual(metadata["warnings"], [])

    def test_omnivoice_budget_warns_on_extreme_density(self):
        segment = renormalize_segments(
            [{"vi": "Gửi lời cảm ơn nhanh tới Kircer, lát nữa sẽ nói thêm nữa nhé."}],
            "omnivoice",
            budgets=[11],
        )[0]
        metadata = segment["normalization"]
        # "Kircer" (tên riêng, không có trong glossary) đếm 2 âm tiết qua
        # heuristic cụm nguyên âm thay vì tính cứng 1 — sát thực tế hơn.
        self.assertEqual(metadata["written_units"], 15)
        self.assertEqual(metadata["spoken_units"], 15)
        self.assertEqual(metadata["max_units"], 13)
        self.assertEqual(metadata["errors"], [])
        self.assertTrue(metadata["warnings"])

    def test_number_expansion_is_compensated_in_budget(self):
        segment = renormalize_segments(
            [{"vi": "Không lâu sau khi ChatGPT ra mắt năm 2022,"}],
            "omnivoice",
            budgets=[17],
        )[0]
        metadata = segment["normalization"]
        self.assertEqual(
            segment["tts"],
            "Không lâu sau khi ChatGPT ra mắt năm hai nghìn không trăm hai mươi hai,",
        )
        # 2022 (1 tiếng viết) → 7 tiếng đọc: expansion 6 được bù vào budget nên
        # câu không bị chặn oan dù spoken_units tăng.
        self.assertEqual(metadata["normalization_expansion"], 6)
        self.assertEqual(metadata["budget"], 17)
        self.assertEqual(metadata["errors"], [])
        self.assertEqual(metadata["warnings"], [])

    def test_supertonic_uses_same_density_gate_as_omnivoice(self):
        segment = renormalize_segments(
            [{"vi": "Gửi lời cảm ơn nhanh tới Kircer, lát nữa sẽ nói thêm."}],
            "supertonic",
            budgets=[11],
        )[0]
        metadata = segment["normalization"]
        # "Kircer" đếm 2 âm tiết qua heuristic cụm nguyên âm (xem test ở trên).
        self.assertEqual(metadata["written_units"], 13)
        self.assertEqual(metadata["spoken_units"], 13)
        self.assertEqual(metadata["budget_tolerance"], 0)
        self.assertIsNone(metadata["allowed_units"])
        self.assertEqual(metadata["warnings"], [])
        self.assertEqual(metadata["max_units"], 13)
        self.assertEqual(metadata["errors"], [])

    def test_count_spoken_units_weighs_foreign_words_by_vowel_clusters(self):
        # "container"/"machine" (đa âm tiết, không có trong glossary) không còn
        # bị đếm cứng 1 như từ tiếng Việt đơn âm.
        self.assertEqual(count_spoken_units("container"), 3)
        self.assertEqual(count_spoken_units("machine"), 3)
        base = count_spoken_units("Video này nói về đó.")  # "đó" = 1 âm tiết
        swapped = count_spoken_units("Video này nói về container.")
        self.assertEqual(swapped - base, 2)  # container(3) thay đó(1) → dư 2

    def test_count_spoken_units_keeps_vietnamese_words_at_one_unit_each(self):
        self.assertEqual(count_spoken_units("Xin chào các bạn"), 4)

    def test_natural_duration_seconds_matches_target_rate(self):
        from backend.services.dubbing.duration_budget import TtsDensityPolicy

        policy = TtsDensityPolicy(target_units_per_sec=4.5)
        text = "Xin chào các bạn"  # 4 âm tiết
        self.assertAlmostEqual(natural_duration_seconds(text, policy), 4 / 4.5)

    def test_expansion_surcharge_estimated_from_source_text(self):
        # "16 GB" (2 viết → 5 đọc) + "LLM" (1 viết → 3 đọc) = surcharge 5.
        self.assertEqual(estimate_expansion_units("The LLM needs 16 GB"), 5)
        self.assertEqual(estimate_expansion_units("No expandable tokens here"), 0)

    def test_prompt_budget_is_reduced_by_expansion_surcharge(self):
        segments = [
            {"text": "It needs 16 GB", "start": 0.0, "duration": 4.0},
            {"text": "Plain sentence here", "start": 4.0, "duration": 4.0},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "segments.json"
            path.write_text(json.dumps(segments), encoding="utf-8")
            batch = build_batches(str(path), batch_size=10)[0]
        # duration 4.0 → budget gốc 24; "16 GB" surcharge 3 → còn 21.
        self.assertIn("1. [≤21 tiếng] It needs 16 GB", batch)
        self.assertIn("2. [≤24 tiếng] Plain sentence here", batch)

    def test_fit_to_slot_preserves_slot_length_and_warns_when_compressed(self):
        import numpy as np

        sample_rate = 24000
        duration = 0.8
        slot_duration = 0.3
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        wav = (0.2 * np.sin(2 * np.pi * 440 * t)).astype("float32")
        fitted, speech_len, meta = fit_to_slot(wav, int(slot_duration * sample_rate), sample_rate)
        self.assertEqual(len(fitted), int(slot_duration * sample_rate))
        self.assertLessEqual(speech_len, len(fitted))
        self.assertGreater(meta["fit_ratio"], 1.75)
        self.assertTrue(meta["warnings"])

    def test_active_range_detects_leading_and_trailing_silence(self):
        import numpy as np

        wav = np.concatenate([
            np.zeros(1000, dtype=np.float32),
            np.ones(2000, dtype=np.float32) * 0.2,
            np.zeros(1000, dtype=np.float32),
        ])
        start, end = active_range_samples(wav, 1000, frame_ms=20, head_ms=0, tail_ms=0)
        self.assertEqual(start, 1000)
        self.assertEqual(end, 3000)

    def test_supertonic_subtitle_timing_uses_active_voice_range(self):
        segments = [
            {"start": 0.0, "duration": 5.0, "text": "Hello", "text_vi": "Xin chào", "text_tts": "Xin chào"},
            {"start": 5.0, "duration": 3.0, "text": "Next", "text_vi": "Tiếp theo", "text_tts": "Tiếp theo"},
        ]
        timings = [{
            "source_indices": [0],
            "start": 1.0,
            "speech_duration": 2.0,
        }]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sub.json"
            path.write_text(json.dumps(segments), encoding="utf-8")
            save_tts_timings_to_file(
                timings,
                str(path),
                {"engine": "supertonic", "model": "M5"},
            )
            data = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(data[0]["tts"]["start"], 1.0)
        self.assertEqual(data[0]["tts"]["end"], 3.0)
        self.assertEqual(data[0]["tts"]["duration"], 2.0)

    def test_omnivoice_without_budget_never_reports_length_warning(self):
        segment = renormalize_segments(
            [{"vi": "Đây là một câu dịch dài hơn nhiều nhưng OmniVoice tự điều khiển duration."}],
            "omnivoice",
            budgets=None,
        )[0]
        self.assertIsNone(segment["normalization"]["budget"])
        self.assertEqual(segment["normalization"]["warnings"], [])


if __name__ == "__main__":
    unittest.main()
