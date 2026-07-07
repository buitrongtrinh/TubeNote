import unittest

from backend.config import CFG, HardwareCfg
from backend.services.hardware import (
    hardware_availability,
    hardware_report,
    pick_batch_size_for_vram,
    pick_tier,
    recommend_setup,
)

TEST_CFG = HardwareCfg(
    omnivoice_min_vram_gb=5.0,
    omnivoice_batch_by_vram={5.0: 4, 12.0: 6, 16.0: 8},
    asr_gpu_by_vram={2.0: "gpu_small", 3.5: "gpu"},
    asr_cpu_by_ram={0.0: "cpu_tiny", 4.0: "cpu_base", 6.0: "cpu"},
    max_auto_threads=16,
)


class PickTierTests(unittest.TestCase):
    TABLE = {2.0: "a", 3.5: "b"}

    def test_none_value_returns_none(self):
        self.assertIsNone(pick_tier(None, self.TABLE))

    def test_below_lowest_returns_none(self):
        self.assertIsNone(pick_tier(1.9, self.TABLE))

    def test_picks_largest_threshold_reached(self):
        self.assertEqual(pick_tier(2.0, self.TABLE), "a")
        self.assertEqual(pick_tier(3.4, self.TABLE), "a")
        self.assertEqual(pick_tier(3.5, self.TABLE), "b")
        self.assertEqual(pick_tier(24.0, self.TABLE), "b")

    def test_empty_table_returns_none(self):
        self.assertIsNone(pick_tier(10.0, {}))


class PickBatchSizeTests(unittest.TestCase):
    TABLE = {5.0: 4, 12.0: 6, 16.0: 8}

    def test_no_gpu_returns_one(self):
        self.assertEqual(pick_batch_size_for_vram(None, self.TABLE), 1)

    def test_below_lowest_threshold_returns_one(self):
        self.assertEqual(pick_batch_size_for_vram(4.0, self.TABLE), 1)

    def test_picks_largest_threshold_reached(self):
        self.assertEqual(pick_batch_size_for_vram(5.6, self.TABLE), 4)
        self.assertEqual(pick_batch_size_for_vram(12.0, self.TABLE), 6)
        self.assertEqual(pick_batch_size_for_vram(24.0, self.TABLE), 8)


class RecommendSetupTests(unittest.TestCase):
    def test_no_gpu_uses_cpu_asr_and_supertonic(self):
        rec = recommend_setup(8, 0, 8, cfg=TEST_CFG)
        self.assertEqual(rec["asr_preset"], "cpu")
        self.assertEqual(rec["tts_engine"], "supertonic")
        self.assertEqual(rec["omnivoice_batch_size"], 0)

    def test_low_ram_machines_get_smaller_cpu_models(self):
        self.assertEqual(recommend_setup(3, 0, 4, cfg=TEST_CFG)["asr_preset"], "cpu_tiny")
        self.assertEqual(recommend_setup(5, 0, 4, cfg=TEST_CFG)["asr_preset"], "cpu_base")

    def test_small_vram_gets_gpu_asr_but_supertonic_tts(self):
        rec = recommend_setup(16, 3, 8, cfg=TEST_CFG)
        self.assertEqual(rec["asr_preset"], "gpu_small")
        self.assertEqual(rec["tts_engine"], "supertonic")

    def test_real_machine_rtx4050(self):
        rec = recommend_setup(15.3, 5.6, 12, cfg=TEST_CFG)
        self.assertEqual(rec["asr_preset"], "gpu")
        self.assertEqual(rec["tts_engine"], "omnivoice")
        self.assertEqual(rec["omnivoice_batch_size"], 4)

    def test_big_gpu_gets_bigger_batch(self):
        rec = recommend_setup(32, 16, 24, cfg=TEST_CFG)
        self.assertEqual(rec["tts_engine"], "omnivoice")
        self.assertEqual(rec["omnivoice_batch_size"], 8)

    def test_threads_clamped_to_max_auto(self):
        rec = recommend_setup(32, 16, 64, cfg=TEST_CFG)
        self.assertEqual(rec["whisper_cpu_threads"], 16)
        self.assertEqual(rec["supertonic_intra_op_threads"], 16)

    def test_notes_explain_choices(self):
        rec = recommend_setup(8, 0, 8, cfg=TEST_CFG)
        self.assertTrue(rec["notes"])

    def test_unknown_preset_in_table_falls_back_to_default(self):
        broken = HardwareCfg(
            omnivoice_min_vram_gb=5.0,
            omnivoice_batch_by_vram={5.0: 4},
            asr_gpu_by_vram={2.0: "khong_ton_tai"},
            asr_cpu_by_ram={0.0: "cung_khong"},
            max_auto_threads=16,
        )
        rec = recommend_setup(8, 8, 8, cfg=broken)
        self.assertIn(rec["asr_preset"], CFG.whisper.presets)


class HardwareAvailabilityTests(unittest.TestCase):
    def test_no_gpu_disables_gpu_options(self):
        availability = hardware_availability(8, 0, cfg=TEST_CFG)
        self.assertTrue(availability["tts"]["supertonic"]["available"])
        self.assertFalse(availability["tts"]["omnivoice"]["available"])
        self.assertFalse(availability["asr"]["gpu_small"]["available"])
        self.assertFalse(availability["asr"]["gpu"]["available"])

    def test_small_gpu_allows_gpu_asr_but_not_omnivoice(self):
        availability = hardware_availability(8, 3, cfg=TEST_CFG)
        self.assertTrue(availability["asr"]["gpu_small"]["available"])
        self.assertFalse(availability["asr"]["gpu"]["available"])
        self.assertFalse(availability["tts"]["omnivoice"]["available"])

    def test_enough_vram_allows_omnivoice(self):
        availability = hardware_availability(8, 5, cfg=TEST_CFG)
        self.assertTrue(availability["asr"]["gpu"]["available"])
        self.assertTrue(availability["tts"]["omnivoice"]["available"])

    def test_known_low_ram_disables_larger_cpu_auto_presets(self):
        availability = hardware_availability(3, 0, cfg=TEST_CFG)
        self.assertTrue(availability["asr"]["cpu_tiny"]["available"])
        self.assertFalse(availability["asr"]["cpu_base"]["available"])
        self.assertFalse(availability["asr"]["cpu"]["available"])


class HardwareReportTests(unittest.TestCase):
    def test_report_shape(self):
        report = hardware_report()
        self.assertIn("detected", report)
        self.assertIn("cpu_cores", report["detected"])
        rec = report["recommendation"]
        self.assertIn(rec["asr_preset"], CFG.whisper.presets)
        self.assertIn(rec["tts_engine"], ("supertonic", "omnivoice"))
        self.assertGreaterEqual(rec["whisper_cpu_threads"], 1)


class ResolveTtsBatchOverrideTests(unittest.TestCase):
    """UI gửi batch tính từ VRAM người dùng NHẬP TAY -> phải thắng config."""

    def test_omnivoice_payload_batch_size_wins(self):
        from backend.pipeline.dubbing import resolve_tts_config

        cfg = resolve_tts_config({"engine": "omnivoice", "batch_size": 6})
        self.assertEqual(cfg["batch_size"], 6)

    def test_missing_batch_size_falls_back_to_config_auto(self):
        from backend.pipeline.dubbing import TTS_POLICIES, resolve_tts_config

        cfg = resolve_tts_config({"engine": "omnivoice"})
        self.assertEqual(cfg["batch_size"], TTS_POLICIES["omnivoice"]["batch_size"])


class ConfigTierTablesTests(unittest.TestCase):
    def test_yaml_tables_parsed(self):
        hw = CFG.hardware
        self.assertTrue(hw.asr_gpu_by_vram)
        self.assertTrue(hw.asr_cpu_by_ram)
        # Mọi preset trong bảng phải tồn tại thật trong whisper.presets
        for preset in list(hw.asr_gpu_by_vram.values()) + list(hw.asr_cpu_by_ram.values()):
            self.assertIn(preset, CFG.whisper.presets)


if __name__ == "__main__":
    unittest.main()
