import json
import math
import os
import tempfile
import unittest
from datetime import datetime, timezone

from tidal.intervals import above_threshold_intervals, sample_window
from tidal.model import fit_model, load_model

EPOCH = datetime(2024, 1, 1, tzinfo=timezone.utc)
W1 = 0.5   # rad/hour
W2 = 1.1   # rad/hour


def true_level(t):
    return 1.2 + 0.8 * math.cos(W1 * t - 0.3) + 0.4 * math.cos(W2 * t - 2.0)


class FitTests(unittest.TestCase):
    def test_irregular_sampling_recovers_coefficients(self):
        # Non-uniform observation times (gaps and jitter).
        times = [0.0, 0.7, 1.9, 2.2, 4.5, 6.1, 7.8, 9.0, 11.3, 13.6,
                 15.0, 18.2, 21.7, 25.4, 30.1]
        values = [true_level(t) for t in times]
        model = fit_model(EPOCH, times, values, [W1, W2])
        self.assertAlmostEqual(model.constant, 1.2, places=9)
        by_freq = {c.frequency: c for c in model.constituents}
        c1, c2 = by_freq[W1], by_freq[W2]
        self.assertAlmostEqual(c1.amplitude, 0.8, places=9)
        self.assertAlmostEqual(c2.amplitude, 0.4, places=9)
        self.assertAlmostEqual(c1.phase, 0.3, places=9)
        self.assertAlmostEqual(c2.phase, 2.0, places=9)
        self.assertTrue(0.0 <= c1.phase < 2.0 * math.pi)

    def test_zero_amplitude_phase_is_none(self):
        times = [0.0, 1.3, 2.7, 5.1, 8.4, 12.9]
        values = [2.0 + 0.5 * math.cos(W1 * t) for t in times]
        model = fit_model(EPOCH, times, values, [W1, W2])
        by_freq = {c.frequency: c for c in model.constituents}
        self.assertAlmostEqual(by_freq[W2].amplitude, 0.0, places=12)
        self.assertIsNone(by_freq[W2].phase)

    def test_rank_deficiency_rejected(self):
        # 4 observations < 5 unknowns (constant + 2*2).
        times = [0.0, 1.0, 2.0, 3.0]
        values = [1.0, 1.1, 0.9, 1.0]
        with self.assertRaises(ValueError):
            fit_model(EPOCH, times, values, [W1, W2])

    def test_duplicate_times_rejected(self):
        times = [0.0, 1.0, 1.0, 2.0, 3.0, 4.0]
        values = [1.0] * 6
        with self.assertRaises(ValueError):
            fit_model(EPOCH, times, values, [W1])

    def test_duplicate_or_bad_frequencies_rejected(self):
        times = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        values = [1.0] * 6
        with self.assertRaises(ValueError):
            fit_model(EPOCH, times, values, [W1, W1])
        with self.assertRaises(ValueError):
            fit_model(EPOCH, times, values, [W1, -0.2])


class ReloadTests(unittest.TestCase):
    def test_model_roundtrip_predictions_match(self):
        times = [0.0, 0.9, 2.3, 3.8, 6.6, 9.4, 12.2, 15.5, 19.9, 24.0]
        values = [true_level(t) for t in times]
        model = fit_model(EPOCH, times, values, [W1, W2])
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.json")
            model.save(path)
            reloaded = load_model(path)
            self.assertEqual(reloaded.epoch, model.epoch)
            for t in (0.0, 3.3, 50.0, 123.4):
                self.assertAlmostEqual(reloaded.predict_hours(t),
                                       model.predict_hours(t), places=12)
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            self.assertIn("epoch_utc", data)
            self.assertEqual(len(data["constituents"]), 2)


class IntervalTests(unittest.TestCase):
    def test_linear_crossing_interpolation(self):
        # Levels 0, 2, 0 at t=0,1,2 with threshold 1.0:
        # crossings at t=0.5 and t=1.5.
        samples = [(0.0, 0.0), (1.0, 2.0), (2.0, 0.0)]
        intervals = above_threshold_intervals(samples, 1.0)
        self.assertEqual(len(intervals), 1)
        self.assertAlmostEqual(intervals[0][0], 0.5)
        self.assertAlmostEqual(intervals[0][1], 1.5)

    def test_tangent_touch_not_an_interval(self):
        # Curve touches threshold at a single sample but never exceeds it.
        samples = [(0.0, 0.0), (1.0, 1.0), (2.0, 0.0)]
        self.assertEqual(above_threshold_intervals(samples, 1.0), [])

    def test_open_interval_at_window_edges(self):
        samples = [(0.0, 2.0), (1.0, 2.0), (2.0, 0.0)]
        intervals = above_threshold_intervals(samples, 1.0)
        self.assertEqual(len(intervals), 1)
        self.assertAlmostEqual(intervals[0][0], 0.0)
        self.assertAlmostEqual(intervals[0][1], 1.5)

    def test_sampling_includes_short_final_segment(self):
        times = [0.0, 0.9, 2.3, 3.8, 6.6, 9.4, 12.2, 15.5, 19.9, 24.0]
        values = [true_level(t) for t in times]
        model = fit_model(EPOCH, times, values, [W1, W2])
        samples = sample_window(model, "2024-01-02T00:00:00Z",
                                "2024-01-02T10:00:00Z", 3.0)
        ts = [t for t, _ in samples]
        self.assertEqual(ts, [24.0, 27.0, 30.0, 33.0, 34.0])

    def test_sampling_rejects_bad_step(self):
        times = [0.0, 0.9, 2.3, 3.8, 6.6, 9.4, 12.2, 15.5, 19.9, 24.0]
        values = [true_level(t) for t in times]
        model = fit_model(EPOCH, times, values, [W1, W2])
        with self.assertRaises(ValueError):
            sample_window(model, "2024-01-02T00:00:00Z",
                          "2024-01-02T10:00:00Z", 0.0)


if __name__ == "__main__":
    unittest.main()
