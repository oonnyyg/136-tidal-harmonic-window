import math
import unittest
from datetime import datetime, timedelta, timezone

from tidal.errors import TidalError
from tidal.model import HarmonicModel
from tidal.predict import (above_threshold_intervals, intervals_report,
                           sample_series)

EPOCH = datetime(2024, 1, 1, tzinfo=timezone.utc)


def simple_model():
    # height = cos(2*pi*t/12): period 12 h, amplitude 1.
    return HarmonicModel(EPOCH, [2.0 * math.pi / 12.0], 0.0, [1.0], [0.0])


class SamplingTests(unittest.TestCase):
    def test_inclusive_endpoint_with_short_tail(self):
        points = sample_series(simple_model(), EPOCH,
                               EPOCH + timedelta(hours=25), 10.0)
        times = [p[0] for p in points]
        self.assertEqual(times[-1], EPOCH + timedelta(hours=25))
        self.assertEqual(len(times), 4)  # 0, 10, 20, 25

    def test_rejects_nonpositive_step(self):
        with self.assertRaises(TidalError):
            sample_series(simple_model(), EPOCH, EPOCH + timedelta(hours=1), 0.0)

    def test_rejects_reversed_window(self):
        with self.assertRaises(TidalError):
            sample_series(simple_model(), EPOCH + timedelta(hours=2), EPOCH, 1.0)


class ThresholdTests(unittest.TestCase):
    def test_crossing_interpolation(self):
        # Linear series 0 -> 2 over one hour, threshold 1.0.
        t0 = EPOCH
        t1 = EPOCH + timedelta(hours=1)
        points = [(t0, 0.0), (t1, 2.0)]
        intervals = above_threshold_intervals(points, 1.0)
        self.assertEqual(len(intervals), 1)
        start, end = intervals[0]
        self.assertEqual(start, t0 + timedelta(minutes=30))
        self.assertEqual(end, t1)

    def test_tangent_touch_is_not_an_interval(self):
        # Series touches the threshold at one sample but never exceeds it.
        t0 = EPOCH
        points = [(t0, 0.0), (t0 + timedelta(hours=1), 1.0),
                  (t0 + timedelta(hours=2), 0.0)]
        self.assertEqual(above_threshold_intervals(points, 1.0), [])

    def test_full_cycle_intervals(self):
        model = simple_model()
        points = sample_series(model, EPOCH, EPOCH + timedelta(hours=24), 0.25)
        intervals = above_threshold_intervals(points, 0.5)
        # Window starts and ends mid-crest: two partial + one full interval.
        self.assertEqual(len(intervals), 3)
        report = intervals_report(intervals)
        # cos > 0.5 for ~4 h per 12 h cycle -> ~8 h total over 24 h.
        self.assertAlmostEqual(report["total_hours_above"], 8.0, delta=0.3)
        for entry in report["intervals"]:
            self.assertGreater(entry["duration_hours"], 0.0)


if __name__ == "__main__":
    unittest.main()
