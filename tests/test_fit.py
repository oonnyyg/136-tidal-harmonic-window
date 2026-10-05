import json
import math
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

from tidal.errors import TidalError
from tidal.io import ConstituentSpec, load_constituents, load_observations
from tidal.model import fit_model, load_model, save_model

EPOCH = datetime(2024, 1, 1, tzinfo=timezone.utc)
OMEGA1 = 0.5
OMEGA2 = 1.3


def true_height(t_hours):
    return 1.2 + 0.8 * math.cos(OMEGA1 * t_hours - 0.7) \
        + 0.3 * math.cos(OMEGA2 * t_hours - 2.0)


def irregular_times(count=40):
    """Deterministic irregular (non-uniform) sample times in hours."""
    times = []
    t = 0.0
    for i in range(count):
        times.append(t)
        t += 0.4 + (i % 5) * 0.37
    return times


class FitTests(unittest.TestCase):
    def setUp(self):
        self.spec = ConstituentSpec(epoch=EPOCH, frequencies=[OMEGA1, OMEGA2])
        self.observations = []
        for t in irregular_times():
            moment = EPOCH + timedelta(hours=t)
            self.observations.append(type("Obs", (), {"time": moment,
                                                      "height": true_height(t)}))

    def test_recovers_coefficients_on_irregular_samples(self):
        model = fit_model(self.observations, self.spec)
        self.assertAlmostEqual(model.mean, 1.2, places=9)
        entries = model.constituents()
        self.assertAlmostEqual(entries[0]["amplitude"], 0.8, places=9)
        self.assertAlmostEqual(entries[0]["phase"], 0.7, places=9)
        self.assertAlmostEqual(entries[1]["amplitude"], 0.3, places=9)
        self.assertAlmostEqual(entries[1]["phase"], 2.0, places=9)

    def test_phase_range_and_zero_amplitude(self):
        from tidal.model import HarmonicModel
        model = HarmonicModel(EPOCH, [OMEGA1, 3.3], 1.0, [0.8, 0.0], [0.6, 0.0])
        entries = model.constituents()
        for entry in entries:
            if entry["phase"] is not None:
                self.assertTrue(0.0 <= entry["phase"] < 2.0 * math.pi)
        # Exactly zero coefficients -> zero amplitude -> phase is null.
        self.assertEqual(entries[1]["amplitude"], 0.0)
        self.assertIsNone(entries[1]["phase"])

    def test_rejects_too_few_observations(self):
        with self.assertRaises(TidalError):
            fit_model(self.observations[:4], self.spec)

    def test_rejects_rank_deficient_design(self):
        # All observations at the same phase of the only constituent.
        spec = ConstituentSpec(epoch=EPOCH, frequencies=[1.0])
        obs = []
        for k in range(5):
            moment = EPOCH + timedelta(hours=2.0 * math.pi * k)
            obs.append(type("Obs", (), {"time": moment, "height": 1.0 + k * 0.0}))
        with self.assertRaises(TidalError):
            fit_model(obs, spec)

    def test_model_roundtrip(self):
        model = fit_model(self.observations, self.spec)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.json")
            save_model(model, path)
            reloaded = load_model(path)
        self.assertEqual(reloaded.epoch, model.epoch)
        self.assertEqual(reloaded.frequencies, model.frequencies)
        for a, b in zip(reloaded.cos_coeffs, model.cos_coeffs):
            self.assertAlmostEqual(a, b, places=12)
        for a, b in zip(reloaded.sin_coeffs, model.sin_coeffs):
            self.assertAlmostEqual(a, b, places=12)
        # Reloaded model reproduces identical predictions (no refit).
        moment = EPOCH + timedelta(hours=123.4)
        self.assertAlmostEqual(reloaded.evaluate(moment), model.evaluate(moment),
                               places=12)


class InputValidationTests(unittest.TestCase):
    def test_duplicate_times_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "obs.csv")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("time,height\n"
                             "2024-01-01T00:00:00Z,1.0\n"
                             "2024-01-01T00:00:00Z,1.1\n")
            with self.assertRaises(TidalError):
                load_observations(path)

    def test_nonfinite_height_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "obs.csv")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("time,height\n2024-01-01T00:00:00Z,nan\n")
            with self.assertRaises(TidalError):
                load_observations(path)

    def test_constituent_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "c.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump({"epoch": "2024-01-01T00:00:00Z",
                           "frequencies": [0.5, 0.5]}, handle)
            with self.assertRaises(TidalError):
                load_constituents(path)
            with open(path, "w", encoding="utf-8") as handle:
                json.dump({"epoch": "2024-01-01T00:00:00Z",
                           "frequencies": [-0.5]}, handle)
            with self.assertRaises(TidalError):
                load_constituents(path)


if __name__ == "__main__":
    unittest.main()
