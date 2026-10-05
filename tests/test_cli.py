import csv
import json
import math
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

from tidal.cli import main

EPOCH = datetime(2024, 1, 1, tzinfo=timezone.utc)


def write_inputs(tmp, count=36):
    obs_path = os.path.join(tmp, "obs.csv")
    with open(obs_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time", "height"])
        t = 0.0
        for i in range(count):
            moment = EPOCH + timedelta(hours=t)
            height = 0.5 + 0.4 * math.cos(0.5 * t - 1.0)
            writer.writerow([moment.strftime("%Y-%m-%dT%H:%M:%SZ"),
                             f"{height:.12f}"])
            t += 0.6 + (i % 4) * 0.5
    con_path = os.path.join(tmp, "constituents.json")
    with open(con_path, "w", encoding="utf-8") as handle:
        json.dump({"epoch": "2024-01-01T00:00:00Z", "frequencies": [0.5]}, handle)
    return obs_path, con_path


class CliTests(unittest.TestCase):
    def test_fit_then_predict_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            obs_path, con_path = write_inputs(tmp)
            model_path = os.path.join(tmp, "model.json")
            res_path = os.path.join(tmp, "residuals.csv")
            rc = main(["fit", "--observations", obs_path,
                       "--constituents", con_path,
                       "--model-out", model_path,
                       "--residuals-out", res_path])
            self.assertEqual(rc, 0)
            with open(model_path, encoding="utf-8") as handle:
                model = json.load(handle)
            self.assertAlmostEqual(model["mean"], 0.5, places=6)
            entry = model["constituents"][0]
            self.assertAlmostEqual(entry["amplitude"], 0.4, places=6)
            self.assertAlmostEqual(entry["phase"], 1.0, places=6)
            with open(res_path, encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 36)
            for row in rows:
                self.assertAlmostEqual(float(row["residual"]), 0.0, places=6)

            csv_path = os.path.join(tmp, "pred.csv")
            int_path = os.path.join(tmp, "intervals.json")
            rc = main(["predict", "--model", model_path,
                       "--start", "2024-01-01T00:00:00Z",
                       "--end", "2024-01-02T01:30:00Z",
                       "--step", "2", "--threshold", "0.7",
                       "--csv-out", csv_path,
                       "--intervals-out", int_path])
            self.assertEqual(rc, 0)
            with open(csv_path, encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            # 25.5 h window with 2 h step: 12 full steps + endpoint.
            self.assertEqual(rows[-1]["time"], "2024-01-02T01:30:00Z")
            self.assertEqual(len(rows), 14)
            with open(int_path, encoding="utf-8") as handle:
                report = json.load(handle)
            self.assertIn("note", report)
            self.assertGreater(report["total_hours_above"], 0.0)
            for interval in report["intervals"]:
                self.assertGreater(interval["duration_hours"], 0.0)

    def test_fit_failure_returns_error_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            obs_path = os.path.join(tmp, "obs.csv")
            with open(obs_path, "w", encoding="utf-8") as handle:
                handle.write("time,height\n2024-01-01T00:00:00Z,1.0\n")
            con_path = os.path.join(tmp, "c.json")
            with open(con_path, "w", encoding="utf-8") as handle:
                json.dump({"epoch": "2024-01-01T00:00:00Z",
                           "frequencies": [0.5, 1.0]}, handle)
            rc = main(["fit", "--observations", obs_path,
                       "--constituents", con_path,
                       "--model-out", os.path.join(tmp, "m.json"),
                       "--residuals-out", os.path.join(tmp, "r.csv")])
            self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
