"""Command line interface: fit a harmonic model, then predict a window."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timedelta, timezone

from .intervals import above_threshold_intervals, sample_window
from .model import fit_model, hours_since, load_model, parse_utc

DISCLAIMER = (
    "Intervals are derived from a piecewise-linear interpolation between "
    "prediction samples, not from exact root-finding on the continuous "
    "harmonic curve."
)


def _iso(epoch: datetime, t_hours: float) -> str:
    dt = epoch + timedelta(hours=t_hours)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_observations(path: str) -> tuple[list[str], list[float]]:
    times, values = [], []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            times.append(row["time_utc"])
            values.append(float(row["level_m"]))
    return times, values


def cmd_fit(args: argparse.Namespace) -> int:
    with open(args.config, encoding="utf-8") as fh:
        config = json.load(fh)
    epoch = parse_utc(config["epoch_utc"])
    frequencies = [float(w) for w in config["frequencies_rad_per_hour"]]

    time_texts, values = _read_observations(args.observations)
    times_hours = [hours_since(epoch, parse_utc(t)) for t in time_texts]
    model = fit_model(epoch, times_hours, values, frequencies)
    model.save(args.model_out)

    with open(args.residuals_out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["time_utc", "observed_m", "predicted_m", "residual_m"])
        for text, t, obs in zip(time_texts, times_hours, values):
            pred = model.predict_hours(t)
            writer.writerow([text, f"{obs:.6f}", f"{pred:.6f}",
                             f"{obs - pred:.6f}"])

    print(f"model written to {args.model_out}")
    for c in model.constituents:
        phase = "null" if c.phase is None else f"{c.phase:.6f}"
        print(f"  w={c.frequency:.8f} rad/h  amplitude={c.amplitude:.6f} m  "
              f"phase={phase} rad")
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    model = load_model(args.model)
    samples = sample_window(model, args.start, args.end, args.step_hours)

    with open(args.csv_out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["time_utc", "level_m"])
        for t, level in samples:
            writer.writerow([_iso(model.epoch, t), f"{level:.6f}"])

    intervals = above_threshold_intervals(samples, args.threshold)
    report = {
        "threshold_m": args.threshold,
        "intervals": [
            {
                "start_utc": _iso(model.epoch, a),
                "end_utc": _iso(model.epoch, b),
                "duration_hours": b - a,
            }
            for a, b in intervals
        ],
        "total_hours_above": sum(b - a for a, b in intervals),
        "note": DISCLAIMER,
    }
    with open(args.intervals_out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
        fh.write("\n")

    print(f"prediction written to {args.csv_out}")
    print(f"intervals written to {args.intervals_out} "
          f"({len(intervals)} above-threshold interval(s))")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tidal",
        description="Simplified tidal harmonic analysis (standard library only).")
    sub = parser.add_subparsers(dest="command", required=True)

    fit = sub.add_parser("fit", help="fit a harmonic model from observations")
    fit.add_argument("--observations", required=True,
                     help="CSV with columns time_utc,level_m")
    fit.add_argument("--config", required=True,
                     help="JSON with epoch_utc and frequencies_rad_per_hour")
    fit.add_argument("--model-out", required=True, help="output model JSON")
    fit.add_argument("--residuals-out", required=True,
                     help="output residuals CSV")
    fit.set_defaults(func=cmd_fit)

    pred = sub.add_parser("predict", help="predict levels for a time window")
    pred.add_argument("--model", required=True, help="model JSON from fit")
    pred.add_argument("--start", required=True, help="window start, ISO UTC")
    pred.add_argument("--end", required=True, help="window end, ISO UTC")
    pred.add_argument("--step-hours", required=True, type=float,
                      help="positive sampling step in hours")
    pred.add_argument("--threshold", required=True, type=float,
                      help="threshold level in metres")
    pred.add_argument("--csv-out", required=True, help="output prediction CSV")
    pred.add_argument("--intervals-out", required=True,
                      help="output above-threshold intervals JSON")
    pred.set_defaults(func=cmd_predict)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, KeyError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
