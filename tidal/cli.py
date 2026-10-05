"""Command line interface: fit and predict subcommands."""

import argparse
import csv
import json
import sys

from .errors import TidalError
from .io import format_utc, load_constituents, load_observations, parse_utc
from .model import fit_model, load_model, residuals, save_model
from .predict import above_threshold_intervals, intervals_report, sample_series


def _cmd_fit(args):
    observations = load_observations(args.observations)
    spec = load_constituents(args.constituents)
    model = fit_model(observations, spec)
    save_model(model, args.model_out)
    with open(args.residuals_out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["time", "observed", "predicted", "residual"]
        )
        writer.writeheader()
        for row in residuals(model, observations):
            writer.writerow(row)
    print(f"fitted {len(spec.frequencies)} constituents from "
          f"{len(observations)} observations")
    print(f"model written to {args.model_out}")
    print(f"residuals written to {args.residuals_out}")
    return 0


def _cmd_predict(args):
    model = load_model(args.model)
    start = parse_utc(args.start)
    end = parse_utc(args.end)
    points = sample_series(model, start, end, args.step)
    with open(args.csv_out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time", "height"])
        for moment, value in points:
            writer.writerow([format_utc(moment), f"{value:.10f}"])
    report = None
    if args.threshold is not None:
        intervals = above_threshold_intervals(points, args.threshold)
        report = intervals_report(intervals)
        report["threshold"] = args.threshold
        with open(args.intervals_out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
    print(f"wrote {len(points)} samples to {args.csv_out}")
    if report is not None:
        print(f"{len(report['intervals'])} above-threshold intervals, "
              f"total {report['total_hours_above']:.4f} h -> {args.intervals_out}")
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        prog="tidal",
        description="Simplified harmonic tidal analysis (standard library only).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    fit = sub.add_parser("fit", help="fit constituents from observations")
    fit.add_argument("--observations", required=True, help="CSV with time,height")
    fit.add_argument("--constituents", required=True,
                     help="JSON with epoch and frequencies (rad/hour)")
    fit.add_argument("--model-out", required=True, help="output model JSON")
    fit.add_argument("--residuals-out", required=True, help="output residuals CSV")
    fit.set_defaults(func=_cmd_fit)

    predict = sub.add_parser("predict", help="predict water levels from a model")
    predict.add_argument("--model", required=True, help="model JSON from fit")
    predict.add_argument("--start", required=True, help="UTC start time")
    predict.add_argument("--end", required=True, help="UTC end time (inclusive)")
    predict.add_argument("--step", required=True, type=float,
                         help="sampling step in hours (positive)")
    predict.add_argument("--csv-out", required=True, help="output prediction CSV")
    predict.add_argument("--threshold", type=float, default=None,
                         help="water level threshold in metres")
    predict.add_argument("--intervals-out", default="intervals.json",
                         help="output JSON for above-threshold intervals")
    predict.set_defaults(func=_cmd_predict)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except TidalError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
