"""Sampling of predictions and threshold-crossing analysis."""

from datetime import timedelta

from .errors import TidalError
from .io import format_utc


def sample_series(model, start, end, step_hours):
    """Sample the model from start to end inclusive.

    The final sample is always exactly at `end`, even if the last
    segment is shorter than one step.
    """
    if not step_hours > 0:
        raise TidalError("step must be a positive number of hours")
    if end < start:
        raise TidalError("prediction end must not be before start")
    step = timedelta(hours=step_hours)
    points = []
    moment = start
    while moment < end:
        points.append((moment, model.evaluate(moment)))
        moment += step
    points.append((end, model.evaluate(end)))
    return points


def above_threshold_intervals(points, threshold):
    """Continuous intervals where the piecewise-linear series is strictly
    above the threshold.

    Boundaries are linearly interpolated between samples. A tangential
    touch at a single sample produces no interval. These intervals are
    derived from the sampled polyline, not from exact root-finding on
    the continuous harmonic curve.
    """
    intervals = []
    open_start = None
    for (t0, v0), (t1, v1) in zip(points, points[1:]):
        above0 = v0 > threshold
        above1 = v1 > threshold
        if above0 and open_start is None:
            open_start = t0
        if above0 != above1:
            fraction = (threshold - v0) / (v1 - v0)
            crossing = t0 + (t1 - t0) * fraction
            if above0:
                intervals.append((open_start, crossing))
                open_start = None
            else:
                open_start = crossing
    if open_start is not None:
        intervals.append((open_start, points[-1][0]))
    return [(start, end) for start, end in intervals if end > start]


def intervals_report(intervals):
    total_seconds = sum((end - start).total_seconds() for start, end in intervals)
    return {
        "note": (
            "Intervals are computed on the linearly interpolated prediction "
            "samples, not by exact root-finding of the continuous harmonic "
            "curve; refine the step for sharper boundaries."
        ),
        "total_hours_above": total_seconds / 3600.0,
        "intervals": [
            {
                "start": format_utc(start),
                "end": format_utc(end),
                "duration_hours": (end - start).total_seconds() / 3600.0,
            }
            for start, end in intervals
        ],
    }
