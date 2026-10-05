"""Prediction sampling and above-threshold interval detection.

Samples are treated as piecewise-linear between consecutive points.
Intervals where the level is strictly above the threshold are reported
with linearly interpolated boundary crossings. A sample that merely
touches the threshold (tangent, zero duration) does not count.
"""
from __future__ import annotations

from .model import HarmonicModel, hours_since, parse_utc


def sample_window(model: HarmonicModel, start: str, end: str,
                  step_hours: float) -> list[tuple[float, float]]:
    """Sample the model from start to end inclusive.

    The end instant is always included, even if the final segment is
    shorter than one step. Returns (hours_since_epoch, level) pairs.
    """
    if step_hours <= 0.0:
        raise ValueError("step must be positive (hours)")
    t0 = hours_since(model.epoch, parse_utc(start))
    t1 = hours_since(model.epoch, parse_utc(end))
    if t1 < t0:
        raise ValueError("end must not be before start")
    samples = []
    t = t0
    while t < t1:
        samples.append((t, model.predict_hours(t)))
        t += step_hours
    samples.append((t1, model.predict_hours(t1)))
    return samples


def above_threshold_intervals(
        samples: list[tuple[float, float]],
        threshold: float) -> list[tuple[float, float]]:
    """Continuous intervals (in hours) where the piecewise-linear curve
    is strictly above the threshold, boundaries linearly interpolated."""
    intervals: list[tuple[float, float]] = []
    open_start: float | None = None

    for (t0, v0), (t1, v1) in zip(samples, samples[1:]):
        above0 = v0 > threshold
        above1 = v1 > threshold
        if above0 and open_start is None:
            open_start = t0
        if above0 != above1:
            frac = (threshold - v0) / (v1 - v0)
            t_cross = t0 + frac * (t1 - t0)
            if above0:
                intervals.append((open_start, t_cross))
                open_start = None
            else:
                open_start = t_cross
    if open_start is not None:
        t_end, v_end = samples[-1]
        if v_end > threshold:
            intervals.append((open_start, t_end))
    return intervals
