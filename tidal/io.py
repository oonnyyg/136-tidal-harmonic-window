"""Loading and validation of observation CSV files and constituent JSON files."""

import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone

from .errors import TidalError

_TIME_FORMATS = ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S%z")


def parse_utc(value):
    """Parse an ISO-8601 UTC timestamp into an aware datetime."""
    text = value.strip()
    for fmt in _TIME_FORMATS:
        try:
            dt = datetime.strptime(text, fmt)
        except ValueError:
            continue
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    raise TidalError(f"invalid UTC timestamp: {value!r}")


def format_utc(dt):
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class Observation:
    time: datetime
    height: float


@dataclass
class ConstituentSpec:
    epoch: datetime
    frequencies: list  # angular frequencies in rad/hour, all distinct and > 0


def load_observations(path):
    """Read a CSV with columns time,height. Times must be unique, heights finite."""
    observations = []
    seen = set()
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {"time", "height"} <= set(reader.fieldnames):
            raise TidalError("observation CSV must have 'time' and 'height' columns")
        for lineno, row in enumerate(reader, start=2):
            try:
                moment = parse_utc(row["time"])
            except TidalError as exc:
                raise TidalError(f"line {lineno}: {exc}") from exc
            try:
                height = float(row["height"])
            except (TypeError, ValueError):
                raise TidalError(f"line {lineno}: invalid height {row['height']!r}")
            if not math.isfinite(height):
                raise TidalError(f"line {lineno}: height must be finite")
            if moment in seen:
                raise TidalError(f"line {lineno}: duplicate observation time {row['time']!r}")
            seen.add(moment)
            observations.append(Observation(moment, height))
    if not observations:
        raise TidalError("observation CSV contains no rows")
    observations.sort(key=lambda obs: obs.time)
    return observations


def load_constituents(path):
    """Read JSON with 'epoch' (UTC) and 'frequencies' (distinct positive rad/hour)."""
    with open(path, encoding="utf-8") as handle:
        try:
            data = json.load(handle)
        except json.JSONDecodeError as exc:
            raise TidalError(f"invalid constituent JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise TidalError("constituent JSON must be an object")
    if "epoch" not in data:
        raise TidalError("constituent JSON missing 'epoch'")
    epoch = parse_utc(str(data["epoch"]))
    freqs = data.get("frequencies")
    if not isinstance(freqs, list) or not freqs:
        raise TidalError("constituent JSON needs a non-empty 'frequencies' array")
    parsed = []
    for value in freqs:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TidalError(f"frequency must be a number, got {value!r}")
        value = float(value)
        if not math.isfinite(value) or value <= 0.0:
            raise TidalError(f"frequency must be positive and finite, got {value!r}")
        parsed.append(value)
    if len(set(parsed)) != len(parsed):
        raise TidalError("constituent frequencies must be distinct")
    return ConstituentSpec(epoch=epoch, frequencies=parsed)
