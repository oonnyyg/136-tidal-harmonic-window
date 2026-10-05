"""Harmonic model: fit, persist, reload and predict.

Model: h(t) = c0 + sum_k [ a_k cos(w_k t) + b_k sin(w_k t) ]
where t is hours since the reference epoch and w_k is an angular
frequency in radians per hour. Amplitude/phase use the lag form
A_k cos(w_k t - phi_k), with phi in [0, 2*pi).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .linalg import solve_least_squares

TWO_PI = 2.0 * math.pi


def parse_utc(text: str) -> datetime:
    """Parse an ISO-8601 UTC timestamp (accepts trailing 'Z')."""
    dt = datetime.fromisoformat(text.strip().replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def hours_since(epoch: datetime, moment: datetime) -> float:
    return (moment - epoch).total_seconds() / 3600.0


@dataclass
class Constituent:
    frequency: float          # rad/hour
    cosine: float             # a_k
    sine: float               # b_k
    amplitude: float          # hypot(a_k, b_k)
    phase: float | None       # lag phase in [0, 2*pi); None if amplitude == 0


@dataclass
class HarmonicModel:
    epoch: datetime
    constant: float
    constituents: list[Constituent] = field(default_factory=list)

    def predict_hours(self, t_hours: float) -> float:
        value = self.constant
        for c in self.constituents:
            angle = c.frequency * t_hours
            value += c.cosine * math.cos(angle) + c.sine * math.sin(angle)
        return value

    def predict(self, moment: datetime) -> float:
        return self.predict_hours(hours_since(self.epoch, moment))

    def to_dict(self) -> dict:
        return {
            "epoch_utc": self.epoch.isoformat().replace("+00:00", "Z"),
            "constant": self.constant,
            "constituents": [
                {
                    "frequency_rad_per_hour": c.frequency,
                    "cosine": c.cosine,
                    "sine": c.sine,
                    "amplitude": c.amplitude,
                    "phase_rad": c.phase,
                }
                for c in self.constituents
            ],
        }

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
            fh.write("\n")

    @classmethod
    def from_dict(cls, data: dict) -> "HarmonicModel":
        epoch = parse_utc(data["epoch_utc"])
        constituents = [
            Constituent(
                frequency=float(c["frequency_rad_per_hour"]),
                cosine=float(c["cosine"]),
                sine=float(c["sine"]),
                amplitude=float(c["amplitude"]),
                phase=None if c["phase_rad"] is None else float(c["phase_rad"]),
            )
            for c in data["constituents"]
        ]
        return cls(epoch=epoch, constant=float(data["constant"]),
                   constituents=constituents)


def load_model(path: str) -> HarmonicModel:
    with open(path, "r", encoding="utf-8") as fh:
        return HarmonicModel.from_dict(json.load(fh))


def fit_model(epoch: datetime, times_hours: list[float],
              values: list[float], frequencies: list[float]) -> HarmonicModel:
    """Ordinary least squares fit of constant + cos/sin per frequency."""
    if len(times_hours) != len(values):
        raise ValueError("times and values length mismatch")
    if not frequencies:
        raise ValueError("at least one frequency is required")
    if any(w <= 0.0 for w in frequencies):
        raise ValueError("frequencies must be positive (rad/hour)")
    if len(set(frequencies)) != len(frequencies):
        raise ValueError("frequencies must be distinct")
    if len(set(times_hours)) != len(times_hours):
        raise ValueError("observation times must be distinct")
    if any(not math.isfinite(v) for v in values):
        raise ValueError("water levels must be finite")

    design = []
    for t in times_hours:
        row = [1.0]
        for w in frequencies:
            row.append(math.cos(w * t))
            row.append(math.sin(w * t))
        design.append(row)

    coeffs = solve_least_squares(design, values)
    constant = coeffs[0]
    constituents = []
    for k, w in enumerate(frequencies):
        a = coeffs[1 + 2 * k]
        b = coeffs[2 + 2 * k]
        amplitude = math.hypot(a, b)
        if amplitude <= 1e-12:
            a = b = amplitude = 0.0
            phase = None
        else:
            phase = math.atan2(b, a) % TWO_PI
        constituents.append(Constituent(w, a, b, amplitude, phase))
    return HarmonicModel(epoch=epoch, constant=constant,
                         constituents=constituents)
