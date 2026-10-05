"""Harmonic model: fitting, persistence, evaluation, threshold intervals."""

import json
import math
from dataclasses import dataclass

from .errors import TidalError
from .io import format_utc, parse_utc
from .linalg import solve_least_squares

_PHASE_TOL = 1e-12


def hours_since(epoch, moment):
    return (moment - epoch).total_seconds() / 3600.0


@dataclass
class HarmonicModel:
    epoch: object
    frequencies: list
    mean: float
    cos_coeffs: list
    sin_coeffs: list

    def evaluate(self, moment):
        t = hours_since(self.epoch, moment)
        total = self.mean
        for omega, a_cos, a_sin in zip(self.frequencies, self.cos_coeffs, self.sin_coeffs):
            angle = omega * t
            total += a_cos * math.cos(angle) + a_sin * math.sin(angle)
        return total

    def constituents(self):
        """Amplitude and cosine-lag phase per constituent.

        Each term is A*cos(omega*t - phi) with phi in [0, 2*pi).
        A zero-amplitude term reports phase as None.
        """
        result = []
        for omega, a_cos, a_sin in zip(self.frequencies, self.cos_coeffs, self.sin_coeffs):
            amplitude = math.hypot(a_cos, a_sin)
            if amplitude <= _PHASE_TOL:
                phase = None
            else:
                phase = math.atan2(a_sin, a_cos) % (2.0 * math.pi)
            result.append({
                "frequency": omega,
                "amplitude": amplitude,
                "phase": phase,
            })
        return result

    def to_dict(self):
        return {
            "epoch": format_utc(self.epoch),
            "mean": self.mean,
            "constituents": [
                {
                    "frequency": omega,
                    "cos": a_cos,
                    "sin": a_sin,
                    "amplitude": entry["amplitude"],
                    "phase": entry["phase"],
                }
                for omega, a_cos, a_sin, entry in zip(
                    self.frequencies, self.cos_coeffs, self.sin_coeffs, self.constituents()
                )
            ],
        }

    @classmethod
    def from_dict(cls, data):
        try:
            epoch = parse_utc(str(data["epoch"]))
            mean = float(data["mean"])
            entries = data["constituents"]
        except (KeyError, TypeError, ValueError) as exc:
            raise TidalError(f"invalid model file: {exc}") from exc
        frequencies, cos_coeffs, sin_coeffs = [], [], []
        for entry in entries:
            frequencies.append(float(entry["frequency"]))
            cos_coeffs.append(float(entry["cos"]))
            sin_coeffs.append(float(entry["sin"]))
        return cls(epoch, frequencies, mean, cos_coeffs, sin_coeffs)


def fit_model(observations, spec):
    """Ordinary least squares fit of mean + cos/sin terms per frequency."""
    design = []
    target = []
    for obs in observations:
        t = hours_since(spec.epoch, obs.time)
        row = [1.0]
        for omega in spec.frequencies:
            row.append(math.cos(omega * t))
            row.append(math.sin(omega * t))
        design.append(row)
        target.append(obs.height)
    coeffs = solve_least_squares(design, target)
    return HarmonicModel(
        epoch=spec.epoch,
        frequencies=list(spec.frequencies),
        mean=coeffs[0],
        cos_coeffs=coeffs[1::2],
        sin_coeffs=coeffs[2::2],
    )


def residuals(model, observations):
    return [
        {"time": format_utc(obs.time), "observed": obs.height,
         "predicted": model.evaluate(obs.time),
         "residual": obs.height - model.evaluate(obs.time)}
        for obs in observations
    ]


def save_model(model, path):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(model.to_dict(), handle, indent=2)
        handle.write("\n")


def load_model(path):
    with open(path, encoding="utf-8") as handle:
        try:
            data = json.load(handle)
        except json.JSONDecodeError as exc:
            raise TidalError(f"invalid model JSON: {exc}") from exc
    return HarmonicModel.from_dict(data)
