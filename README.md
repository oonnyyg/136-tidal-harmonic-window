# tidal-harmonic-window

Simplified tidal harmonic analysis using only the Python standard library.
Observations may be irregularly spaced; no online tide service is used and
no spectral (FFT-style) transform is performed — the model is fitted by
ordinary least squares directly on the observation times.

## Model

```
h(t) = c0 + sum_k [ a_k cos(w_k t) + b_k sin(w_k t) ]
```

- `t` is hours since a reference epoch given in the config JSON.
- `w_k` are positive, mutually distinct angular frequencies in rad/hour,
  supplied by the user (constituent frequencies are never guessed).
- Each constituent is reported as amplitude `A_k = hypot(a_k, b_k)` and
  lag phase `phi_k` in the form `A_k cos(w_k t - phi_k)`, with
  `phi_k` in `[0, 2*pi)`. A zero-amplitude constituent has phase `null`.

## Input formats

Observations CSV (`time_utc,level_m`), UTC ISO-8601 times, metres, no
duplicate timestamps:

```
time_utc,level_m
2024-01-01T00:00:00Z,1.234
```

Fit config JSON:

```json
{
  "epoch_utc": "2024-01-01T00:00:00Z",
  "frequencies_rad_per_hour": [0.5, 1.1]
}
```

## Usage

```
python -m tidal.cli fit --observations obs.csv --config config.json \
    --model-out model.json --residuals-out residuals.csv

python -m tidal.cli predict --model model.json \
    --start 2024-01-04T00:00:00Z --end 2024-01-05T00:00:00Z \
    --step-hours 0.5 --threshold 1.8 \
    --csv-out pred.csv --intervals-out intervals.json
```

- `fit` writes the model JSON (epoch, constant, per-frequency cosine/sine
  coefficients, amplitude, phase) and a residuals CSV.
- `predict` reloads the saved model (no refitting), samples from start to
  end inclusive — the end instant is kept even if the last segment is
  shorter than one step — and writes the prediction CSV plus a JSON report
  of above-threshold intervals and total duration.

## Validation and failure modes

- Duplicate observation times, non-finite levels, non-positive or
  duplicate frequencies are rejected.
- If the design matrix is rank deficient (e.g. fewer observations than
  the `1 + 2K` unknowns, or degenerate time/frequency combinations), the
  fit is refused with an error instead of returning zero-padded
  coefficients.

## Limitations

- Least squares uses the normal equations with partial pivoting; it is
  adequate for small, well-separated frequency sets but less stable than
  QR/SVD for nearly collinear constituents or very long time spans.
- Above-threshold intervals treat the prediction samples as
  piecewise-linear and interpolate crossing times linearly. They are
  **not** exact roots of the continuous harmonic curve; a finer
  `--step-hours` reduces the interpolation error. A sample that merely
  touches the threshold (zero-duration tangent) does not count as an
  interval.
- The model is a constant plus the listed constituents only: no nodal
  corrections, no shallow-water overtones, no datum adjustments.

## Tests

```
python -m unittest discover -s tests
```

Covers irregular sampling recovery, rank-deficiency rejection, model
save/reload round-trip, threshold crossing interpolation, tangent
exclusion, and inclusive end sampling.
