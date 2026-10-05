# tidal-harmonic-window

Simplified harmonic tidal analysis using only the Python standard library.
Fits tidal constituents from local observations (no online tide services)
and predicts water levels plus above-threshold intervals for a time window.

## Method

The model is a constant mean level plus one sine/cosine pair per
constituent angular frequency:

```
h(t) = mean + sum_k [ C_k cos(w_k t) + S_k sin(w_k t) ]
```

where `t` is hours since the reference epoch and `w_k` is in radians per
hour. Coefficients come from ordinary least squares on the actual
observation times, so irregularly spaced records are handled
naturally — no resampling, no FFT, no assumption of equal spacing.


Each constituent is reported as amplitude and cosine-lag phase,
`A cos(w t - phi)`, with `phi` in `[0, 2*pi)`. Exactly zero-amplitude
constituents report `phase: null`.

The fit is refused (non-zero exit) when the design matrix cannot uniquely
determine the coefficients: too few observations (`n < 2K + 1`) or a
rank-deficient design (e.g. all samples at the same tidal phase).

## Input formats

Observations CSV (`time,height`), UTC timestamps, unique times, finite
heights in metres:

```
time,height
2024-01-01T00:00:00Z,1.23
2024-01-01T01:17:00Z,1.41
```

Constituents JSON — frequencies are never guessed, they must be supplied:

```json
{"epoch": "2024-01-01T00:00:00Z", "frequencies": [0.505868, 0.523599]}
```

## Usage

```bash
python -m tidal fit --observations obs.csv --constituents constituents.json \
    --model-out model.json --residuals-out residuals.csv

python -m tidal predict --model model.json \
    --start 2024-02-01T00:00:00Z --end 2024-02-03T00:00:00Z --step 0.5 \
    --threshold 2.0 --csv-out prediction.csv --intervals-out intervals.json
```

`predict` reuses the stored epoch and coefficients — it never refits.
Sampling runs from start to end inclusive; the final sample is exactly at
the end time even when the last segment is shorter than one step.

## Outputs

- `model.json` — epoch, mean, per-constituent cos/sin coefficients plus
  derived amplitude/phase.
- `residuals.csv` — per-observation `time,observed,predicted,residual`.
- `prediction.csv` — `time,height` samples over the window.
- `intervals.json` — contiguous intervals strictly above the threshold,
  with per-interval and total durations in hours.

## Limitations

- Above-threshold intervals treat the prediction samples as piecewise
  linear and interpolate crossing points between samples. They are **not**
  exact roots of the continuous harmonic curve; use a smaller `--step`
  for sharper boundaries. A single-sample tangential touch of the
  threshold produces no interval.
- Least squares is solved via normal equations, which is adequate for a
  handful of constituents but less numerically robust than QR/SVD for
  nearly collinear frequencies or very long records.
- Closely spaced frequencies need a record longer than their beat period
  to be separable; otherwise the design matrix becomes rank deficient
  and the fit is rejected.
- No nodal corrections, no inference of constituent frequencies, no
  uncertainty estimates.

## Tests

```bash
python -m unittest discover -s tests
```
