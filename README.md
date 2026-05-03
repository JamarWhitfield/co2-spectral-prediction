# Spectral Prediction Model

This workspace implements the model described in `instructions.md` with a single training entrypoint:

```bash
python spectral_model.py
```

## What the script does

- reshapes the wide-form spectra CSV into long-form training data
- validates and cleans numeric types
- uses `data/series_metadata.csv` when available for `sensor_type` and `humidity`
- falls back to deterministic inferred metadata when the CSV is incomplete
- trains three baseline regressors
- trains a PyTorch neural network with monotonic, smoothness, and resonance penalties
- writes metrics, plots, and generated spectra to `outputs/`

## Metadata note

The supplied raw dataset does not contain explicit `sensor_type` or `humidity` columns. Update `data/series_metadata.csv` with the true mapping for each `series_id` to improve physical validity.# co2-spectral-prediction
