from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


EXPECTED_DIPS = {
    "Ag/MOF": 573.0,
    "Au/MOF": 611.0,
    "MOF": 757.0,
}

SENSOR_STRENGTH = {
    "Ag/MOF": 1.00,
    "Au/MOF": 0.72,
    "MOF": 0.45,
}

DEFAULT_SENSOR_SEQUENCE = ("Ag/MOF", "Au/MOF", "MOF")
DEFAULT_HUMIDITY_LEVELS = (20.0, 50.0, 80.0)
DEFAULT_TARGET_CLIP = (0.0, 80.0)


@dataclass
class DatasetBundle:
    dataframe: pd.DataFrame
    metadata_used: bool
    inferred_fields: list[str]


class SpectralRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Iterable[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train physics-aware spectra models.")
    parser.add_argument(
        "--data",
        default="data/spectra-data.csv",
        help="Path to the wide-form spectra CSV.",
    )
    parser.add_argument(
        "--metadata",
        default="data/series_metadata.csv",
        help="Optional path to metadata with series_id, sensor_type, humidity.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory for metrics, plots, and generated spectra.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=140,
        help="Training epochs for the neural model.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Mini-batch size for the neural model.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate for Adam.",
    )
    return parser.parse_args()


def reshape_wide_spectra(csv_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(csv_path, low_memory=False)
    raw.columns = [str(column).strip() for column in raw.columns]
    raw = raw.rename(columns={raw.columns[0]: "wavelength"})

    melted_frames: list[pd.DataFrame] = []
    header_pattern = re.compile(r"^(T\d+)\s+(\d+)%$")

    for column_index, column in enumerate(raw.columns[1:], start=1):
        if not column or column.startswith("Unnamed"):
            continue

        match = header_pattern.match(column)
        if match is None:
            continue

        series_id, co2_percent = match.groups()
        column_frame = pd.DataFrame(
            {
                "wavelength": raw.iloc[:, 0],
                "transmittance": raw.iloc[:, column_index],
            }
        )
        column_frame["series_id"] = series_id
        column_frame["co2"] = float(co2_percent)
        melted_frames.append(column_frame)

    if not melted_frames:
        raise ValueError("No spectral columns matching 'T<number> <co2>%' were found.")

    spectra = pd.concat(melted_frames, ignore_index=True)
    spectra["wavelength"] = pd.to_numeric(spectra["wavelength"], errors="coerce")
    spectra["transmittance"] = pd.to_numeric(spectra["transmittance"], errors="coerce")
    spectra = spectra.dropna(subset=["wavelength", "transmittance", "co2"]) 
    return spectra


def load_metadata(metadata_path: Path) -> pd.DataFrame | None:
    if not metadata_path.exists():
        return None

    metadata = pd.read_csv(metadata_path)
    metadata.columns = [str(column).strip() for column in metadata.columns]
    required_columns = {"series_id", "sensor_type", "humidity"}
    missing_columns = required_columns.difference(metadata.columns)
    if missing_columns:
        raise ValueError(
            f"Metadata file is missing required columns: {sorted(missing_columns)}"
        )
    metadata = metadata[["series_id", "sensor_type", "humidity"]].copy()
    metadata["series_id"] = metadata["series_id"].astype(str).str.strip()
    metadata["sensor_type"] = metadata["sensor_type"].astype(str).str.strip()
    metadata["humidity"] = pd.to_numeric(metadata["humidity"], errors="coerce")
    metadata = metadata.dropna(subset=["series_id", "sensor_type", "humidity"])
    return metadata


def infer_metadata(series_ids: Iterable[str]) -> pd.DataFrame:
    rows = []
    sorted_ids = sorted(series_ids, key=lambda value: int(re.search(r"\d+", value).group(0)))
    for index, series_id in enumerate(sorted_ids):
        rows.append(
            {
                "series_id": series_id,
                "sensor_type": DEFAULT_SENSOR_SEQUENCE[index % len(DEFAULT_SENSOR_SEQUENCE)],
                "humidity": DEFAULT_HUMIDITY_LEVELS[(index // len(DEFAULT_SENSOR_SEQUENCE)) % len(DEFAULT_HUMIDITY_LEVELS)],
            }
        )
    return pd.DataFrame(rows)


def load_dataset(data_path: Path, metadata_path: Path) -> DatasetBundle:
    spectra = reshape_wide_spectra(data_path)
    metadata = load_metadata(metadata_path)
    inferred_fields: list[str] = []

    if metadata is None or metadata.empty:
        metadata = infer_metadata(spectra["series_id"].unique())
        inferred_fields.extend(["sensor_type", "humidity"])

    dataset = spectra.merge(metadata, on="series_id", how="left")
    if dataset["sensor_type"].isna().any() or dataset["humidity"].isna().any():
        fallback = infer_metadata(dataset.loc[dataset["sensor_type"].isna() | dataset["humidity"].isna(), "series_id"].unique())
        dataset = dataset.drop(columns=["sensor_type", "humidity"]).merge(
            pd.concat([metadata, fallback], ignore_index=True).drop_duplicates("series_id", keep="first"),
            on="series_id",
            how="left",
        )
        if "sensor_type" not in inferred_fields:
            inferred_fields.extend(["sensor_type", "humidity"])

    dataset["humidity"] = pd.to_numeric(dataset["humidity"], errors="coerce")
    dataset = dataset.dropna(subset=["wavelength", "co2", "humidity", "sensor_type", "transmittance"]) 
    dataset = dataset.drop_duplicates(subset=["series_id", "wavelength", "co2"])
    dataset = dataset.loc[dataset["wavelength"].between(180.0, 900.0)]
    dataset = dataset.loc[dataset["transmittance"].between(-20.0, 120.0)]
    dataset["transmittance"] = dataset["transmittance"].clip(*DEFAULT_TARGET_CLIP)
    dataset["sensor_type"] = pd.Categorical(
        dataset["sensor_type"],
        categories=list(EXPECTED_DIPS.keys()),
    )
    dataset = dataset.dropna(subset=["sensor_type"]).reset_index(drop=True)

    return DatasetBundle(
        dataframe=dataset,
        metadata_used=metadata_path.exists(),
        inferred_fields=inferred_fields,
    )


def build_baseline_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                ["wavelength", "co2", "humidity"],
            ),
            (
                "sensor",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ["sensor_type"],
            ),
        ]
    )


def fit_baselines(train_frame: pd.DataFrame, test_frame: pd.DataFrame) -> tuple[dict[str, dict[str, float]], dict[str, np.ndarray], Pipeline]:
    features = ["wavelength", "co2", "humidity", "sensor_type"]
    baseline_preprocessor = build_baseline_preprocessor()
    baseline_models = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=140,
            max_depth=20,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingRegressor(random_state=42),
    }

    metrics: dict[str, dict[str, float]] = {}
    predictions: dict[str, np.ndarray] = {}
    fitted_gradient_pipeline: Pipeline | None = None

    for model_name, estimator in baseline_models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", baseline_preprocessor),
                ("model", estimator),
            ]
        )
        pipeline.fit(train_frame[features], train_frame["transmittance"])
        predicted = pipeline.predict(test_frame[features])
        predictions[model_name] = predicted
        metrics[model_name] = score_predictions(test_frame["transmittance"].to_numpy(), predicted)
        if model_name == "gradient_boosting":
            fitted_gradient_pipeline = pipeline

    if fitted_gradient_pipeline is None:
        raise RuntimeError("Gradient boosting pipeline was not fitted.")

    return metrics, predictions, fitted_gradient_pipeline


def score_predictions(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    rmse = float(math.sqrt(mean_squared_error(actual, predicted)))
    r2 = float(r2_score(actual, predicted))
    return {"rmse": rmse, "r2": r2}


def build_neural_features(frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    numeric = frame[["wavelength", "co2", "humidity"]].to_numpy(dtype=np.float32)
    sensor = pd.get_dummies(frame["sensor_type"], prefix="sensor")
    ordered_columns = [f"sensor_{name}" for name in EXPECTED_DIPS]
    sensor = sensor.reindex(columns=ordered_columns, fill_value=0)
    features = np.concatenate([numeric, sensor.to_numpy(dtype=np.float32)], axis=1)
    return features, ["wavelength", "co2", "humidity", *ordered_columns]


def scale_neural_inputs(train_features: np.ndarray, test_features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train_features[:, :3].mean(axis=0, keepdims=True)
    std = train_features[:, :3].std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    train_scaled = train_features.copy()
    test_scaled = test_features.copy()
    train_scaled[:, :3] = (train_scaled[:, :3] - mean) / std
    test_scaled[:, :3] = (test_scaled[:, :3] - mean) / std
    return train_scaled, test_scaled, mean.astype(np.float32), std.astype(np.float32)


def expected_dip_targets(frame: pd.DataFrame) -> np.ndarray:
    return frame["sensor_type"].map(EXPECTED_DIPS).to_numpy(dtype=np.float32)


def train_neural_model(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> tuple[SpectralRegressor, dict[str, float], np.ndarray, dict[str, np.ndarray]]:
    train_features_raw, feature_names = build_neural_features(train_frame)
    test_features_raw, _ = build_neural_features(test_frame)
    train_features, test_features, numeric_mean, numeric_std = scale_neural_inputs(train_features_raw, test_features_raw)

    train_targets = train_frame["transmittance"].to_numpy(dtype=np.float32).reshape(-1, 1)
    test_targets = test_frame["transmittance"].to_numpy(dtype=np.float32).reshape(-1, 1)

    train_dataset = TensorDataset(
        torch.tensor(train_features, dtype=torch.float32),
        torch.tensor(train_targets, dtype=torch.float32),
        torch.tensor(train_frame["co2"].to_numpy(dtype=np.float32).reshape(-1, 1)),
        torch.tensor(train_frame["humidity"].to_numpy(dtype=np.float32).reshape(-1, 1)),
        torch.tensor(train_frame["wavelength"].to_numpy(dtype=np.float32).reshape(-1, 1)),
        torch.tensor(expected_dip_targets(train_frame).reshape(-1, 1), dtype=torch.float32),
        torch.tensor(train_frame["sensor_type"].map(SENSOR_STRENGTH).to_numpy(dtype=np.float32).reshape(-1, 1)),
    )

    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    model = SpectralRegressor(input_dim=train_features.shape[1], hidden_dims=[128, 128, 64, 32])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    mse_loss = nn.MSELoss()

    numeric_mean_tensor = torch.tensor(numeric_mean, dtype=torch.float32)
    numeric_std_tensor = torch.tensor(numeric_std, dtype=torch.float32)

    for epoch in range(epochs):
        model.train()
        for batch_features, batch_targets, batch_co2, batch_humidity, batch_wavelength, batch_dip_target, batch_strength in loader:
            optimizer.zero_grad()
            predictions = model(batch_features)
            base_loss = mse_loss(predictions, batch_targets)

            co2_probe = batch_features.clone().detach()
            co2_probe.requires_grad_(True)
            co2_predictions = model(co2_probe)
            co2_grad = torch.autograd.grad(
                co2_predictions.sum(),
                co2_probe,
                create_graph=True,
            )[0][:, 1:2]
            monotonic_penalty = torch.relu(co2_grad).mean()

            wavelength_probe = batch_features.clone().detach()
            wavelength_probe.requires_grad_(True)
            smooth_predictions = model(wavelength_probe)
            wavelength_grad = torch.autograd.grad(
                smooth_predictions.sum(),
                wavelength_probe,
                create_graph=True,
            )[0][:, 0:1]
            smoothness_penalty = (wavelength_grad ** 2).mean()

            unscaled_wavelength = wavelength_probe[:, 0:1] * numeric_std_tensor[:, 0:1] + numeric_mean_tensor[:, 0:1]
            dip_distance = ((unscaled_wavelength - batch_dip_target) / 75.0) ** 2
            dip_floor = 60.0 - (12.0 * batch_strength) - (0.18 * batch_co2) - (0.08 * batch_humidity)
            dip_target = torch.clamp(dip_floor + (16.0 * dip_distance), min=0.0, max=80.0)
            resonance_penalty = ((predictions - dip_target) ** 2).mean()

            humidity_penalty = torch.relu(predictions - (82.0 - 0.16 * batch_humidity)).mean()

            loss = base_loss + 0.8 * monotonic_penalty + 0.02 * smoothness_penalty + 0.015 * resonance_penalty + 0.02 * humidity_penalty
            loss.backward()
            optimizer.step()

        if epoch % 50 == 0 or epoch == epochs - 1:
            print(f"epoch={epoch:03d} loss={loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        raw_predictions = model(torch.tensor(test_features, dtype=torch.float32)).numpy().reshape(-1)
        clipped_predictions = np.clip(raw_predictions, *DEFAULT_TARGET_CLIP)

    metrics = score_predictions(test_targets.reshape(-1), clipped_predictions)
    training_state = {
        "feature_names": np.array(feature_names, dtype=object),
        "numeric_mean": numeric_mean,
        "numeric_std": numeric_std,
    }
    return model, metrics, clipped_predictions, training_state


def predict_with_neural_bundle(model: SpectralRegressor, state: dict[str, np.ndarray], frame: pd.DataFrame) -> np.ndarray:
    features_raw, _ = build_neural_features(frame)
    features = features_raw.copy()
    features[:, :3] = (features[:, :3] - state["numeric_mean"]) / state["numeric_std"]
    model.eval()
    with torch.no_grad():
        predicted = model(torch.tensor(features, dtype=torch.float32)).numpy().reshape(-1)
    return np.clip(predicted, *DEFAULT_TARGET_CLIP)


def plot_predicted_vs_actual(frame: pd.DataFrame, predicted: np.ndarray, output_path: Path, title: str) -> None:
    sample = frame.copy()
    sample["predicted"] = predicted
    groups = sample.groupby(["sensor_type", "co2", "humidity"], observed=True)
    selected_key, selected_group = max(groups, key=lambda item: len(item[1]))

    ordered = selected_group.sort_values("wavelength")
    plt.figure(figsize=(10, 6))
    plt.plot(ordered["wavelength"], ordered["transmittance"], label="Actual", linewidth=2.0)
    plt.plot(ordered["wavelength"], ordered["predicted"], label="Predicted", linewidth=2.0)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Transmittance (%)")
    plt.title(
        f"{title}: {selected_key[0]} | CO2={selected_key[1]:.0f}% | RH={selected_key[2]:.0f}%"
    )
    plt.ylim(*DEFAULT_TARGET_CLIP)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def generate_full_spectra(
    model: SpectralRegressor,
    state: dict[str, np.ndarray],
    dataset: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    rows = []
    for sensor_type in EXPECTED_DIPS:
        for humidity in sorted(dataset["humidity"].unique())[:3]:
            for co2 in sorted(dataset["co2"].unique())[:6]:
                wavelengths = np.linspace(550.0, 800.0, 400)
                sweep = pd.DataFrame(
                    {
                        "wavelength": wavelengths,
                        "co2": float(co2),
                        "humidity": float(humidity),
                        "sensor_type": sensor_type,
                    }
                )
                sweep["predicted_transmittance"] = predict_with_neural_bundle(model, state, sweep)
                rows.append(sweep)
    spectra = pd.concat(rows, ignore_index=True)
    spectra.to_csv(output_path, index=False)
    return spectra


def summarize_constraints(frame: pd.DataFrame, predictions: np.ndarray) -> dict[str, float]:
    scored = frame.copy()
    scored["predicted"] = predictions

    monotonic_pairs = []
    for _, group in scored.groupby(["sensor_type", "humidity", "wavelength"], observed=True):
        ordered = group.sort_values("co2")
        monotonic_pairs.extend(np.diff(ordered["predicted"].to_numpy()) <= 1e-4)

    smoothness = []
    for _, group in scored.groupby(["sensor_type", "humidity", "co2"], observed=True):
        ordered = group.sort_values("wavelength")
        second_diff = np.diff(ordered["predicted"].to_numpy(), n=2)
        if len(second_diff) > 0:
            smoothness.append(float(np.mean(np.abs(second_diff))))

    dip_errors = []
    for sensor_type, expected in EXPECTED_DIPS.items():
        sensor_group = scored.loc[scored["sensor_type"] == sensor_type]
        if sensor_group.empty:
            continue
        averaged = sensor_group.groupby("wavelength", observed=True)["predicted"].mean().reset_index()
        dip_wave = float(averaged.loc[averaged["predicted"].idxmin(), "wavelength"])
        dip_errors.append(abs(dip_wave - expected))

    return {
        "monotonic_fraction": float(np.mean(monotonic_pairs)) if monotonic_pairs else float("nan"),
        "mean_abs_second_diff": float(np.mean(smoothness)) if smoothness else float("nan"),
        "mean_dip_error_nm": float(np.mean(dip_errors)) if dip_errors else float("nan"),
    }


def write_metrics(output_dir: Path, baseline_metrics: dict[str, dict[str, float]], neural_metrics: dict[str, float], constraint_metrics: dict[str, float], bundle: DatasetBundle, dataset: pd.DataFrame) -> None:
    payload = {
        "rows": int(len(dataset)),
        "series": int(dataset["series_id"].nunique()),
        "metadata_used": bundle.metadata_used,
        "inferred_fields": bundle.inferred_fields,
        "baseline": baseline_metrics,
        "neural": neural_metrics,
        "constraints": constraint_metrics,
    }
    (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2))


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    metadata_path = Path(args.metadata)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_dataset(data_path, metadata_path)
    dataset = bundle.dataframe

    train_frame, test_frame = train_test_split(
        dataset,
        test_size=0.2,
        random_state=42,
        stratify=dataset[["sensor_type", "co2"]],
    )

    baseline_metrics, _, gradient_pipeline = fit_baselines(train_frame, test_frame)
    neural_model, neural_metrics, neural_predictions, neural_state = train_neural_model(
        train_frame,
        test_frame,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )

    baseline_predictions = gradient_pipeline.predict(test_frame[["wavelength", "co2", "humidity", "sensor_type"]])
    baseline_predictions = np.clip(baseline_predictions, *DEFAULT_TARGET_CLIP)
    plot_predicted_vs_actual(
        test_frame,
        baseline_predictions,
        output_dir / "baseline_predicted_vs_actual.png",
        "Gradient Boosting",
    )
    plot_predicted_vs_actual(
        test_frame,
        neural_predictions,
        output_dir / "neural_predicted_vs_actual.png",
        "Physics-Aware Neural Network",
    )
    generate_full_spectra(
        neural_model,
        neural_state,
        dataset,
        output_dir / "generated_spectra.csv",
    )

    constraint_metrics = summarize_constraints(test_frame, neural_predictions)
    write_metrics(output_dir, baseline_metrics, neural_metrics, constraint_metrics, bundle, dataset)

    print("Dataset summary")
    print(f"rows={len(dataset)} series={dataset['series_id'].nunique()} metadata_used={bundle.metadata_used}")
    if bundle.inferred_fields:
        print(f"inferred_fields={','.join(bundle.inferred_fields)}")
    print("Baseline metrics")
    for model_name, model_metrics in baseline_metrics.items():
        print(f"{model_name}: RMSE={model_metrics['rmse']:.4f} R2={model_metrics['r2']:.4f}")
    print("Neural metrics")
    print(f"physics_aware_nn: RMSE={neural_metrics['rmse']:.4f} R2={neural_metrics['r2']:.4f}")
    print("Constraint summary")
    for key, value in constraint_metrics.items():
        print(f"{key}={value:.4f}")


if __name__ == "__main__":
    main()