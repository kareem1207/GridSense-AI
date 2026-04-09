"""Synthetic data generation for GridSense ML model training."""
from __future__ import annotations
import numpy as np


def generate_normal_readings(n_samples: int = 10_000, random_seed: int = 42) -> np.ndarray:
    """Generate synthetic normal transformer readings.

    Returns:
        Array of shape (n_samples, 11) with columns:
        [Va, Vb, Vc, Ia, Ib, Ic, oil_temp, power_factor, thd_pct, active_power_kw, reactive_power_kvar]
    """
    rng = np.random.default_rng(random_seed)
    n = n_samples
    Va = rng.normal(230.0, 3.0, n)
    Vb = rng.normal(230.0, 3.0, n)
    Vc = rng.normal(230.0, 3.0, n)
    Ia = rng.normal(50.0, 5.0, n)
    Ib = rng.normal(50.0, 5.0, n)
    Ic = rng.normal(50.0, 5.0, n)
    oil_temp = rng.normal(60.0, 2.0, n)
    power_factor = np.clip(rng.normal(0.92, 0.02, n), 0.0, 1.0)
    thd_pct = np.abs(rng.normal(2.0, 0.3, n))
    active_power_kw = rng.normal(35.0, 5.0, n)
    reactive_power_kvar = rng.normal(8.0, 2.0, n)
    return np.column_stack([Va, Vb, Vc, Ia, Ib, Ic, oil_temp, power_factor, thd_pct, active_power_kw, reactive_power_kvar])


def generate_anomalous_readings(n_samples: int = 500, random_seed: int = 99) -> np.ndarray:
    """Generate synthetic anomalous transformer readings with injected faults.

    Fault types injected: voltage sag, current imbalance, overheating, high THD.

    Returns:
        Array of shape (n_samples, 11)
    """
    rng = np.random.default_rng(random_seed)
    data = generate_normal_readings(n_samples=n_samples, random_seed=random_seed)
    fault_type = rng.integers(0, 4, size=n_samples)
    for i in range(n_samples):
        ftype = fault_type[i]
        if ftype == 0:  # voltage sag
            data[i, 0] -= rng.uniform(30, 60)
            data[i, 1] -= rng.uniform(30, 60)
        elif ftype == 1:  # current imbalance
            data[i, 3] += rng.uniform(30, 80)
        elif ftype == 2:  # overheating
            data[i, 6] = rng.uniform(85, 120)
        elif ftype == 3:  # high THD
            data[i, 8] = rng.uniform(8, 20)
    return data


def extract_lstm_features(data: np.ndarray) -> np.ndarray:
    """Extract 7 LSTM features from 11-column DTM array.

    Selected features: [Va, Ib, oil_temp, power_factor, thd_pct, active_power_kw, reactive_power_kvar]
    Column indices:     [0,  4,  6,        7,            8,        9,               10]

    Args:
        data: array of shape (n_samples, 11)
    Returns:
        array of shape (n_samples, 7)
    """
    return data[:, [0, 4, 6, 7, 8, 9, 10]]


def make_sequences(data: np.ndarray, timesteps: int = 48) -> np.ndarray:
    """Create sliding window sequences for LSTM input.

    Args:
        data: array of shape (n_samples, n_features)
        timesteps: sequence window length
    Returns:
        array of shape (n_sequences, timesteps, n_features)
    """
    n = len(data) - timesteps
    if n <= 0:
        return np.empty((0, timesteps, data.shape[1]))
    return np.stack([data[i: i + timesteps] for i in range(n)])


def generate_degrading_sequence(
    n_steps: int = 200,
    transformer_id: str = "T-047",
    random_seed: int = 77,
) -> np.ndarray:
    """Generate a time-series sequence for a degrading transformer.

    Args:
        n_steps: number of timesteps to simulate
        transformer_id: ID for logging purposes
        random_seed: reproducibility seed
    Returns:
        array of shape (n_steps, 11) with progressive degradation
    """
    rng = np.random.default_rng(random_seed)
    rows = []
    for step in range(n_steps):
        Va = rng.normal(230.0, 3.0) - 0.05 * step
        Vb = rng.normal(230.0, 3.0) - 0.05 * step
        Vc = rng.normal(230.0, 3.0) - 0.05 * step
        Ia = rng.normal(50.0, 5.0)
        Ib = rng.normal(50.0, 5.0)
        Ic = rng.normal(50.0, 5.0)
        oil_temp = rng.normal(60.0, 2.0) + 0.3 * step
        power_factor = float(np.clip(rng.normal(0.92, 0.02) - 0.001 * step, 0.5, 1.0))
        thd_pct = rng.normal(2.0, 0.3) + 0.05 * step
        active_power_kw = rng.normal(35.0, 5.0)
        reactive_power_kvar = rng.normal(8.0, 2.0)
        rows.append([Va, Vb, Vc, Ia, Ib, Ic, oil_temp, power_factor, thd_pct, active_power_kw, reactive_power_kvar])
    return np.array(rows)
