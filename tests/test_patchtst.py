"""Tests for EXP-005 (src/transformer/*): shapes/reload, train-only scaler,
seeded determinism, no-future-leakage, and window/split integrity."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from src.classical.garch_mean_spec import EXPECTED_BOUNDARIES
from src.classical.sarima import load_returns
from src.data.splits import chronological_split
from src.transformer.dataset import (
    Scaler,
    make_loader,
    make_windows,
    validation_input_matrix,
)
from src.transformer.patchtst import PatchTST, build_model, count_parameters, default_config
from src.transformer.train import TRAIN_CONFIG, fit_model, seed_everything

ARCH = default_config()          # registered EXP-005 config
LOOKBACK = ARCH["lookback"]      # 96
TINY_ARCH = {                    # small model for speed-sensitive tests
    "lookback": 32, "patch_len": 8, "stride": 8,
    "d_model": 16, "n_heads": 2, "n_layers": 1, "ffn_dim": 32, "dropout": 0.1,
}


# --------------------------------------------------------------------------- #
# Model shapes / reload
# --------------------------------------------------------------------------- #
def test_registered_config_shape_and_fingerprint():
    model = build_model(ARCH)
    assert model.n_patches == (96 - 16) // 8 + 1 == 11
    x = torch.randn(5, 96)
    out = model(x)
    assert out.shape == (5, 1)
    # deterministic fingerprint of the config dict
    assert PatchTST.config_hash(ARCH) == PatchTST.config_hash(dict(ARCH))
    assert count_parameters(model) > 0


def test_forward_rejects_wrong_lookback():
    model = build_model(ARCH)
    with pytest.raises(ValueError, match="Expected"):
        model(torch.randn(2, 95))


def test_state_dict_reload_reproduces_predictions():
    seed_everything(7)
    m1 = build_model(TINY_ARCH)
    m2 = build_model(TINY_ARCH)
    m2.load_state_dict(m1.state_dict())
    x = torch.randn(4, TINY_ARCH["lookback"])
    assert torch.equal(m1.eval()(x), m2.eval()(x))


def test_invalid_patch_geometry_raises():
    with pytest.raises(ValueError, match="divisible"):
        PatchTST(lookback=96, patch_len=16, stride=7)


# --------------------------------------------------------------------------- #
# Scaler: TRAIN ONLY
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def splits() -> tuple[pd.Series, pd.Series, pd.Series]:
    return chronological_split(load_returns(), 0.70, 0.15)


def test_scaler_fitted_on_train_partition_only(splits):
    train, validation, _test = splits
    scaler = Scaler.fit(train.to_numpy())
    # frozen stats must equal the TRAIN statistics exactly, not any other window
    assert scaler.mean == pytest.approx(float(train.mean()), abs=0, rel=1e-12)
    assert scaler.std == pytest.approx(float(train.std(ddof=0)), abs=0, rel=1e-12)
    # sanity: train stats differ from full-sample stats (else the test is vacuous)
    full = pd.concat([train, validation])
    assert scaler.mean != pytest.approx(float(full.mean()), rel=1e-6)
    # round trip
    v = validation.to_numpy()
    np.testing.assert_allclose(scaler.inverse(scaler.transform(v)), v, atol=1e-12)
    assert scaler.to_dict()["fit_on"] == "train partition only"


def test_scaler_rejects_degenerate_input():
    with pytest.raises(ValueError, match="std"):
        Scaler.fit(np.zeros(10))
    with pytest.raises(ValueError, match="empty"):
        Scaler.fit(np.array([]))


# --------------------------------------------------------------------------- #
# Windows: no-future-leakage + split integrity
# --------------------------------------------------------------------------- #
def test_make_windows_inputs_end_at_t_minus_1():
    series = np.arange(200, dtype=float)
    X, y = make_windows(series, lookback=16, target_slice=slice(16, 100))
    # target_slice.indices -> (16, 100, 1); targets t = 16..99
    assert X.shape == (84, 16) and y.shape == (84,)
    for i, t in enumerate(range(16, 100)):
        np.testing.assert_array_equal(X[i], series[t - 16:t])   # ends at t-1
        assert y[i] == series[t]
        assert t - 1 not in range(t, t + 1) or X[i][-1] == series[t - 1]


def test_make_windows_never_sees_target_value_in_input():
    """Perturbing series at indices >= t must not change the window for t."""
    series = np.random.default_rng(0).normal(size=300)
    X, _ = make_windows(series, lookback=24, target_slice=slice(24, 150))
    perturbed = series.copy()
    perturbed[100:] += 1e6                    # targets t >= 100 and beyond
    X2, _ = make_windows(perturbed, lookback=24, target_slice=slice(24, 150))
    # windows whose targets are < 100 are untouched by construction...
    np.testing.assert_array_equal(X[: 100 - 24], X2[: 100 - 24])
    # ...and every window ends strictly before its own target index:
    for i, t in enumerate(range(24, 150)):
        last_used_index = t - 1
        assert last_used_index < t
        # the window content equals the original slice up to t-1
        np.testing.assert_allclose(X[i], series[t - 24:t], rtol=1e-6, atol=1e-7)


def test_validation_origins_stop_at_validation_end(splits):
    """The origin matrix must cover exactly the validation origins, never test."""
    train, validation, test = splits
    assert len(train) == EXPECTED_BOUNDARIES["train_n"]
    assert len(validation) == EXPECTED_BOUNDARIES["validation_n"]
    assert len(test) == EXPECTED_BOUNDARIES["test_n"]

    series = np.arange(len(train) + len(validation), dtype=float)
    X = validation_input_matrix(series, len(train), LOOKBACK, n_origins=len(validation))
    assert X.shape == (len(validation), LOOKBACK)
    for i, t in enumerate(range(len(train), len(train) + len(validation))):
        np.testing.assert_allclose(X[i], series[t - LOOKBACK:t], rtol=1e-6, atol=1e-7)
    # beyond validation end -> hard error (test origins are unreachable)
    with pytest.raises(ValueError, match="exceed"):
        validation_input_matrix(series, len(train), LOOKBACK,
                                n_origins=len(validation) + len(test))


def test_validation_inputs_do_not_use_their_own_target(splits):
    """Forecast at origin t must be invariant to series[t:] (forecast-before-reveal)."""
    train, validation, _test = splits
    n = len(train) + len(validation)
    series = np.random.default_rng(1).normal(size=n)
    X = validation_input_matrix(series, len(train), LOOKBACK, n_origins=len(validation))
    perturbed = series.copy()
    perturbed[len(train):] += 1e6            # every validation target
    X2 = validation_input_matrix(perturbed, len(train), LOOKBACK,
                                 n_origins=len(validation))
    # first origin's window lies entirely in train -> identical
    np.testing.assert_array_equal(X[0], X2[0])
    # all windows end before their own origin regardless:
    for i, t in enumerate(range(len(train), n)):
        np.testing.assert_allclose(X[i], series[t - LOOKBACK:t], rtol=1e-6, atol=1e-7)


def test_phase1_windows_stay_inside_train(splits):
    """Phase-1 train targets < internal split; internal-val targets within train."""
    train, _, _ = splits
    internal_end = int(len(train) * (1 - TRAIN_CONFIG["internal_val_fraction"]))
    train_scaled = np.arange(len(train), dtype=float)   # index-as-value sentinel
    X_tr, y_tr = make_windows(train_scaled, LOOKBACK, slice(LOOKBACK, internal_end))
    X_iv, y_iv = make_windows(train_scaled, LOOKBACK, slice(internal_end, len(train)))
    # sentinel value == index, so values reveal which indices were used
    assert y_tr.max() < internal_end
    assert y_iv.min() >= internal_end and y_iv.max() <= len(train) - 1
    assert X_tr[:, -1].max() < internal_end - 1          # inputs before their targets
    assert X_iv[:, -1].max() <= len(train) - 2


def test_make_loader_is_deterministic():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(64, 8)).astype(np.float32)
    y = rng.normal(size=64).astype(np.float32)
    batches1 = [(xb.clone(), yb.clone()) for xb, yb in make_loader(X, y, 16, True, seed=42)]
    batches2 = [(xb.clone(), yb.clone()) for xb, yb in make_loader(X, y, 16, True, seed=42)]
    for (a1, b1), (a2, b2) in zip(batches1, batches2):
        assert torch.equal(a1, a2) and torch.equal(b1, b2)


# --------------------------------------------------------------------------- #
# Training: seeded determinism (EXP-005 acceptance criterion)
# --------------------------------------------------------------------------- #
def _tiny_training_set(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    series = np.cumsum(rng.normal(size=400)) * 0.01
    series = ((series - series.mean()) / series.std()).astype(np.float32)
    return make_windows(series, TINY_ARCH["lookback"],
                        slice(TINY_ARCH["lookback"], 300))


def test_two_seeded_runs_produce_identical_losses():
    """EXP-005 acceptance: determinism — same seed -> byte-identical loss curves."""
    X, y = _tiny_training_set()
    cfg = {**TRAIN_CONFIG, "max_epochs": 4, "early_stopping_patience": 4}
    _, h1, e1 = fit_model(X, y, TINY_ARCH, cfg, seed=42)
    _, h2, e2 = fit_model(X, y, TINY_ARCH, cfg, seed=42)
    assert e1 == e2
    assert h1["train"] == h2["train"]           # exact float equality
    assert len(h1["train"]) == 4


def test_phase2_runs_exactly_best_epoch_epochs():
    X, y = _tiny_training_set()
    cfg = {**TRAIN_CONFIG, "max_epochs": 6, "early_stopping_patience": 6,
           "best_epoch": 3}
    _, history, epochs = fit_model(X, y, TINY_ARCH, cfg, seed=42)   # no X_val = phase 2
    assert epochs == 3
    assert len(history["train"]) == 3
    assert history["val"] == []


def test_early_stopping_restores_best_weights():
    X, y = _tiny_training_set()
    cfg = {**TRAIN_CONFIG, "max_epochs": 10, "early_stopping_patience": 10}
    model, history, best_epoch = fit_model(
        X, y, TINY_ARCH, cfg,
        X_val=X[:32], y_val=y[:32], seed=42,
    )
    assert 1 <= best_epoch <= 10
    assert len(history["val"]) >= 1
    # restored weights must reproduce the recorded best validation loss
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(X[:32])).squeeze(-1)
        loss = float(torch.nn.functional.mse_loss(pred, torch.from_numpy(y[:32])))
    assert loss == pytest.approx(min(history["val"]), abs=1e-6)
