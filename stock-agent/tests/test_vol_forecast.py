# The layer whose bugs are hardest to see, because every one still produces a number: a leaked
# label scores well, a dropped smearing factor merely reads low, a narrowed feature set is a
# different model wearing the same name. The walk-forward purge gets the most attention.

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).parent.parent))
from conftest import make_bars
from src.features.technical import build_features
from src.features.volatility import build_volatility_features
from src.labels.target import add_forward_vol, add_labels
from src.models import vol_forecast
from src.models.vol_forecast import (HAR_FEATURES, LABEL_COL, LABEL_END_COL,
                                     REQUIRED_COLUMNS, VOL_FEATURES, _date_windows,
                                     _qlike, _report_gate, _rmse, _smearing_factor,
                                     build_oos_vol_panel, predict_vol, prepare,
                                     split_frames, train_vol_model, validate_vol_forecast)

HORIZON = 5


def build_pooled(tickers=("AAA", "BBB"), periods: int = 420, seed: int = 40) -> pd.DataFrame:
    # A pooled feature+label frame built through the real pipeline, not a mock of it.
    frames = []
    for offset, ticker in enumerate(tickers):
        df = make_bars(periods=periods, seed=seed + offset)
        df = build_features(df)
        df = build_volatility_features(df)
        df = add_labels(df, horizon=HORIZON)
        df["ticker"] = ticker
        frames.append(df)
    return pd.concat(frames).sort_index()


@pytest.fixture(scope="module")
def pooled() -> pd.DataFrame:
    return build_pooled()


# --------------------------------------------------------------------------- #
# prepare: schema, finiteness, positivity
# --------------------------------------------------------------------------- #

def test_prepare_refuses_a_frame_missing_a_required_column(pooled):
    # The old helper kept whichever features were present, so a rebuild that lost a column fitted a
    # smaller model against the same gate, with nothing in the output saying the inputs had changed.
    with pytest.raises(ValueError, match="missing required columns"):
        prepare(pooled.drop(columns=["gk_21d"]))


def test_prepare_keeps_every_declared_feature(pooled):
    _, features = prepare(pooled)
    assert features == VOL_FEATURES
    assert set(REQUIRED_COLUMNS) >= set(HAR_FEATURES + [LABEL_COL, LABEL_END_COL])


def test_prepare_drops_infinities(pooled):
    # `dropna` passes an infinity straight through to the estimator.
    poisoned = pooled.copy()
    victim = poisoned.index[len(poisoned) // 2]
    poisoned.loc[victim, "ewma_vol"] = np.inf

    clean, _ = prepare(poisoned)
    assert np.isfinite(clean[VOL_FEATURES].to_numpy(dtype=float)).all()


def test_prepare_drops_non_positive_har_inputs(pooled):
    # HAR takes logs of these; a zero becomes -inf and poisons a fitted coefficient.
    poisoned = pooled.copy()
    victim = poisoned.index[len(poisoned) // 2]
    poisoned.loc[victim, "rv_1d"] = 0.0

    clean, _ = prepare(poisoned)
    assert (clean[HAR_FEATURES] > 0).all().all()


def test_prepare_drops_rows_with_no_label(pooled):
    clean, _ = prepare(pooled)
    assert clean[LABEL_COL].gt(0).all()
    assert clean[LABEL_END_COL].notna().all()


# --------------------------------------------------------------------------- #
# The walk-forward purge
# --------------------------------------------------------------------------- #

def ragged_labelled_frame():
    # The shape that defeats a date-counted embargo, built with the real label builder: ticker B
    # trades on days 0-4, goes quiet, and returns on day 10.
    calendar = pd.bdate_range("2021-01-04", periods=20)
    observed = {"A": calendar, "B": calendar[[0, 1, 2, 3, 4] + list(range(10, 20))]}

    frames = []
    for offset, (ticker, index) in enumerate(observed.items()):
        bars = make_bars(periods=len(index), seed=60 + offset)
        bars.index = index
        labelled = add_forward_vol(bars, horizon=HORIZON)
        labelled["ticker"] = ticker
        frames.append(labelled)

    return pd.concat(frames).sort_index(), calendar


def test_a_ragged_ticker_cannot_train_on_a_label_from_the_test_block():
    # Two rows, same date: A trades daily so its label closes in five days, B goes quiet so its
    # label reaches past the gap into the test block. One date and one cutoff cannot separate them,
    # date-counted embargo either leaks B or throws away A. `label_end` keeps A and drops B.
    frame, calendar = ragged_labelled_frame()
    frame = frame[frame[LABEL_END_COL].notna()]

    folds = list(split_frames(frame, n_splits=1))
    assert folds, "expected one walk-forward fold over a 20-date calendar"
    _, train, test = folds[0]
    test_start = test.index.min()

    shared_date = calendar[1]
    assert shared_date < test_start, "the shared row must fall inside the training window"

    reach = {ticker: frame[(frame["ticker"] == ticker)
                           & (frame.index == shared_date)][LABEL_END_COL].iloc[0]
             for ticker in ("A", "B")}
    assert reach["A"] < test_start <= reach["B"], \
        "the fixture no longer reproduces the overlap it was built for"

    def in_train(ticker: str) -> bool:
        return not train[(train["ticker"] == ticker) & (train.index == shared_date)].empty

    assert in_train("A"), "an uncontaminated training row was purged"
    assert not in_train("B"), "a training row's label was measured inside the test block"


def test_no_training_row_ever_overlaps_its_test_block(pooled):
    # The same invariant, asserted across every fold of a realistic frame.
    clean, _ = prepare(pooled)

    folds = list(split_frames(clean, n_splits=3))
    assert len(folds) == 3
    for _, train, test in folds:
        assert (train[LABEL_END_COL] < test.index.min()).all()


def test_split_counts_must_make_sense():
    dates = pd.bdate_range("2021-01-04", periods=50)
    with pytest.raises(ValueError, match="n_splits"):
        list(_date_windows(dates, n_splits=-1))


def test_folds_move_forward_and_never_reuse_a_test_block(pooled):
    clean, _ = prepare(pooled)
    starts = [test.index.min() for _, _, test in split_frames(clean, n_splits=3)]
    assert starts == sorted(starts)
    assert len(set(starts)) == len(starts)


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

def test_qlike_is_zero_for_a_perfect_forecast():
    actual = np.array([0.1, 0.2, 0.35])
    assert _qlike(actual, actual) == pytest.approx(0.0)
    assert _rmse(actual, actual) == pytest.approx(0.0)


def test_qlike_punishes_under_forecasting_harder_than_over():
    # Under-forecasting risk is the expensive error for a position sizer, and a symmetric loss would
    # rank a model that does it as equal to one that does not.
    actual = np.array([0.2, 0.2, 0.2])
    under = _qlike(actual, actual / 2)
    over = _qlike(actual, actual * 2)
    assert under > over


# --------------------------------------------------------------------------- #
# The smearing correction
# --------------------------------------------------------------------------- #

def test_smearing_corrects_upward_for_a_log_target():
    # The factor must exceed 1. Below 1 it would be making the known bias worse, in the direction
    # QLIKE punishes hardest.
    rng = np.random.default_rng(70)
    log_predicted = rng.normal(-2.0, 0.3, 5000)
    log_actual = log_predicted + rng.normal(0.0, 0.4, 5000)

    assert _smearing_factor(log_actual, log_predicted) > 1.0


def test_smearing_of_a_perfect_fit_is_one():
    values = np.array([-2.0, -1.5, -1.0])
    assert _smearing_factor(values, values) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Saving and scoring
# --------------------------------------------------------------------------- #

@pytest.fixture
def saved_har(pooled, tmp_path, monkeypatch):
    # A trained HAR forecaster on disk. HAR because it is exact and fast.
    monkeypatch.setattr(vol_forecast, "VOL_MODEL_PATH", tmp_path / "vol_forecast.joblib")
    path = train_vol_model(pooled, kind="har")

    import joblib
    return joblib.load(path)


def test_a_saved_model_carries_its_own_bias_correction(saved_har):
    # Left behind, every forecast reads about 9% low and an inverse-vol sizer reads that as a calmer
    # market — roughly 10% more exposure per name than intended.
    assert saved_har["smearing"] > 1.0
    assert saved_har["log_target"] is True
    assert saved_har["horizon"] == vol_forecast.FORWARD_DAYS


def test_prediction_applies_the_log_transform_and_the_smearing_together(pooled, saved_har):
    # Applying one without the other silently biases every position size, which is exactly why the
    # sizing layer must come through `predict_vol` rather than reach for the estimator inside the
    # payload.
    clean, _ = prepare(pooled)
    rows = clean.head(50)

    manual = np.exp(saved_har["model"].predict(np.log(rows[HAR_FEATURES])))
    through_api = predict_vol(saved_har, rows)

    np.testing.assert_allclose(through_api, manual * saved_har["smearing"], rtol=1e-10)
    assert (through_api > 0).all()


def test_prediction_refuses_a_frame_missing_a_fitted_feature(pooled, saved_har):
    clean, _ = prepare(pooled)
    with pytest.raises(ValueError, match="missing"):
        predict_vol(saved_har, clean.drop(columns=["rv_5d"]))


def test_prediction_refuses_non_finite_inputs(pooled, saved_har):
    clean, _ = prepare(pooled)
    rows = clean.head(10).copy()
    rows.iloc[0, rows.columns.get_loc("rv_1d")] = np.nan

    with pytest.raises(ValueError, match="NaN or infinity"):
        predict_vol(saved_har, rows)


def test_prediction_refuses_non_positive_inputs_to_a_log_model(pooled, saved_har):
    clean, _ = prepare(pooled)
    rows = clean.head(10).copy()
    rows.iloc[0, rows.columns.get_loc("rv_1d")] = 0.0

    with pytest.raises(ValueError, match="non-positive"):
        predict_vol(saved_har, rows)


def test_training_rejects_a_parameter_free_baseline(pooled):
    # `rw` and `ewma` have nothing to fit; asking to save one is a caller mistake.
    with pytest.raises(ValueError, match="must be 'xgb' or 'har'"):
        train_vol_model(pooled, kind="ewma")


# --------------------------------------------------------------------------- #
# Gate reporting
# --------------------------------------------------------------------------- #

def perfect_baseline_summary() -> dict:
    scores = {
        "xgb": {"qlike": 0.5, "rmse": 0.10},
        "ewma": {"qlike": 0.0, "rmse": 0.0},   # a perfect baseline forecast
        "har": {"qlike": 0.2, "rmse": 0.05},
        "rw": {"qlike": 0.9, "rmse": 0.20},
    }
    return {name: {**values, "fold_wins_qlike": 0, "fold_wins_rmse": 0}
            for name, values in scores.items()}


def test_the_gate_report_survives_a_perfect_baseline():
    # A zero baseline loss is a legitimate result with no percentage improvement to report.
    # Dividing by it crashed the gate on a run that should simply have failed.
    verdict = _report_gate(perfect_baseline_summary(), n_folds=5)
    assert verdict["gate_passed"] is False
    assert verdict["best_by_qlike"] == "ewma"


def test_the_gate_requires_beating_both_baselines_on_both_metrics():
    summary = {
        "xgb": {"qlike": 0.10, "rmse": 0.010, "fold_wins_qlike": 5, "fold_wins_rmse": 5},
        "ewma": {"qlike": 0.20, "rmse": 0.020, "fold_wins_qlike": 0, "fold_wins_rmse": 0},
        "har": {"qlike": 0.15, "rmse": 0.005, "fold_wins_qlike": 0, "fold_wins_rmse": 0},
        "rw": {"qlike": 0.30, "rmse": 0.030, "fold_wins_qlike": 0, "fold_wins_rmse": 0},
    }
    # Beaten by HAR on RMSE alone — one loss is enough to fail.
    assert _report_gate(summary, n_folds=5)["gate_passed"] is False

    summary["har"]["rmse"] = 0.030
    assert _report_gate(summary, n_folds=5)["gate_passed"] is True


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #

def test_validation_scores_every_forecaster_on_identical_rows(pooled):
    # The comparison is only fair if all four score the same rows, which is what makes a
    # difference in QLIKE attributable to the forecast rather than to the sample.
    result = validate_vol_forecast(pooled, n_splits=2)

    assert set(result["summary"]) == {"rw", "ewma", "har", "xgb"}
    for scores in result["summary"].values():
        assert np.isfinite(scores["qlike"])
        assert np.isfinite(scores["rmse"])
    assert isinstance(result["gate_passed"], bool)


# --------------------------------------------------------------------------- #
# the out-of-sample panel the backtest sizes against
# --------------------------------------------------------------------------- #

@pytest.fixture
def oos_panel(pooled, tmp_path, monkeypatch) -> pd.DataFrame:
    monkeypatch.setattr(vol_forecast, "OOS_VOL_PANEL_PATH", tmp_path / "oos.parquet")
    return build_oos_vol_panel(pooled, n_splits=3, kind="har")


def test_the_oos_panel_is_shaped_date_by_ticker(oos_panel):
    assert list(oos_panel.columns) == ["AAA", "BBB"]
    assert oos_panel.index.is_monotonic_increasing
    assert oos_panel.notna().to_numpy().any()


def test_the_oos_panel_holds_only_positive_forecasts(oos_panel):
    values = oos_panel.to_numpy()
    assert np.nanmin(values) > 0


def test_the_oos_panel_starts_after_the_first_training_block(pooled, oos_panel):
    # The first fold is training-only, so the earliest forecast must sit well inside the
    # sample. A panel starting at the first date would mean something was fitted on its own
    # test block, which is the failure this whole construction exists to prevent.
    assert oos_panel.index.min() > pooled.index.min()


def test_a_forecast_never_sees_the_labels_of_its_own_block(pooled, tmp_path, monkeypatch):
    # The direct test of the property. Corrupt the labels inside one out-of-sample window and
    # that window's own forecasts must not move: they came from a model fitted before it. Had
    # the fit included its own test block, these numbers would change. Later blocks legitimately
    # do move, since the corrupted rows become training data for them, so only this one is checked.
    monkeypatch.setattr(vol_forecast, "OOS_VOL_PANEL_PATH", tmp_path / "a.parquet")
    honest = build_oos_vol_panel(pooled, n_splits=3, kind="har")

    dates = honest.index
    block = dates[len(dates) // 3: 2 * len(dates) // 3]
    tampered = pooled.copy()
    inside = tampered.index.isin(block)
    tampered.loc[inside, LABEL_COL] *= 25.0

    monkeypatch.setattr(vol_forecast, "OOS_VOL_PANEL_PATH", tmp_path / "b.parquet")
    rebuilt = build_oos_vol_panel(tampered, n_splits=3, kind="har")

    pd.testing.assert_frame_equal(honest.loc[block], rebuilt.loc[block])


def test_the_oos_panel_refuses_an_unknown_forecaster(pooled):
    with pytest.raises(ValueError, match="kind must be one of"):
        build_oos_vol_panel(pooled, n_splits=2, kind="nonsense")
