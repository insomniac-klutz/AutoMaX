from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression

from amx.baseline import (
    TIE_EPS,
    NotFittedError,
    ScoredPredictor,
    SignalScorer,
    TrivialClassifier,
    TrivialRegressor,
    fit_trivial,
    kfold_train_mask,
)
from amx.cert import TauGrid
from amx.data import UnitFrame
from amx.loss import err_gt_tol, zero_one
from amx.spec import parse_taskspec
from tests.unit.baseline.conftest import Split, classification_frame, outer_split

GRID = TauGrid().values


class FitSpy:
    """Records the ``tag`` values of every preprocessing fit and the rows of every model fit."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.prep_tags: list[set[float]] = []
        self.model_rows: list[int] = []
        prep_fit = ColumnTransformer.fit_transform
        lr_fit = LogisticRegression.fit
        hgb_fit = HistGradientBoostingRegressor.fit

        def spy_prep(est: Any, X: Any, y: Any = None, **kw: Any) -> Any:
            self.prep_tags.append(set(np.asarray(X["tag"], dtype=float).tolist()))
            return prep_fit(est, X, y, **kw)

        def spy_lr(est: Any, X: Any, y: Any, *a: Any, **kw: Any) -> Any:
            self.model_rows.append(X.shape[0])
            return lr_fit(est, X, y, *a, **kw)

        def spy_hgb(est: Any, X: Any, y: Any, *a: Any, **kw: Any) -> Any:
            self.model_rows.append(X.shape[0])
            return hgb_fit(est, X, y, *a, **kw)

        monkeypatch.setattr(ColumnTransformer, "fit_transform", spy_prep)
        monkeypatch.setattr(LogisticRegression, "fit", spy_lr)
        monkeypatch.setattr(HistGradientBoostingRegressor, "fit", spy_hgb)


def tags(uf: Any) -> np.ndarray:
    return np.asarray(uf.column("tag"), dtype=float)


def check_rows_seen(
    spy: FitSpy,
    split: Split,
    train_masks: list[np.ndarray],
    per_fold: int,
    folds: np.ndarray | None = None,
) -> None:
    fo = split.folds if folds is None else folds
    dev_tags = set(tags(split.dev).tolist())
    rest_tags = set(tags(split.rest).tolist())
    assert len(spy.prep_tags) == len(train_masks)
    assert spy.model_rows == [int(m.sum()) for m in train_masks for _ in range(per_fold)]
    for f, (seen, mask) in enumerate(zip(spy.prep_tags, train_masks, strict=True)):
        assert seen <= dev_tags
        assert not seen & rest_tags
        assert seen == set(tags(split.dev)[mask].tolist())
        assert not seen & set(tags(split.dev)[fo == f].tolist())


def test_classifier_fit_sees_only_dev_training_rows(
    clf_split: Split, monkeypatch: pytest.MonkeyPatch
) -> None:
    spy = FitSpy(monkeypatch)
    model = TrivialClassifier().fit(clf_split.dev, clf_split.folds, kfold_train_mask, zero_one(), 0)
    masks = [clf_split.folds != f for f in range(5)]
    check_rows_seen(spy, clf_split, masks, per_fold=1)
    assert sorted(model.oof_ids().tolist()) == sorted(clf_split.dev.ids.tolist())
    assert len(model.models_) == 5


def test_regressor_fit_sees_only_dev_training_rows(
    reg_split: Split, monkeypatch: pytest.MonkeyPatch
) -> None:
    spy = FitSpy(monkeypatch)
    loss = err_gt_tol(0.5)
    model = TrivialRegressor().fit(reg_split.dev, reg_split.folds, kfold_train_mask, loss, 0)
    masks = [reg_split.folds != f for f in range(3)]
    check_rows_seen(spy, reg_split, masks, per_fold=3)
    assert model.oof_losses().shape == (reg_split.dev.n,)


def test_training_only_units_are_never_predicted(monkeypatch: pytest.MonkeyPatch) -> None:
    split = outer_split(classification_frame(600, seed=31), k=3, seed=32)
    folds = np.where(np.arange(split.dev.n) < 90, -1, split.folds)

    def forward(fo: np.ndarray, f: int) -> np.ndarray:
        return (fo == -1) | ((fo >= 0) & (fo < f))

    spy = FitSpy(monkeypatch)
    model = TrivialClassifier().fit(split.dev, folds, forward, zero_one(), 0)
    check_rows_seen(spy, split, [forward(folds, f) for f in range(3)], per_fold=1, folds=folds)
    assert spy.prep_tags[0] == set(tags(split.dev)[:90].tolist())
    assert set(model.oof_ids().tolist()) == set(split.dev.ids[folds >= 0].tolist())


def _forward(fo: np.ndarray, f: int) -> np.ndarray:
    """Forward chaining: the training-only block plus every earlier fold."""
    return (fo == -1) | ((fo >= 0) & (fo < f))


def _single_class_first_fold() -> tuple[Split, UnitFrame, np.ndarray]:
    """Three classes on dev, but fold 0 trains on a training-only block of one class."""
    split = outer_split(classification_frame(600, seed=31), k=3, seed=32)
    df = split.dev.table.to_pandas()
    df.loc[(df.index >= 90) & (df["x2"] > 0.8), "y"] = "mid"
    df.loc[: 90 - 1, "y"] = "neg"
    folds = np.where(np.arange(split.dev.n) < 90, -1, split.folds)
    return split, UnitFrame.from_pandas(df, split.dev.roles), folds


def test_single_class_training_fold_gives_maximal_uncertainty(
    caplog: pytest.LogCaptureFixture,
) -> None:
    split, dev, folds = _single_class_first_fold()
    with caplog.at_level(logging.WARNING, logger="amx"):
        model = TrivialClassifier().fit(dev, folds, _forward, zero_one(), 0)
    assert model.classes_.tolist() == ["mid", "neg", "pos"]
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "fold 0" in warnings[0].getMessage()
    held0 = folds[folds >= 0] == 0  # aligned with oof_ids()
    assert held0.any()
    assert np.allclose(model.oof_signal_[held0], 1.0 - 1.0 / 3.0)
    assert np.all(model.oof_signal_[~held0] < 1.0 - 1.0 / 3.0)
    # The prior-only model is not deployed: the bag holds the two two-class fold models.
    assert len(model.models_) == 2
    assert all(len(m.classes_) >= 2 for m in model.models_)
    X = model.design_.frame(split.rest.inputs_pandas())
    mean_proba = np.mean([m.predict_proba(X) for m in model.models_], axis=0)
    assert np.allclose(model.signal(split.rest), 1.0 - mean_proba.max(axis=1))
    # Maximal uncertainty sorts those units last: they get the highest commit scores.
    s = model.oof_scores()
    assert np.min(s[held0]) >= np.max(s[~held0])


def test_classifier_refuses_when_no_fold_trains_on_two_classes() -> None:
    _, dev, folds = _single_class_first_fold()
    only_fold0 = np.where(folds >= 0, 0, -1)
    with pytest.raises(ValueError, match="two classes"):
        TrivialClassifier().fit(dev, only_fold0, _forward, zero_one(), 0)


def test_train_mask_overlapping_the_held_out_fold_is_refused(clf_split: Split) -> None:
    with pytest.raises(ValueError, match="contains units of fold"):
        TrivialClassifier().fit(
            clf_split.dev, clf_split.folds, lambda fo, f: np.ones(fo.size, bool), zero_one(), 0
        )
    with pytest.raises(ValueError, match="one entry per dev unit"):
        TrivialClassifier().fit(
            clf_split.dev, clf_split.folds[:-1], kfold_train_mask, zero_one(), 0
        )
    with pytest.raises(ValueError, match="target"):
        TrivialClassifier().fit(
            clf_split.dev.without_target(), clf_split.folds, kfold_train_mask, zero_one(), 0
        )


@pytest.fixture(scope="module")
def fitted(clf_split: Split, reg_split: Split) -> dict[str, tuple[ScoredPredictor, Split]]:
    clf = TrivialClassifier().fit(clf_split.dev, clf_split.folds, kfold_train_mask, zero_one(), 5)
    reg = TrivialRegressor().fit(
        reg_split.dev,
        reg_split.folds,
        kfold_train_mask,
        err_gt_tol(0.15, relative=True, abs_floor=1.0),
        5,
    )
    return {"classification": (clf, clf_split), "regression": (reg, reg_split)}


@pytest.mark.parametrize("family", ["classification", "regression"])
def test_commit_score_strictly_increasing_in_signal(
    family: str, fitted: dict[str, tuple[ScoredPredictor, Split]]
) -> None:
    model, split = fitted[family]
    u = model.signal(split.full)
    s = model.commit_score(split.full)
    assert np.all((s >= 0.0) & (s < 1.0))
    order = np.argsort(u, kind="stable")
    du, ds = np.diff(u[order]), np.diff(s[order])
    assert np.all(ds >= 0.0)
    assert isinstance(model, TrivialClassifier | TrivialRegressor)
    phi = u[order] / (u[order] + model.scorer_.scale_)
    resolvable = np.diff(phi) * TIE_EPS > 1e-15  # above float64 spacing of s in [0, 1)
    assert resolvable.mean() > 0.95
    assert np.all(ds[resolvable] > 0.0)
    assert np.all(ds[du == 0.0] == 0.0)


def test_scorer_breaks_isotonic_ties_by_the_signal() -> None:
    u = np.linspace(0.0, 1.0, 401)
    losses = (u > 0.5).astype(float)
    scorer = SignalScorer().fit(u, losses)
    probe = np.linspace(0.0, 50.0, 5001)
    s = scorer(probe)
    assert np.all(np.diff(s) > 0.0)
    assert np.all((s >= 0.0) & (s < 1.0))
    iso = scorer.iso_.predict(probe)
    assert np.max(np.abs(s - iso)) <= 2 * TIE_EPS
    with pytest.raises(ValueError):
        scorer(np.array([-0.1]))
    with pytest.raises(NotFittedError):
        SignalScorer()(np.array([0.1]))


def test_scorer_breaks_ties_for_signals_far_above_one() -> None:
    """Unnormalised signals (widths, vol·√h) on the scale 1e2 to 1e3 still break ties."""
    rng = np.random.default_rng(4)
    u = rng.uniform(100.0, 1000.0, size=2000)
    losses = (rng.uniform(size=u.size) < 0.2 + 0.5 * (u > 550.0)).astype(float)
    scorer = SignalScorer().fit(u, losses)
    assert scorer.scale_ == float(np.median(u))
    probe = np.linspace(100.0, 1000.0, 90_001)  # steps of 0.01, far above float spacing of u
    s = scorer(probe)
    assert np.all(np.diff(s) > 0.0)
    assert np.all((s >= 0.0) & (s < 1.0))
    assert np.max(np.abs(s - scorer.iso_.predict(probe))) <= 2 * TIE_EPS


def test_scorer_scale_ignores_zero_signals_and_falls_back_to_one() -> None:
    u = np.array([0.0, 0.0, 0.0, 2.0, 4.0, 9.0])
    assert SignalScorer().fit(u, np.zeros(u.size)).scale_ == 4.0
    zeros = SignalScorer().fit(np.zeros(5), np.array([0.0, 1.0, 0.0, 1.0, 0.0]))
    assert zeros.scale_ == 1.0
    s = zeros(np.array([0.0, 0.5, 1.0, 3.0]))
    assert np.all(np.diff(s) > 0.0)


@pytest.mark.parametrize("family", ["classification", "regression"])
def test_dev_cov_non_decreasing(
    family: str, fitted: dict[str, tuple[ScoredPredictor, Split]]
) -> None:
    model, _ = fitted[family]
    cov = model.dev_cov(GRID)
    assert cov.shape == GRID.shape
    assert np.all(np.diff(cov) >= 0.0)
    assert cov[-1] == 1.0
    assert model.dev_cov([-1.0])[0] == 0.0
    s = model.oof_scores()
    assert np.allclose(cov, [(s <= t).mean() for t in GRID])
    groups = np.arange(s.size) // 4
    gcov = model.dev_cov(GRID, groups=groups)
    assert np.all(np.diff(gcov) >= 0.0)
    gmin = np.array([s[groups == g].min() for g in np.unique(groups)])
    assert np.allclose(gcov, [(gmin <= t).mean() for t in GRID])


def test_classifier_bag_predicts_labels(fitted: dict[str, tuple[ScoredPredictor, Split]]) -> None:
    model, split = fitted["classification"]
    assert isinstance(model, TrivialClassifier)
    pred = model.predict(split.rest)
    assert set(pred.tolist()) <= {"pos", "neg"}
    held_out_error = float(np.mean(pred != split.rest.target))
    oof = model.oof_losses()
    assert set(np.unique(oof).tolist()) <= {0.0, 1.0}
    assert held_out_error < 0.3
    assert abs(float(np.mean(oof)) - held_out_error) < 0.06
    # The bag averages the fold models' probabilities (OQ Q13 default (a)).
    X = model.design_.frame(split.rest.inputs_pandas())
    mean_proba = np.mean([m.predict_proba(X) for m in model.models_], axis=0)
    assert np.allclose(model.signal(split.rest), 1.0 - mean_proba.max(axis=1))


def test_regressor_relative_signal_uses_loss_params(
    fitted: dict[str, tuple[ScoredPredictor, Split]],
) -> None:
    model, split = fitted["regression"]
    assert isinstance(model, TrivialRegressor)
    assert model.relative_ and model.abs_floor_ == 1.0
    point = model.predict(split.rest)
    absolute = TrivialRegressor()
    absolute.__dict__.update(model.__dict__)
    absolute.relative_ = False
    width = absolute.signal(split.rest)
    expected = width / np.maximum(np.abs(point), 1.0)
    assert np.allclose(model.signal(split.rest), expected)


def _spec(family: str) -> Any:
    task: dict[str, Any] = {"family": family}
    if family == "extraction":
        task["commit_unit"] = "field"
    return parse_taskspec(
        {
            "spec_version": 1,
            "name": "t",
            "data": {
                "uri": "x",
                "format": "csv",
                "unit_id": "id",
                "target": {"name": "y", "kind": "categorical"},
            },
            "task": task,
            "bands": {"alphas": [0.05], "policies": ["auto"]},
        }
    )


def test_fit_trivial_dispatch(clf_split: Split) -> None:
    model = fit_trivial(
        _spec("classification"), clf_split.dev, clf_split.folds, kfold_train_mask, zero_one(), 0
    )
    assert isinstance(model, TrivialClassifier)
    assert isinstance(model, ScoredPredictor)
    with pytest.raises(NotImplementedError, match="A1"):
        fit_trivial(
            _spec("extraction"), clf_split.dev, clf_split.folds, kfold_train_mask, zero_one(), 0
        )
    with pytest.raises(NotFittedError):
        _ = TrivialClassifier().roles


@pytest.mark.parametrize("family", ["classification", "regression"])
def test_outputs_do_not_depend_on_the_batch(
    family: str, fitted: dict[str, tuple[ScoredPredictor, Split]]
) -> None:
    model, split = fitted[family]
    batch_values = model.predict(split.rest)
    batch_scores = model.commit_score(split.rest)
    for i in (0, 7, split.rest.n - 1):
        one = split.rest.take([i])
        assert model.predict(one)[0] == batch_values[i]
        assert model.commit_score(one)[0] == batch_scores[i]
    rev = np.arange(split.rest.n)[::-1]
    assert np.array_equal(model.commit_score(split.rest.take(rev)), batch_scores[rev])


@pytest.mark.parametrize("family", ["classification", "regression"])
def test_empty_batch_gives_empty_typed_outputs(
    family: str, fitted: dict[str, tuple[ScoredPredictor, Split]]
) -> None:
    model, split = fitted[family]
    empty = split.rest.take(np.empty(0, dtype=np.int64))
    values = model.predict(empty)
    assert values.shape == (0,)
    assert values.dtype == model.predict(split.rest.take([0])).dtype
    for out in (model.signal(empty), model.commit_score(empty)):
        assert out.shape == (0,) and out.dtype == np.float64
    assert isinstance(model, TrivialClassifier | TrivialRegressor)
    assert model.design_ is not None
    no_x1 = empty.inputs_pandas().drop(columns=["x1"])
    with pytest.raises(ValueError, match="missing columns"):
        model.design_.frame(no_x1)


def test_scorer_of_an_empty_signal_is_empty() -> None:
    scorer = SignalScorer().fit(np.array([0.0, 1.0, 2.0]), np.array([0.0, 0.0, 1.0]))
    s = scorer(np.empty(0))
    assert s.shape == (0,) and s.dtype == np.float64
