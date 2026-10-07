from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from amx.loss import anomaly_cost, err_gt_tol, missed_anomaly, one_minus_f1, zero_one

finite = st.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False)
labels = st.sampled_from(["n", "a", "b"])
texts = st.one_of(st.none(), st.text(alphabet="abc ", max_size=12))


@settings(max_examples=100, deadline=None)
@given(
    st.lists(st.tuples(finite, finite), min_size=1, max_size=40), st.floats(1e-6, 10), st.booleans()
)
def test_err_gt_tol_bounded_and_identity(
    pairs: list[tuple[float, float]], tol: float, rel: bool
) -> None:
    p, g = np.array(pairs).T
    loss = err_gt_tol(tol, relative=rel)
    out = loss(p, g)
    assert set(np.unique(out)) <= {0.0, 1.0}
    assert np.all(loss(g, g) == 0.0)


@settings(max_examples=100, deadline=None)
@given(st.lists(st.tuples(labels, labels), min_size=1, max_size=40))
def test_label_losses_bounded_and_identity(pairs: list[tuple[str, str]]) -> None:
    p, g = [a for a, _ in pairs], [b for _, b in pairs]
    for loss in (zero_one(), missed_anomaly("n"), anomaly_cost("n", 1.0, 0.3)):
        out = loss(p, g)
        assert np.all((out >= 0) & (out <= 1))
        assert np.all(loss(g, g) == 0.0)


@settings(max_examples=100, deadline=None)
@given(st.lists(st.tuples(texts, texts), min_size=1, max_size=30), st.booleans())
def test_f1_bounded_and_identity(pairs: list[tuple[str | None, str | None]], exact: bool) -> None:
    p, g = [a for a, _ in pairs], [b for _, b in pairs]
    loss = one_minus_f1(exact=exact)
    out = loss(p, g)
    assert np.all((out >= 0) & (out <= 1))
    assert np.all(loss(g, g) == 0.0)
