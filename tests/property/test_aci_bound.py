"""Deterministic ACI bound under adversarial error sequences (C12, HANDOFF A.4).

The adversary picks err_t freely only while 0 < α_t < 1; at α_t ≤ 0 (infinite interval) the
error is forced to 0 and at α_t ≥ 1 (empty interval) to 1. For every horizon h in {1, 3, 24}
the long-run miscoverage of the errors whose feedback entered the updates must stay within
:func:`amx.cert.aci_bound`, each side within its one-sided bound
(:func:`amx.cert.aci_side_bounds`, D20), and α_t must stay in [−γh, 1 + γh].
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from amx.cert import aci_alphas, aci_bound, aci_side_bounds

Policy = Callable[[int, float], int]


def _policy(kind: str, seed: int, p: float, theta: float, period: int) -> Policy:
    if kind == "bernoulli":
        bits = np.random.default_rng(seed).uniform(size=10_000) < p
        return lambda t, a: int(bits[t])
    if kind == "always_miss":
        return lambda t, a: 1
    if kind == "always_cover":
        return lambda t, a: 0
    if kind == "threshold":  # miss while α is high, cover while it is low: chases θ
        return lambda t, a: int(a > theta)
    if kind == "anti_threshold":  # the opposite: pushes α towards the boundaries
        return lambda t, a: int(a < theta)
    if kind == "periodic":
        return lambda t, a: int((t // period) % 2 == 0)
    raise AssertionError(kind)


KINDS = ["bernoulli", "always_miss", "always_cover", "threshold", "anti_threshold", "periodic"]


@settings(max_examples=300, deadline=None)
@given(
    h=st.sampled_from([1, 3, 24]),
    gamma=st.floats(1e-3, 0.3),
    alpha_target=st.floats(0.01, 0.99),
    alpha0=st.floats(0.0, 1.0),
    extra=st.integers(1, 3000),
    kind=st.sampled_from(KINDS),
    seed=st.integers(0, 2**32 - 1),
    p=st.floats(0.0, 1.0),
    theta=st.floats(-0.1, 1.1),
    period=st.integers(1, 200),
)
def test_aci_bound_holds_for_any_adversary(
    h: int,
    gamma: float,
    alpha_target: float,
    alpha0: float,
    extra: int,
    kind: str,
    seed: int,
    p: float,
    theta: float,
    period: int,
) -> None:
    T = h + extra
    tr = aci_alphas(_policy(kind, seed, p, theta, period), alpha_target, gamma, T, h, alpha0=alpha0)
    # range claim of the derivation
    assert float(np.min(tr.alpha)) >= -gamma * h - 1e-12
    assert float(np.max(tr.alpha)) <= 1.0 + gamma * h + 1e-12
    # forced errors at the boundaries
    assert np.all(tr.err[tr.alpha <= 0.0] == 0)
    assert np.all(tr.err[tr.alpha >= 1.0] == 1)
    # the bound on the errors whose feedback entered the updates
    dev = abs(tr.feedback_miscoverage() - alpha_target)
    assert dev <= aci_bound(alpha0, gamma, T, h) + 1e-9
    # and each side within its own bound (D20)
    over, under = aci_side_bounds(alpha0, gamma, T, h)
    signed = tr.feedback_miscoverage() - alpha_target
    assert -under - 1e-9 <= signed <= over + 1e-9


@settings(max_examples=60, deadline=None)
@given(h=st.sampled_from([1, 3, 24]), seed=st.integers(0, 2**32 - 1))
def test_aci_bound_at_the_default_step_size(h: int, seed: int) -> None:
    """γ = 0.005, α = 0.1, T = 2000: the regime T1-forecast runs in."""
    bits = np.random.default_rng(seed).uniform(size=2000) < 0.3
    tr = aci_alphas(lambda t, a: int(bits[t]), 0.1, 0.005, 2000, h)
    b = aci_bound(0.1, 0.005, 2000, h)
    assert abs(tr.feedback_miscoverage() - 0.1) <= b + 1e-9
    if h == 1:
        assert abs(b - 0.0905) < 1e-3  # the classic Gibbs-Candès scale


def test_bound_is_nearly_attained() -> None:
    """The adversary that always misses from α_0 = 1 gets within a few γ of the bound."""
    T, g = 400, 0.05
    tr = aci_alphas(lambda t, a: 1, 0.5, g, T, 1, alpha0=1.0)
    dev = abs(tr.feedback_miscoverage() - 0.5)
    assert dev <= aci_bound(1.0, g, T, 1)
    assert dev >= aci_bound(1.0, g, T, 1) - 2 * g / (g * (T - 1)) - 1e-12
