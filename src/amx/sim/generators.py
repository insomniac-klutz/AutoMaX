"""Synthetic generators with a known conditional risk (HANDOFF 12 T1-synth, C2, C6).

Every selective generator fixes the predictor and the commit score in advance (they play the
role of a graph and scorer fitted on synthetic dev) and exposes:

* ``sample(n, rng) -> SimBatch``: calibration draws with sampled 0/1 losses;
* ``oracle(grid, n_mc, rng) -> OracleCurve``: population coverage and selective risk from
  Rao–Blackwell averaging of the KNOWN conditional risk r(x), never from sampled labels;
* ``dev_cov(grid)``: the coverage curve LTT uses to pick its start point, taken from the
  oracle with a pre-registered seed, so it is independent of every calibration draw.

Parameters are fixed here and documented per class; changing them is a decision-log change.
Generator (c) (:class:`SeasonalARShift`) is a forecasting stream for the T1-forecast path
only: no distribution-free guarantee survives its regime shift, so it is never gated on δ_j
(OQ Q6). Its STATIONARY variant (:meth:`SeasonalARShift.stationary`, same law without the
shift) carries the T1-forecast gate; the shifted path is reported, not gated (D20).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import expit, ndtr

from amx.sim.oracle import CurveSums, OracleCurve, curve_from_sums, group_sums, unit_sums

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

GAUSSIAN_MIXTURE_SEED = 20_261_001
"""Seed of the fixed plug-in mean perturbation of generator (a)."""

DEV_COV_SEED = 20_261_010
"""Seed of the oracle run that provides ``dev_cov`` when no oracle is passed in."""

DEFAULT_N_MC = 2_000_000
"""Default Monte Carlo size of unit-level oracles (inputs)."""

DEFAULT_N_MC_GROUPS = 400_000
"""Default Monte Carlo size of the group-level oracle (groups; C6 asks for at least 2e5)."""

_CHUNK = 500_000

STATIONARY_NAME = "seasonal_ar_stationary"
"""Name of the stationary variant of generator (c) (D20)."""


@dataclass(frozen=True)
class SimBatch:
    """One synthetic calibration draw.

    ``losses`` are sampled 0/1 losses of the fixed predictor; ``cond_risk`` is the known
    r(x) = P(loss = 1 | x) of each unit (tests only; the certifier never sees it).
    ``groups`` holds integer group ids for clustered generators, else None.
    """

    scores: FloatArray
    losses: FloatArray
    cond_risk: FloatArray
    groups: IntArray | None = None

    @property
    def n_units(self) -> int:
        return int(self.scores.shape[0])


@runtime_checkable
class SelectiveGenerator(Protocol):
    """What :func:`amx.sim.t1_synth.run_t1_synth` needs from a generator."""

    @property
    def name(self) -> str: ...

    @property
    def grouped(self) -> bool: ...

    @property
    def default_n_mc(self) -> int: ...

    def sample(self, n: int, rng: np.random.Generator) -> SimBatch: ...

    def oracle(self, grid: ArrayLike, n_mc: int, rng: np.random.Generator) -> OracleCurve: ...

    def dev_cov(self, grid: ArrayLike, oracle: OracleCurve | None = None) -> FloatArray: ...


def _check_n(n: int) -> None:
    if n < 1:
        raise ValueError("n must be >= 1")


class _UnitOracleMixin:
    """Chunked Rao–Blackwell oracle for generators with one independence unit per input."""

    default_n_mc: int = DEFAULT_N_MC

    def _inputs(self, n: int, rng: np.random.Generator) -> tuple[FloatArray, FloatArray]:
        """(scores, cond_risk) of n fresh inputs from the population."""
        raise NotImplementedError

    def oracle(self, grid: ArrayLike, n_mc: int, rng: np.random.Generator) -> OracleCurve:
        tau = np.asarray(grid, dtype=np.float64).reshape(-1)
        _check_n(n_mc)
        sums = CurveSums.zeros(tau.shape[0])
        done = 0
        while done < n_mc:
            m = min(_CHUNK, n_mc - done)
            s, r = self._inputs(m, rng)
            sums = sums + unit_sums(s, r, tau)
            done += m
        return curve_from_sums(tau, sums, n_mc, n_mc=n_mc)

    def dev_cov(self, grid: ArrayLike, oracle: OracleCurve | None = None) -> FloatArray:
        if oracle is None:
            oracle = self.oracle(grid, self.default_n_mc, np.random.default_rng(DEV_COV_SEED))
        else:
            oracle.check_grid(grid)
        return np.asarray(oracle.cov, dtype=np.float64)


@dataclass(frozen=True)
class GaussianMixture(_UnitOracleMixin):
    """Generator (a): K-class Gaussian mixture with a miscalibrated plug-in classifier.

    * K = 4 classes in 2-D, means μ_k = radius·(cos 2πk/K, sin 2πk/K) with radius 1.5,
      identity covariance, equal priors. True posterior p(k | x) = softmax_k(μ_k·x − ½|μ_k|²).
    * Plug-in classifier: means m_k = μ_k + N(0, 0.15²) (fixed draw from ``seed``), logits
      (m_k·x − ½|m_k|²) / temperature with temperature 1.6 (under-confident).
    * ŷ = argmax of the plug-in posterior; commit score s = 1 − max plug-in probability;
      known conditional risk r(x) = 1 − p(ŷ | x).
    """

    n_classes: int = 4
    radius: float = 1.5
    perturb_sd: float = 0.15
    temperature: float = 1.6
    seed: int = GAUSSIAN_MIXTURE_SEED
    name: str = "gaussian_mixture"
    grouped: bool = False
    _plugin: FloatArray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.n_classes < 2 or self.temperature <= 0 or self.perturb_sd < 0:
            raise ValueError("need n_classes >= 2, temperature > 0, perturb_sd >= 0")
        noise = np.random.default_rng(self.seed).normal(0.0, self.perturb_sd, (self.n_classes, 2))
        object.__setattr__(self, "_plugin", self.means + noise)

    @property
    def means(self) -> FloatArray:
        ang = 2.0 * np.pi * np.arange(self.n_classes) / self.n_classes
        return np.asarray(self.radius * np.column_stack([np.cos(ang), np.sin(ang)]))

    @property
    def plugin_means(self) -> FloatArray:
        return np.asarray(self._plugin, dtype=np.float64)

    @staticmethod
    def _softmax(z: FloatArray) -> FloatArray:
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return np.asarray(e / e.sum(axis=1, keepdims=True), dtype=np.float64)

    def true_posterior(self, x: ArrayLike) -> FloatArray:
        mu = self.means
        xx = np.asarray(x, dtype=np.float64).reshape(-1, 2)
        return self._softmax(xx @ mu.T - 0.5 * np.sum(mu * mu, axis=1))

    def plugin_posterior(self, x: ArrayLike) -> FloatArray:
        m = self.plugin_means
        xx = np.asarray(x, dtype=np.float64).reshape(-1, 2)
        return self._softmax((xx @ m.T - 0.5 * np.sum(m * m, axis=1)) / self.temperature)

    def predict(self, x: ArrayLike) -> tuple[IntArray, FloatArray, FloatArray]:
        """(ŷ, commit score s, known conditional risk r) for inputs x of shape (n, 2)."""
        ph = self.plugin_posterior(x)
        yhat = np.argmax(ph, axis=1).astype(np.int64)
        s = 1.0 - ph[np.arange(ph.shape[0]), yhat]
        pt = self.true_posterior(x)
        r = 1.0 - pt[np.arange(pt.shape[0]), yhat]
        return yhat, np.asarray(s, np.float64), np.asarray(np.clip(r, 0.0, 1.0), np.float64)

    def draw(self, n: int, rng: np.random.Generator) -> tuple[FloatArray, IntArray]:
        """n labelled inputs (x, y) from the mixture."""
        _check_n(n)
        y = rng.integers(0, self.n_classes, size=n).astype(np.int64)
        x = self.means[y] + rng.standard_normal((n, 2))
        return np.asarray(x, dtype=np.float64), y

    def _inputs(self, n: int, rng: np.random.Generator) -> tuple[FloatArray, FloatArray]:
        x, _ = self.draw(n, rng)
        _, s, r = self.predict(x)
        return s, r

    def sample(self, n: int, rng: np.random.Generator) -> SimBatch:
        x, y = self.draw(n, rng)
        yhat, s, r = self.predict(x)
        return SimBatch(scores=s, losses=(yhat != y).astype(np.float64), cond_risk=r)


@dataclass(frozen=True)
class HeteroscedasticRegression(_UnitOracleMixin):
    """Generator (b): heteroscedastic regression with a biased predictor and a tolerance loss.

    * x ~ U(0, 1), f(x) = sin(2πx), σ(x) = 0.1 + 0.4x, y = f(x) + σ(x)·ε with ε ~ N(0, 1).
    * Predictor f̂(x) = f(x) + b(x) with bias b(x) = 0.05·cos(3x); loss 1[|f̂ − y| > tol],
      tol = 0.3. Known risk r(x) = 1 − [Φ((tol − b)/σ) − Φ((−tol − b)/σ)].
    * Commit score s(x) = 2Φ(−tol / (0.8·σ(x))): the scorer under-states the noise by 20% and
      ignores the bias. Note min_x r(x) ≈ 0.0064, so the 0.5% band is infeasible by design.
    """

    tol: float = 0.3
    sigma0: float = 0.1
    sigma_slope: float = 0.4
    bias_amp: float = 0.05
    bias_freq: float = 3.0
    scorer_sigma_factor: float = 0.8
    name: str = "heteroscedastic"
    grouped: bool = False

    def sigma(self, x: ArrayLike) -> FloatArray:
        return np.asarray(self.sigma0 + self.sigma_slope * np.asarray(x, dtype=np.float64))

    def bias(self, x: ArrayLike) -> FloatArray:
        return np.asarray(self.bias_amp * np.cos(self.bias_freq * np.asarray(x, np.float64)))

    def cond_risk(self, x: ArrayLike) -> FloatArray:
        sg, b = self.sigma(x), self.bias(x)
        # 1 − [Φ(a) − Φ(c)] written as Φ(−a) + Φ(c) to keep precision in the tails.
        return np.asarray(ndtr(-(self.tol - b) / sg) + ndtr((-self.tol - b) / sg))

    def score(self, x: ArrayLike) -> FloatArray:
        return np.asarray(2.0 * ndtr(-self.tol / (self.scorer_sigma_factor * self.sigma(x))))

    def _inputs(self, n: int, rng: np.random.Generator) -> tuple[FloatArray, FloatArray]:
        x = rng.uniform(0.0, 1.0, n)
        return self.score(x), self.cond_risk(x)

    def sample(self, n: int, rng: np.random.Generator) -> SimBatch:
        _check_n(n)
        x = rng.uniform(0.0, 1.0, n)
        y = np.sin(2.0 * np.pi * x) + self.sigma(x) * rng.standard_normal(n)
        fhat = np.sin(2.0 * np.pi * x) + self.bias(x)
        losses = (np.abs(fhat - y) > self.tol).astype(np.float64)
        return SimBatch(scores=self.score(x), losses=losses, cond_risk=self.cond_risk(x))


@dataclass(frozen=True)
class ClusteredUnits:
    """Generator (d): clustered units, certified at GROUP level (OQ Q2).

    * Group size 1 + Poisson(4); per unit x ~ N(0, 1); per group u_g ~ N(0, 0.8²).
    * Known risk r(x, u) = expit(−3 + 1.2x + u_g); the loss is Bernoulli(r).
    * Commit score s = expit(−3.2 + 1.0x): it does not see u_g, so losses within a group are
      correlated given the scores.
    * ``sample(n, rng)`` draws n GROUPS (the independence unit; n_calib counts groups).
    * Oracle: group-weighted risk E[mean_{i ∈ C_g(τ)} r(x_i, u_g) | |C_g(τ)| ≥ 1] and the share
      of groups with ≥ 1 committed unit, by Rao–Blackwell Monte Carlo over n_mc groups; the
      unit-weighted risk and unit coverage are kept alongside.
    """

    mean_extra_size: float = 4.0
    u_sd: float = 0.8
    risk_intercept: float = -3.0
    risk_slope: float = 1.2
    score_intercept: float = -3.2
    score_slope: float = 1.0
    name: str = "clustered"
    grouped: bool = True
    default_n_mc: int = DEFAULT_N_MC_GROUPS

    def draw(
        self, n_groups: int, rng: np.random.Generator
    ) -> tuple[FloatArray, FloatArray, IntArray]:
        """(scores, cond_risk, group ids) of the units of n_groups fresh groups."""
        _check_n(n_groups)
        sizes = 1 + rng.poisson(self.mean_extra_size, n_groups)
        gid = np.repeat(np.arange(n_groups, dtype=np.int64), sizes)
        x = rng.standard_normal(gid.shape[0])
        u = rng.normal(0.0, self.u_sd, n_groups)[gid]
        r = expit(self.risk_intercept + self.risk_slope * x + u)
        s = expit(self.score_intercept + self.score_slope * x)
        return np.asarray(s, np.float64), np.asarray(r, np.float64), gid

    def sample(self, n: int, rng: np.random.Generator) -> SimBatch:
        s, r, gid = self.draw(n, rng)
        losses = (rng.uniform(size=r.shape[0]) < r).astype(np.float64)
        return SimBatch(scores=s, losses=losses, cond_risk=r, groups=gid)

    def oracle(self, grid: ArrayLike, n_mc: int, rng: np.random.Generator) -> OracleCurve:
        tau = np.asarray(grid, dtype=np.float64).reshape(-1)
        _check_n(n_mc)
        g_sums = CurveSums.zeros(tau.shape[0])
        u_sums = CurveSums.zeros(tau.shape[0])
        n_units = 0
        done = 0
        chunk = max(1, _CHUNK // 5)
        while done < n_mc:
            m = min(chunk, n_mc - done)
            s, r, gid = self.draw(m, rng)
            g_sums = g_sums + group_sums(s, r, gid, tau)
            u_sums = u_sums + unit_sums(s, r, tau)
            n_units += int(s.shape[0])
            done += m
        return curve_from_sums(
            tau,
            g_sums,
            n_mc,
            n_mc=n_mc,
            estimand="group_weighted",
            unit=u_sums,
            n_units=n_units,
        )

    def dev_cov(self, grid: ArrayLike, oracle: OracleCurve | None = None) -> FloatArray:
        """Share of groups with ≥ 1 committed unit (the certified independence unit)."""
        if oracle is None:
            oracle = self.oracle(grid, self.default_n_mc, np.random.default_rng(DEV_COV_SEED))
        else:
            oracle.check_grid(grid)
        return np.asarray(oracle.cov, dtype=np.float64)


@dataclass(frozen=True)
class ForecastStream:
    """Point forecasts issued at ``origins`` for ``origins + horizon`` and the realised targets."""

    horizon: int
    origins: IntArray
    preds: FloatArray
    targets: FloatArray

    @property
    def abs_resid(self) -> FloatArray:
        return np.asarray(np.abs(self.targets - self.preds), dtype=np.float64)


@dataclass(frozen=True)
class SeasonalARShift:
    """Generator (c): AR(1) + seasonal series with a regime shift (T1-forecast path only).

    * y_t = A·sin(2πt / P) + z_t, z_t = φ z_{t-1} + ε_t, φ = 0.6, period P = 24, amplitude
      A = 1.0, ε_t ~ N(0, σ_t²) with σ_t = 0.5 before the shift index and 2 × 0.5 = 1.0 from it
      on (the noise sd doubles). z_0 is drawn from the stationary law of the first regime.
    * Point forecast: seasonal naive, ŷ_{o+h} = y_{o+h−L}, L = P·⌈h/P⌉; it uses history ≤ o.
      Its error is e = z_{o+h} − z_{o+h−L}.
    * Label-free selective score at origin o (a scorer "fitted on dev": it knows A, P and φ
      of the first regime): with z_t = y_t − A·sin(2πt/P) observed up to o,
      e | history ~ N(μ, v), μ = φ^h z_o − z_{o+h−L}, v = σ̂_o² (1 − φ^{2h}) / (1 − φ²), where
      σ̂_o is the RMS of the last ``vol_window`` innovations z_t − φ z_{t−1}, t ≤ o. The score is
      the implied exceedance probability s_o = P(|e| > tol) = Φ((−tol − μ)/√v) + Φ((μ − tol)/√v).

    No distribution-free guarantee survives the shift; the generator is never gated on δ_j.
    :meth:`stationary` gives the same law with ``shift_factor`` = 1 (no shift): the path the
    T1-forecast gate runs on (D20).
    """

    phi: float = 0.6
    period: int = 24
    amplitude: float = 1.0
    noise_sd: float = 0.5
    shift_factor: float = 2.0
    vol_window: int = 72
    name: str = "seasonal_ar_shift"

    @property
    def has_shift(self) -> bool:
        """True unless the noise sd stays the same across the shift index."""
        return self.shift_factor != 1.0

    def stationary(self) -> SeasonalARShift:
        """The stationary variant of generator (c): the same law without a regime shift (D20)."""
        return replace(self, shift_factor=1.0, name=STATIONARY_NAME)

    def season(self, t: ArrayLike) -> FloatArray:
        tt = np.asarray(t, dtype=np.float64)
        return np.asarray(self.amplitude * np.sin(2.0 * np.pi * tt / self.period))

    def lag(self, horizon: int) -> int:
        """Seasonal-naive look-back L = P·⌈h/P⌉."""
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        return self.period * (-(-horizon // self.period))

    def series(self, n_time: int, shift_at: int, rng: np.random.Generator) -> FloatArray:
        """One path y_0 … y_{n_time−1}; the noise sd doubles from index ``shift_at`` on."""
        if n_time < 2 or not (0 <= shift_at <= n_time):
            raise ValueError("need n_time >= 2 and 0 <= shift_at <= n_time")
        sd = np.where(
            np.arange(n_time) < shift_at, self.noise_sd, self.noise_sd * self.shift_factor
        )
        eps = rng.standard_normal(n_time) * sd
        z = np.empty(n_time, dtype=np.float64)
        z[0] = rng.standard_normal() * sd[0] / np.sqrt(1.0 - self.phi**2)
        for t in range(1, n_time):
            z[t] = self.phi * z[t - 1] + eps[t]
        return np.asarray(self.season(np.arange(n_time)) + z, dtype=np.float64)

    def min_origin(self, horizon_max: int) -> int:
        """Smallest origin at which every forecast and the selective score are defined."""
        return max(self.lag(horizon_max) - 1, self.vol_window)

    def forecast(self, y: ArrayLike, origins: ArrayLike, horizon: int) -> ForecastStream:
        """Seasonal-naive forecasts issued at ``origins`` for ``origins + horizon``."""
        yy = np.asarray(y, dtype=np.float64).reshape(-1)
        o = np.asarray(origins, dtype=np.int64).reshape(-1)
        src = o + horizon - self.lag(horizon)
        if o.size and (int(np.min(src)) < 0 or int(np.max(o)) + horizon >= yy.shape[0]):
            raise ValueError("origins out of range for this horizon")
        return ForecastStream(
            horizon=horizon, origins=o, preds=yy[src].copy(), targets=yy[o + horizon].copy()
        )

    def exceed_scores(
        self, y: ArrayLike, origins: ArrayLike, horizon: int, tol: float
    ) -> FloatArray:
        """Implied P(|e| > tol | y_{≤o}) for each origin (uses no value after the origin)."""
        yy = np.asarray(y, dtype=np.float64).reshape(-1)
        o = np.asarray(origins, dtype=np.int64).reshape(-1)
        L, W = self.lag(horizon), self.vol_window
        if tol <= 0:
            raise ValueError("tol must be > 0")
        if o.size and (int(np.min(o)) < max(W, L - horizon) or int(np.max(o)) >= yy.shape[0]):
            raise ValueError("origins out of range for the score")
        z = yy - self.season(np.arange(yy.shape[0]))
        innov2 = np.zeros(yy.shape[0], dtype=np.float64)
        innov2[1:] = (z[1:] - self.phi * z[:-1]) ** 2
        cs = np.concatenate([[0.0], np.cumsum(innov2)])
        sig2 = (cs[o + 1] - cs[o + 1 - W]) / W
        v = sig2 * (1.0 - self.phi ** (2 * horizon)) / (1.0 - self.phi**2)
        sd = np.sqrt(np.maximum(v, 1e-24))
        mu = self.phi**horizon * z[o] - z[o + horizon - L]
        return np.asarray(ndtr((-tol - mu) / sd) + ndtr((mu - tol) / sd), dtype=np.float64)


GENERATORS: dict[str, type] = {
    "a": GaussianMixture,
    "b": HeteroscedasticRegression,
    "c": SeasonalARShift,
    "d": ClusteredUnits,
}
"""Generator letters of HANDOFF 12 T1-synth."""


def selective_generator(key: str) -> SelectiveGenerator:
    """Instantiate generator (a), (b) or (d) with its documented parameters."""
    if key == "a":
        return GaussianMixture()
    if key == "b":
        return HeteroscedasticRegression()
    if key == "d":
        return ClusteredUnits()
    raise ValueError(f"no selective generator {key!r}; (c) is forecasting-only (Q6)")
