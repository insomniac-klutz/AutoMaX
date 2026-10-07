"""Rolling-origin evaluation indices with an embargo (HANDOFF 7.5, ROLLER step 7).

Index conventions (time indices 0 … n_time − 1):

* ``train_end`` is the exclusive end of the fitting window: only observations with index
  < ``train_end`` (targets included) may be used to fit or calibrate.
* ``origin`` is the forecast origin: the last observed index when the forecast is issued.
  Forecasts target ``origin + h`` for h = 1 … ``horizon_max``; history up to ``origin`` may be
  read as input, never fitted on.
* ``embargo`` is the number of time indices left unused between the fitting window and the
  first target: the first target is ``train_end + embargo``, so ``origin = train_end − 1 +
  embargo``. Embargo 0 is the classic rolling origin (forecasts start right after training).
  Split-level temporal regimes require an embargo of at least the maximum horizon plus lag
  (ROLLER step 5); here the caller chooses.

Folds advance by ``stride`` and stop when ``origin + horizon_max`` would leave the series.
"""

from __future__ import annotations

from collections.abc import Iterator


def rolling_origins(
    n_time: int, initial: int, horizon_max: int, stride: int = 1, embargo: int = 0
) -> Iterator[tuple[int, int]]:
    """Yield ``(train_end, origin)`` pairs, ``train_end = initial, initial + stride, …``.

    Every yielded pair satisfies ``origin = train_end − 1 + embargo`` and
    ``origin + horizon_max ≤ n_time − 1``, so all targets exist and none lies inside the
    fitting window.
    """
    if n_time < 1 or initial < 1 or horizon_max < 1 or stride < 1 or embargo < 0:
        raise ValueError(
            "need n_time >= 1, initial >= 1, horizon_max >= 1, stride >= 1, embargo >= 0"
        )
    train_end = initial
    while True:
        origin = train_end - 1 + embargo
        if origin + horizon_max > n_time - 1:
            return
        yield train_end, origin
        train_end += stride
