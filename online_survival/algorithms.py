"""Online calibration algorithms for lower survival bounds.

The classes here implement only the three methods requested:

* ACI with IPCW
* ACI without IPCW
* AdaFTRL
* AdaFTRL-V2, the ambiguity-neutral AdaFTRL variant

Each method predicts a quantile level ``tau_t`` and updates from the observed
censored outcome ``(Y_t, Delta_t)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Callable


QuantileFunction = Callable[[float, float], float]
CensoringSurvival = Callable[[float], float]


@dataclass(frozen=True)
class Prediction:
    """Prediction made before observing the round outcome."""

    tau: float
    lower_bound: float
    info: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class UpdateResult:
    """Feedback quantities computed after observing the round outcome."""

    error: float
    info: dict[str, float] = field(default_factory=dict)


def _clip(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def _check_common(alpha: float, tau_max: float) -> None:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1).")
    if not alpha <= tau_max <= 1.0:
        raise ValueError("tau_max must satisfy alpha <= tau_max <= 1.")


def ipcw_error(
    y: float,
    delta: int,
    lower_bound: float,
    censoring_survival: CensoringSurvival,
) -> float:
    """Observable IPCW surrogate: Delta / G(Y) * 1{Y < L}."""

    if delta and y < lower_bound:
        g_y = float(censoring_survival(y))
        if g_y <= 0.0:
            raise ValueError("censoring survival must be positive on active rounds.")
        return float(delta) / g_y
    return 0.0


def unweighted_observed_error(y: float, delta: int, lower_bound: float) -> float:
    """Naive observed-event error: Delta * 1{Y < L}.

    This deliberately omits the IPCW correction, so censored rounds contribute
    zero. It is useful as a simple biased baseline under right censoring.
    """

    return float(y < lower_bound)


@dataclass
class ACIWithIPCW:
    """Direct quantile-level ACI using the IPCW surrogate error.

    Update:
        tau_{t+1} = tau_t + eta * (alpha - e_hat_t)

    The manuscript notes that this raw recursion does not preserve
    ``(0, tau_max]`` pathwise. For practical experiments we clip the played
    state by default.
    """

    alpha: float
    eta: float
    tau_max: float = 1.0
    tau_min: float = 1e-6
    tau: float | None = None
    clip_tau: bool = True
    name: str = "ACI with IPCW"

    def __post_init__(self) -> None:
        _check_common(self.alpha, self.tau_max)
        if self.eta <= 0.0:
            raise ValueError("eta must be positive.")
        if not 0.0 < self.tau_min <= self.tau_max:
            raise ValueError("tau_min must be in (0, tau_max].")
        if self.tau is None:
            self.tau = self.alpha
        if self.clip_tau:
            self.tau = _clip(self.tau, self.tau_min, self.tau_max)

    def predict(self, x: float, quantile_fn: QuantileFunction) -> Prediction:
        tau = _clip(float(self.tau), self.tau_min, self.tau_max)
        lower_bound = float(quantile_fn(x, tau))
        return Prediction(
            tau=tau,
            lower_bound=lower_bound,
            info={"state_tau": float(self.tau)},
        )

    def update(
        self,
        y: float,
        delta: int,
        lower_bound: float,
        censoring_survival: CensoringSurvival,
    ) -> UpdateResult:
        error = ipcw_error(y, delta, lower_bound, censoring_survival)
        raw_next_tau = float(self.tau) + self.eta * (self.alpha - error)
        self.tau = (
            _clip(raw_next_tau, self.tau_min, self.tau_max)
            if self.clip_tau
            else raw_next_tau
        )
        return UpdateResult(
            error=error,
            info={"raw_next_tau": raw_next_tau, "next_tau": float(self.tau)},
        )


@dataclass
class ACIWithoutIPCW:
    """Direct quantile-level ACI with no inverse-censoring correction."""

    alpha: float
    eta: float
    tau_max: float = 1.0
    tau_min: float = 1e-6
    tau: float | None = None
    clip_tau: bool = True
    name: str = "ACI without IPCW"

    def __post_init__(self) -> None:
        _check_common(self.alpha, self.tau_max)
        if self.eta <= 0.0:
            raise ValueError("eta must be positive.")
        if not 0.0 < self.tau_min <= self.tau_max:
            raise ValueError("tau_min must be in (0, tau_max].")
        if self.tau is None:
            self.tau = self.alpha
        if self.clip_tau:
            self.tau = _clip(self.tau, self.tau_min, self.tau_max)

    def predict(self, x: float, quantile_fn: QuantileFunction) -> Prediction:
        tau = _clip(float(self.tau), self.tau_min, self.tau_max)
        lower_bound = float(quantile_fn(x, tau))
        return Prediction(
            tau=tau,
            lower_bound=lower_bound,
            info={"state_tau": float(self.tau)},
        )

    def update(
        self,
        y: float,
        delta: int,
        lower_bound: float,
        censoring_survival: CensoringSurvival | None = None,
    ) -> UpdateResult:
        del censoring_survival
        error = unweighted_observed_error(y, delta, lower_bound)
        raw_next_tau = float(self.tau) + self.eta * (self.alpha - error)
        self.tau = (
            _clip(raw_next_tau, self.tau_min, self.tau_max)
            if self.clip_tau
            else raw_next_tau
        )
        return UpdateResult(
            error=error,
            info={"raw_next_tau": raw_next_tau, "next_tau": float(self.tau)},
        )


@dataclass
class AdaFTRL:
    """Half-line shifted AdaFTRL from the manuscript.

    Prediction:
        a_t = [a_init - H_{t-1} / sqrt(1 + S_{t-1})]_+
        tau_t = tau_max * exp(-a_t)

    Update:
        e_hat_t = Delta / G(Y) * 1{Y < L}
        h_t = g_min * (alpha - e_hat_t)
        H_t = H_{t-1} + h_t
        S_t = S_{t-1} + h_t^2
    """

    alpha: float
    tau_max: float
    g_min: float
    tau_min: float = 1e-12
    cumulative_gradient: float = 0.0
    squared_gradient_sum: float = 0.0
    name: str = "AdaFTRL"

    def __post_init__(self) -> None:
        _check_common(self.alpha, self.tau_max)
        if self.g_min <= 0.0:
            raise ValueError("g_min must be positive.")
        if not 0.0 < self.tau_min <= self.tau_max:
            raise ValueError("tau_min must be in (0, tau_max].")
        self.a_init = math.log(self.tau_max / self.alpha)

    def predict(self, x: float, quantile_fn: QuantileFunction) -> Prediction:
        scale = math.sqrt(1.0 + self.squared_gradient_sum)
        action = max(0.0, self.a_init - self.cumulative_gradient / scale)
        raw_tau = self.tau_max * math.exp(-action)
        tau = _clip(raw_tau, self.tau_min, self.tau_max)
        lower_bound = float(quantile_fn(x, tau))
        return Prediction(
            tau=tau,
            lower_bound=lower_bound,
            info={
                "action": action,
                "raw_tau": raw_tau,
                "scale": scale,
                "cumulative_gradient": self.cumulative_gradient,
            },
        )

    def update(
        self,
        y: float,
        delta: int,
        lower_bound: float,
        censoring_survival: CensoringSurvival,
    ) -> UpdateResult:
        error = ipcw_error(y, delta, lower_bound, censoring_survival)
        normalized_excess = self.g_min * (error - self.alpha)
        internal_gradient = -normalized_excess
        self.cumulative_gradient += internal_gradient
        self.squared_gradient_sum += internal_gradient * internal_gradient
        return UpdateResult(
            error=error,
            info={
                "normalized_excess": normalized_excess,
                "internal_gradient": internal_gradient,
                "cumulative_gradient": self.cumulative_gradient,
                "squared_gradient_sum": self.squared_gradient_sum,
            },
        )


@dataclass
class AdaFTRLV2:
    """Ambiguity-neutral AdaFTRL-V2.

    This follows ``adaftrl_ambiguity_neutral_updated.tex``. The unbiased IPCW
    surrogate is still reported as ``error``. The internal update instead uses

        e_ad = e_hat + alpha * 1{Delta = 0 and Y < L},

    so a censored-below-bound ambiguous round has ``e_ad = alpha`` and therefore
    zero internal gradient.
    """

    alpha: float
    tau_max: float
    g_min: float
    tau_min: float = 1e-12
    cumulative_gradient: float = 0.0
    squared_gradient_sum: float = 0.0
    name: str = "AdaFTRL-V2"

    def __post_init__(self) -> None:
        _check_common(self.alpha, self.tau_max)
        if self.g_min <= 0.0:
            raise ValueError("g_min must be positive.")
        if not 0.0 < self.tau_min <= self.tau_max:
            raise ValueError("tau_min must be in (0, tau_max].")
        self.a_init = math.log(self.tau_max / self.alpha)

    def predict(self, x: float, quantile_fn: QuantileFunction) -> Prediction:
        scale = math.sqrt(1.0 + self.squared_gradient_sum)
        action = max(0.0, self.a_init - self.cumulative_gradient / scale)
        raw_tau = self.tau_max * math.exp(-action)
        tau = _clip(raw_tau, self.tau_min, self.tau_max)
        lower_bound = float(quantile_fn(x, tau))
        return Prediction(
            tau=tau,
            lower_bound=lower_bound,
            info={
                "action": action,
                "raw_tau": raw_tau,
                "scale": scale,
                "cumulative_gradient": self.cumulative_gradient,
            },
        )

    def update(
        self,
        y: float,
        delta: int,
        lower_bound: float,
        censoring_survival: CensoringSurvival,
    ) -> UpdateResult:
        error = ipcw_error(y, delta, lower_bound, censoring_survival)
        ambiguous = int((not delta) and y < lower_bound)
        neutral_feedback = error + self.alpha * ambiguous
        normalized_excess = self.g_min * (neutral_feedback - self.alpha)
        internal_gradient = -normalized_excess
        self.cumulative_gradient += internal_gradient
        self.squared_gradient_sum += internal_gradient * internal_gradient
        return UpdateResult(
            error=error,
            info={
                "ambiguity_indicator": float(ambiguous),
                "neutral_feedback": neutral_feedback,
                "normalized_excess": normalized_excess,
                "internal_gradient": internal_gradient,
                "cumulative_gradient": self.cumulative_gradient,
                "squared_gradient_sum": self.squared_gradient_sum,
            },
        )


def adaftrl_v2(alpha: float, tau_max: float, g_min: float, **kwargs) -> AdaFTRLV2:
    """Factory for the renamed AdaFTRL-V2 manuscript variant."""

    return AdaFTRLV2(alpha=alpha, tau_max=tau_max, g_min=g_min, **kwargs)
