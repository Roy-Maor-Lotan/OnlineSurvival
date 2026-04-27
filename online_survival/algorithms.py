"""Online calibration algorithms for lower survival bounds.

The classes here implement the experimental lower-bound baselines:

* ACI with IPCW
* ACI without IPCW
* AdaFTRL
* AdaFTRL-V2, the ambiguity-neutral AdaFTRL variant
* XuLPB, a pure model-based online Cox lower prediction bound

Each method predicts a quantile level ``tau_t`` and updates from the observed
censored outcome ``(Y_t, Delta_t)``.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import Iterable
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


def _as_feature_tuple(x) -> tuple[float, ...]:
    if isinstance(x, (int, float)):
        return (float(x),)
    if isinstance(x, str):
        raise TypeError("x must be numeric or an iterable of numeric features.")
    if isinstance(x, Iterable):
        features = tuple(float(value) for value in x)
        if not features:
            raise ValueError("x must contain at least one feature.")
        return features
    raise TypeError("x must be numeric or an iterable of numeric features.")


def _dot(left: tuple[float, ...], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _exp_clipped(value: float, limit: float = 50.0) -> float:
    return math.exp(_clip(value, -limit, limit))


def _l2_norm(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def _scale_to_max_norm(values: list[float], max_norm: float) -> list[float]:
    norm = _l2_norm(values)
    if norm <= max_norm or norm == 0.0:
        return values
    scale = max_norm / norm
    return [scale * value for value in values]


def _solve_linear_system(
    matrix: list[list[float]], rhs: list[float], pivot_tol: float = 1e-12
) -> list[float] | None:
    n = len(rhs)
    augmented = [row[:] + [rhs_i] for row, rhs_i in zip(matrix, rhs)]

    for col in range(n):
        pivot_row = max(range(col, n), key=lambda row: abs(augmented[row][col]))
        if abs(augmented[pivot_row][col]) < pivot_tol:
            return None
        if pivot_row != col:
            augmented[col], augmented[pivot_row] = augmented[pivot_row], augmented[col]

        pivot = augmented[col][col]
        for item in range(col, n + 1):
            augmented[col][item] /= pivot

        for row in range(n):
            if row == col:
                continue
            factor = augmented[row][col]
            if factor == 0.0:
                continue
            for item in range(col, n + 1):
                augmented[row][item] -= factor * augmented[col][item]

    return [augmented[row][n] for row in range(n)]


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


@dataclass
class XuLPB:
    """Pure model-based online Cox lower prediction bound.

    The baseline keeps ``tau`` fixed at ``alpha`` and does not run ACI,
    AdaFTRL, IPCW calibration, or any adaptive quantile update. Each call to
    ``update`` appends the newly observed censored survival record. Observed
    events trigger an online Newton-style Cox score step and a Breslow baseline
    cumulative-hazard increment on the observed event-time grid.
    """

    alpha: float
    min_lower_bound: float = 0.0
    beta_step_size: float = 0.35
    ridge: float = 1e-4
    max_step_norm: float = 1.0
    max_beta_norm: float = 8.0
    name: str = "XuLPB"
    beta_hat: list[float] = field(default_factory=list, init=False)
    event_time_grid: list[float] = field(default_factory=list, init=False)
    baseline_hazard_increments: list[float] = field(default_factory=list, init=False)
    baseline_cumulative_hazard: list[float] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be in (0, 1).")
        if self.min_lower_bound < 0.0:
            raise ValueError("min_lower_bound must be nonnegative.")
        if self.beta_step_size <= 0.0:
            raise ValueError("beta_step_size must be positive.")
        if self.ridge < 0.0:
            raise ValueError("ridge must be nonnegative.")
        if self.max_step_norm <= 0.0:
            raise ValueError("max_step_norm must be positive.")
        if self.max_beta_norm <= 0.0:
            raise ValueError("max_beta_norm must be positive.")
        self._target_cumulative_hazard = -math.log1p(-self.alpha)
        self._observations: list[tuple[tuple[float, ...], float, int]] = []
        self._pending_x: tuple[float, ...] | None = None
        self._num_events = 0

    def predict(self, x, quantile_fn: QuantileFunction | None = None) -> Prediction:
        del quantile_fn
        features = self._coerce_features(x)
        self._pending_x = features

        lower_bound = self.min_lower_bound
        linear_predictor = _dot(features, self.beta_hat)
        threshold = self._target_cumulative_hazard * _exp_clipped(-linear_predictor)
        if self.baseline_cumulative_hazard:
            grid_index = bisect_right(self.baseline_cumulative_hazard, threshold) - 1
            if grid_index >= 0:
                lower_bound = self.event_time_grid[grid_index]

        return Prediction(
            tau=self.alpha,
            lower_bound=float(lower_bound),
            info={
                "state_tau": self.alpha,
                "linear_predictor": linear_predictor,
                "baseline_threshold": threshold,
                "beta_norm": _l2_norm(self.beta_hat),
                "event_grid_size": float(len(self.event_time_grid)),
            },
        )

    def update(
        self,
        y: float,
        delta: int,
        lower_bound: float,
        censoring_survival: CensoringSurvival | None = None,
    ) -> UpdateResult:
        del censoring_survival
        if self._pending_x is None:
            raise RuntimeError("XuLPB.predict must be called before XuLPB.update.")

        features = self._pending_x
        self._pending_x = None
        observed_time = float(y)
        if observed_time < 0.0:
            raise ValueError("observed survival time must be nonnegative.")
        event_indicator = int(delta)
        self._observations.append((features, observed_time, event_indicator))

        step_norm = 0.0
        baseline_increment = 0.0
        if event_indicator:
            self._num_events += 1
            step_norm = self._update_beta_from_event(features, observed_time)
            baseline_increment = self._breslow_increment(observed_time)
            if baseline_increment > 0.0:
                self._add_baseline_increment(observed_time, baseline_increment)

        observed_error = unweighted_observed_error(y, delta, lower_bound)
        return UpdateResult(
            error=observed_error,
            info={
                "next_tau": self.alpha,
                "n_observations": float(len(self._observations)),
                "n_events": float(self._num_events),
                "beta_norm": _l2_norm(self.beta_hat),
                "event_grid_size": float(len(self.event_time_grid)),
                "last_beta_step_norm": step_norm,
                "last_baseline_increment": baseline_increment,
            },
        )

    def _coerce_features(self, x) -> tuple[float, ...]:
        features = _as_feature_tuple(x)
        if not self.beta_hat:
            self.beta_hat = [0.0] * len(features)
        if len(features) != len(self.beta_hat):
            raise ValueError(
                "all XuLPB feature vectors must have the same dimension."
            )
        return features

    def _risk_set_moments(
        self, event_time: float
    ) -> tuple[float, float, list[float], list[list[float]]] | None:
        risk_rows = []
        max_eta = -math.inf
        for features, observed_time, _ in self._observations:
            if observed_time >= event_time:
                eta = _dot(features, self.beta_hat)
                max_eta = max(max_eta, eta)
                risk_rows.append((features, eta))

        if not risk_rows:
            return None

        dimension = len(self.beta_hat)
        sum_weight = 0.0
        sum_weighted_x = [0.0] * dimension
        sum_weighted_xx = [[0.0] * dimension for _ in range(dimension)]

        for features, eta in risk_rows:
            weight = math.exp(_clip(eta - max_eta, -50.0, 0.0))
            sum_weight += weight
            for row in range(dimension):
                weighted_feature = weight * features[row]
                sum_weighted_x[row] += weighted_feature
                for col in range(dimension):
                    sum_weighted_xx[row][col] += weighted_feature * features[col]

        if sum_weight <= 0.0:
            return None

        mean_x = [value / sum_weight for value in sum_weighted_x]
        covariance = [[0.0] * dimension for _ in range(dimension)]
        for row in range(dimension):
            for col in range(dimension):
                covariance[row][col] = (
                    sum_weighted_xx[row][col] / sum_weight
                    - mean_x[row] * mean_x[col]
                )

        return max_eta, sum_weight, mean_x, covariance

    def _update_beta_from_event(
        self, event_features: tuple[float, ...], event_time: float
    ) -> float:
        moments = self._risk_set_moments(event_time)
        if moments is None:
            return 0.0
        _, _, mean_x, covariance = moments
        dimension = len(self.beta_hat)
        score = [
            event_features[index] - mean_x[index] - self.ridge * self.beta_hat[index]
            for index in range(dimension)
        ]
        information = [row[:] for row in covariance]
        for index in range(dimension):
            information[index][index] += self.ridge

        step = _solve_linear_system(information, score)
        if step is None:
            step = [
                score[index] / max(information[index][index], 1e-8)
                for index in range(dimension)
            ]

        step = _scale_to_max_norm(step, self.max_step_norm)
        event_scale = self.beta_step_size / math.sqrt(max(1, self._num_events))
        beta_candidate = [
            self.beta_hat[index] + event_scale * step[index]
            for index in range(dimension)
        ]
        self.beta_hat = _scale_to_max_norm(beta_candidate, self.max_beta_norm)
        return event_scale * _l2_norm(step)

    def _breslow_increment(self, event_time: float) -> float:
        moments = self._risk_set_moments(event_time)
        if moments is None:
            return 0.0
        max_eta, sum_weight, _, _ = moments
        return _exp_clipped(-max_eta) / sum_weight

    def _add_baseline_increment(self, event_time: float, increment: float) -> None:
        insert_at = bisect_left(self.event_time_grid, event_time)
        if (
            insert_at < len(self.event_time_grid)
            and self.event_time_grid[insert_at] == event_time
        ):
            self.baseline_hazard_increments[insert_at] += increment
        else:
            self.event_time_grid.insert(insert_at, event_time)
            self.baseline_hazard_increments.insert(insert_at, increment)
            self.baseline_cumulative_hazard.insert(insert_at, 0.0)
        self._recompute_baseline_cumulative_hazard(start=insert_at)

    def _recompute_baseline_cumulative_hazard(self, start: int = 0) -> None:
        running = (
            self.baseline_cumulative_hazard[start - 1] if start > 0 else 0.0
        )
        for index in range(start, len(self.baseline_hazard_increments)):
            running += self.baseline_hazard_increments[index]
            self.baseline_cumulative_hazard[index] = running


def adaftrl_v2(alpha: float, tau_max: float, g_min: float, **kwargs) -> AdaFTRLV2:
    """Factory for the renamed AdaFTRL-V2 manuscript variant."""

    return AdaFTRLV2(alpha=alpha, tau_max=tau_max, g_min=g_min, **kwargs)
