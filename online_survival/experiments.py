"""Synthetic experiments for the implemented online survival calibrators.

This module intentionally requires only the Python standard library. If tqdm is
available, repeated experiment helpers show progress bars in notebooks/scripts.
"""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import html
import math
import os
import random
from statistics import NormalDist

try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:  # pragma: no cover - exercised only without optional tqdm
    _tqdm = None

from .algorithms import (
    ACIWithIPCW,
    ACIWithoutIPCW,
    AdaFTRL,
    AdaFTRLV2,
    XuLPB,
    ipcw_error,
    unweighted_observed_error,
)


NORMAL = NormalDist()


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for a simple right-censored log-normal survival problem."""

    n_rounds: int = 600
    alpha: float = 0.10
    tau_max: float = 0.50
    eta: float = 0.015
    tau_min: float = 1e-6
    x_low: float = -1.0
    x_high: float = 1.0
    true_intercept: float = 1.0
    true_slope: float = 0.35
    true_sigma: float = 0.55
    model_log_shift: float = 0.30
    model_sigma: float = 0.55
    censoring_rate: float = 0.12


class LogNormalSurvivalExperiment:
    """Known synthetic data-generating process used by the notebook."""

    def __init__(self, config: SimulationConfig):
        self.config = config
        if not 0.0 < config.alpha <= config.tau_max < 1.0:
            raise ValueError("Require 0 < alpha <= tau_max < 1 for this experiment.")
        if config.censoring_rate <= 0.0:
            raise ValueError("censoring_rate must be positive.")

    @property
    def g_min(self) -> float:
        q_max = max(
            self.quantile(self.config.x_low, self.config.tau_max),
            self.quantile(self.config.x_high, self.config.tau_max),
        )
        return math.exp(-self.config.censoring_rate * q_max)

    def true_mu(self, x: float) -> float:
        return self.config.true_intercept + self.config.true_slope * x

    def model_mu(self, x: float) -> float:
        return self.true_mu(x) + self.config.model_log_shift

    def quantile(self, x: float, tau: float) -> float:
        tau = min(max(float(tau), self.config.tau_min), 1.0 - 1e-12)
        z_tau = NORMAL.inv_cdf(tau)
        return math.exp(self.model_mu(x) + self.config.model_sigma * z_tau)

    def true_cdf(self, x: float, value: float) -> float:
        if value <= 0.0:
            return 0.0
        z_value = (math.log(value) - self.true_mu(x)) / self.config.true_sigma
        return NORMAL.cdf(z_value)

    def censoring_survival(self, value: float, x: float | None = None) -> float:
        del x
        return math.exp(-self.config.censoring_rate * max(value, 0.0))

    def sample_round(self, rng: random.Random) -> dict[str, float | int]:
        x = rng.uniform(self.config.x_low, self.config.x_high)
        event_time = math.exp(self.true_mu(x) + self.config.true_sigma * rng.gauss(0, 1))
        censoring_time = rng.expovariate(self.config.censoring_rate)
        y = min(event_time, censoring_time)
        delta = int(event_time < censoring_time)
        return {
            "x": x,
            "event_time": event_time,
            "censoring_time": censoring_time,
            "y": y,
            "delta": delta,
        }


def make_algorithms(
    config: SimulationConfig,
    g_min: float,
    include_adaftrl_v2: bool = False,
    include_xu_lpb: bool = False,
):
    """Create fresh algorithm instances for one run."""

    algorithms = [
        ACIWithIPCW(
            alpha=config.alpha,
            eta=config.eta,
            tau_max=config.tau_max,
            tau_min=config.tau_min,
        ),
        ACIWithoutIPCW(
            alpha=config.alpha,
            eta=config.eta,
            tau_max=config.tau_max,
            tau_min=config.tau_min,
        ),
        AdaFTRL(
            alpha=config.alpha,
            tau_max=config.tau_max,
            g_min=g_min,
            tau_min=config.tau_min,
        ),
    ]
    if include_adaftrl_v2:
        algorithms.append(
            AdaFTRLV2(
                alpha=config.alpha,
                tau_max=config.tau_max,
                g_min=g_min,
                tau_min=config.tau_min,
            )
        )
    if include_xu_lpb:
        algorithms.append(XuLPB(alpha=config.alpha, min_lower_bound=0.0))
    return algorithms


def run_one_experiment(
    config: SimulationConfig | None = None,
    seed: int = 0,
    include_adaftrl_v2: bool = False,
    include_xu_lpb: bool = False,
) -> list[dict[str, float | int | str]]:
    """Run all implemented methods on the same censored data stream."""

    config = config or SimulationConfig()
    rng = random.Random(seed)
    experiment = LogNormalSurvivalExperiment(config)
    algorithms = make_algorithms(
        config,
        experiment.g_min,
        include_adaftrl_v2=include_adaftrl_v2,
        include_xu_lpb=include_xu_lpb,
    )
    records: list[dict[str, float | int | str]] = []

    for t in range(1, config.n_rounds + 1):
        round_data = experiment.sample_round(rng)
        x = float(round_data["x"])
        y = float(round_data["y"])
        delta = int(round_data["delta"])
        event_time = float(round_data["event_time"])

        for algorithm in algorithms:
            prediction = algorithm.predict(x, experiment.quantile)
            lower_bound = prediction.lower_bound
            update = algorithm.update(
                y=y,
                delta=delta,
                lower_bound=lower_bound,
                censoring_survival=lambda u, x=x: experiment.censoring_survival(u, x),
            )

            conditional_miscoverage = experiment.true_cdf(x, lower_bound)
            latent_error = float(event_time < lower_bound)

            records.append(
                {
                    "t": t,
                    "algorithm": algorithm.name,
                    "x": x,
                    "event_time": event_time,
                    "censoring_time": float(round_data["censoring_time"]),
                    "y": y,
                    "delta": delta,
                    "tau": prediction.tau,
                    "lower_bound": lower_bound,
                    "surrogate_error": update.error,
                    "latent_error": latent_error,
                    "covered": 1.0 - latent_error,
                    "conditional_miscoverage": conditional_miscoverage,
                    "conditional_coverage": 1.0 - conditional_miscoverage,
                    "next_tau": update.info.get(
                        "next_tau", getattr(algorithm, "tau", math.nan)
                    ),
                    "action": prediction.info.get("action", math.nan),
                    "ambiguity_indicator": update.info.get(
                        "ambiguity_indicator",
                        float((not delta) and y < lower_bound),
                    ),
                    "neutral_feedback": update.info.get("neutral_feedback", math.nan),
                    "cox_beta_norm": update.info.get(
                        "beta_norm", prediction.info.get("beta_norm", math.nan)
                    ),
                    "cox_event_grid_size": update.info.get(
                        "event_grid_size",
                        prediction.info.get("event_grid_size", math.nan),
                    ),
                    "g_min": experiment.g_min,
                    "alpha": config.alpha,
                    "target_coverage": 1.0 - config.alpha,
                }
            )

    return records


def run_censor_only_stress_experiment(
    config: SimulationConfig | None = None,
    seed: int = 0,
    censor_only_start: int = 300,
    forced_censor_fraction: float = 0.25,
    include_adaftrl_v2: bool = True,
    include_xu_lpb: bool = False,
) -> list[dict[str, float | int | str]]:
    """Run a stream with normal warm-up followed by only censored observations.

    This is an adversarial stress display, not a model-validity experiment:
    after ``censor_only_start`` the learner receives ``Delta=0`` for every
    round. Latent event times are still sampled so realized coverage can be
    evaluated.
    """

    config = config or SimulationConfig()
    if censor_only_start < 1 or censor_only_start > config.n_rounds:
        raise ValueError("censor_only_start must be in [1, n_rounds].")
    if not 0.0 < forced_censor_fraction < 1.0:
        raise ValueError("forced_censor_fraction must be in (0, 1).")

    rng = random.Random(seed)
    experiment = LogNormalSurvivalExperiment(config)
    algorithms = make_algorithms(
        config,
        experiment.g_min,
        include_adaftrl_v2=include_adaftrl_v2,
        include_xu_lpb=include_xu_lpb,
    )
    records: list[dict[str, float | int | str]] = []

    for t in range(1, config.n_rounds + 1):
        round_data = experiment.sample_round(rng)
        phase = "warmup"
        if t >= censor_only_start:
            phase = "censor_only"
            event_time = float(round_data["event_time"])
            x = float(round_data["x"])
            typical_time = math.exp(experiment.true_mu(x))
            censoring_time = min(
                forced_censor_fraction * typical_time,
                forced_censor_fraction * event_time,
            )
            round_data = {
                **round_data,
                "censoring_time": censoring_time,
                "y": censoring_time,
                "delta": 0,
            }

        x = float(round_data["x"])
        y = float(round_data["y"])
        delta = int(round_data["delta"])
        event_time = float(round_data["event_time"])

        for algorithm in algorithms:
            prediction = algorithm.predict(x, experiment.quantile)
            lower_bound = prediction.lower_bound
            update = algorithm.update(
                y=y,
                delta=delta,
                lower_bound=lower_bound,
                censoring_survival=lambda u, x=x: experiment.censoring_survival(u, x),
            )

            conditional_miscoverage = experiment.true_cdf(x, lower_bound)
            latent_error = float(event_time < lower_bound)

            records.append(
                {
                    "t": t,
                    "phase": phase,
                    "algorithm": algorithm.name,
                    "x": x,
                    "event_time": event_time,
                    "censoring_time": float(round_data["censoring_time"]),
                    "y": y,
                    "delta": delta,
                    "tau": prediction.tau,
                    "lower_bound": lower_bound,
                    "surrogate_error": update.error,
                    "latent_error": latent_error,
                    "covered": 1.0 - latent_error,
                    "conditional_miscoverage": conditional_miscoverage,
                    "conditional_coverage": 1.0 - conditional_miscoverage,
                    "next_tau": update.info.get(
                        "next_tau", getattr(algorithm, "tau", math.nan)
                    ),
                    "action": prediction.info.get("action", math.nan),
                    "ambiguity_indicator": update.info.get(
                        "ambiguity_indicator",
                        float((not delta) and y < lower_bound),
                    ),
                    "neutral_feedback": update.info.get("neutral_feedback", math.nan),
                    "cox_beta_norm": update.info.get(
                        "beta_norm", prediction.info.get("beta_norm", math.nan)
                    ),
                    "cox_event_grid_size": update.info.get(
                        "event_grid_size",
                        prediction.info.get("event_grid_size", math.nan),
                    ),
                    "g_min": experiment.g_min,
                    "alpha": config.alpha,
                    "target_coverage": 1.0 - config.alpha,
                }
            )

    return records


def summarize_one_run(
    records: list[dict[str, float | int | str]],
) -> list[dict[str, float | str]]:
    """Summarize one experiment by algorithm."""

    rows = []
    target_coverage = float(records[0]["target_coverage"]) if records else math.nan
    for algorithm, group in _group_records(records).items():
        realized_coverage = _mean(row["covered"] for row in group)
        average_conditional_coverage = _mean(
            row["conditional_coverage"] for row in group
        )
        rows.append(
            {
                "algorithm": algorithm,
                "realized_coverage": realized_coverage,
                "average_conditional_coverage": average_conditional_coverage,
                "target_coverage": target_coverage,
                "coverage_error": realized_coverage - target_coverage,
                "coverage_abs_error": abs(realized_coverage - target_coverage),
                "coverage_shortfall": max(0.0, target_coverage - realized_coverage),
                "realized_miscoverage": _mean(row["latent_error"] for row in group),
                "average_conditional_miscoverage": _mean(
                    row["conditional_miscoverage"] for row in group
                ),
                "average_surrogate_error": _mean(
                    row["surrogate_error"] for row in group
                ),
                "average_lower_bound": _mean(row["lower_bound"] for row in group),
                "ambiguous_below_bound_fraction": _mean(
                    row["ambiguity_indicator"] for row in group
                ),
                "final_tau": float(group[-1]["tau"]),
                "censoring_fraction": 1.0 - _mean(row["delta"] for row in group),
            }
        )
    return rows


def run_repeated_experiments(
    config: SimulationConfig | None = None,
    n_repeats: int = 100,
    seed: int = 2026,
    include_adaftrl_v2: bool = False,
    include_xu_lpb: bool = False,
    *,
    use_multiprocessing: bool = True,
    n_jobs: int | None = None,
    show_progress: bool = True,
) -> list[dict[str, float | int | str]]:
    """Run repeated experiments under the same configuration.

    Repeats run in separate processes by default. Set ``use_multiprocessing`` to
    ``False`` or ``n_jobs=1`` to recover the previous serial behavior. Set
    ``show_progress`` to ``False`` to suppress the optional tqdm progress bar.
    """

    config = config or SimulationConfig()
    summaries: list[dict[str, float | int | str]] = []
    tasks = [
        (repeat, config, seed + repeat, include_adaftrl_v2, include_xu_lpb)
        for repeat in range(n_repeats)
    ]

    if not tasks:
        return summaries

    worker_count = _resolve_worker_count(n_repeats, n_jobs)
    if use_multiprocessing and worker_count > 1:
        chunksize = max(1, len(tasks) // (worker_count * 4))
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            repeat_summaries = executor.map(
                _run_one_repeat_summary,
                tasks,
                chunksize=chunksize,
            )
            for repeat_summary in _progress(
                repeat_summaries,
                total=len(tasks),
                desc="Repeated experiments",
                show_progress=show_progress,
            ):
                summaries.extend(repeat_summary)
        return summaries

    for task in _progress(
        tasks,
        total=len(tasks),
        desc="Repeated experiments",
        show_progress=show_progress,
    ):
        summaries.extend(_run_one_repeat_summary(task))

    return summaries


def _run_one_repeat_summary(
    task: tuple[int, SimulationConfig, int, bool, bool],
) -> list[dict[str, float | int | str]]:
    repeat, config, run_seed, include_adaftrl_v2, include_xu_lpb = task
    run = run_one_experiment(
        config=config,
        seed=run_seed,
        include_adaftrl_v2=include_adaftrl_v2,
        include_xu_lpb=include_xu_lpb,
    )
    return [{"repeat": repeat, **row} for row in summarize_one_run(run)]


def _resolve_worker_count(n_repeats: int, n_jobs: int | None) -> int:
    if n_jobs is not None and n_jobs < 1:
        raise ValueError("n_jobs must be a positive integer or None.")
    cpu_count = os.cpu_count() or 1
    requested = cpu_count if n_jobs is None else n_jobs
    return max(1, min(n_repeats, requested))


def _progress(iterable, total: int, desc: str, show_progress: bool):
    if not show_progress or _tqdm is None:
        return iterable
    return _tqdm(iterable, total=total, desc=desc)


def run_repeated_censor_only_stress_experiments(
    config: SimulationConfig | None = None,
    n_repeats: int = 40,
    seed: int = 2026,
    censor_only_start: int = 300,
    forced_censor_fraction: float = 0.25,
    include_adaftrl_v2: bool = True,
    include_xu_lpb: bool = False,
    *,
    show_progress: bool = True,
) -> list[dict[str, float | int | str]]:
    """Repeat the long censor-only stress experiment across seeds."""

    config = config or SimulationConfig()
    summaries: list[dict[str, float | int | str]] = []

    for repeat in _progress(
        range(n_repeats),
        total=n_repeats,
        desc="Censor-only stress experiments",
        show_progress=show_progress,
    ):
        run = run_censor_only_stress_experiment(
            config=config,
            seed=seed + repeat,
            censor_only_start=censor_only_start,
            forced_censor_fraction=forced_censor_fraction,
            include_adaftrl_v2=include_adaftrl_v2,
            include_xu_lpb=include_xu_lpb,
        )
        for row in summarize_one_run(run):
            summaries.append({"repeat": repeat, **row})

    return summaries


def fixed_rule_bias_diagnostic(
    config: SimulationConfig | None = None,
    seed: int = 0,
    tau: float | None = None,
) -> list[dict[str, float | str]]:
    """Compare latent, unweighted, and IPCW errors for a fixed quantile rule."""

    config = config or SimulationConfig()
    tau = config.alpha if tau is None else tau
    rng = random.Random(seed)
    experiment = LogNormalSurvivalExperiment(config)

    latent_errors = []
    unweighted_errors = []
    ipcw_errors = []
    deltas = []

    for _ in range(config.n_rounds):
        round_data = experiment.sample_round(rng)
        x = float(round_data["x"])
        y = float(round_data["y"])
        delta = int(round_data["delta"])
        lower_bound = experiment.quantile(x, tau)

        latent_errors.append(float(float(round_data["event_time"]) < lower_bound))
        unweighted_errors.append(unweighted_observed_error(y, delta, lower_bound))
        ipcw_errors.append(
            ipcw_error(
                y=y,
                delta=delta,
                lower_bound=lower_bound,
                censoring_survival=lambda u, x=x: experiment.censoring_survival(u, x),
            )
        )
        deltas.append(delta)

    return [
        {
            "quantity": "latent error 1{T < L}",
            "mean": _mean(latent_errors),
            "target": config.alpha,
        },
        {
            "quantity": "unweighted observed error Delta * 1{Y < L}",
            "mean": _mean(unweighted_errors),
            "target": config.alpha,
        },
        {
            "quantity": "IPCW error Delta/G(Y) * 1{Y < L}",
            "mean": _mean(ipcw_errors),
            "target": config.alpha,
        },
        {
            "quantity": "censoring fraction",
            "mean": 1.0 - _mean(deltas),
            "target": math.nan,
        },
    ]


def summarize_repeats(
    repeated: list[dict[str, float | int | str]],
) -> list[dict[str, float | str]]:
    """Compact mean and standard-error table for repeated experiments."""

    metrics = [
        "realized_coverage",
        "average_conditional_coverage",
        "realized_miscoverage",
        "average_conditional_miscoverage",
        "average_surrogate_error",
        "average_lower_bound",
        "ambiguous_below_bound_fraction",
        "coverage_error",
        "coverage_abs_error",
        "coverage_shortfall",
        "final_tau",
        "censoring_fraction",
    ]
    rows = []
    for algorithm, group in _group_records(repeated).items():
        row: dict[str, float | str] = {"algorithm": algorithm}
        for metric in metrics:
            values = [float(item[metric]) for item in group]
            row[f"{metric}_mean"] = _mean(values)
            row[f"{metric}_se"] = _standard_error(values)
        rows.append(row)
    return rows


def format_markdown_table(
    rows: list[dict[str, float | int | str]],
    columns: list[str] | None = None,
    digits: int = 3,
) -> str:
    """Format a list of dictionaries as a small markdown table."""

    if not rows:
        return ""
    columns = columns or list(rows[0].keys())
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        formatted = [_format_value(row.get(column, ""), digits) for column in columns]
        lines.append("| " + " | ".join(formatted) + " |")
    return "\n".join(lines)


def behavior_svg(
    records: list[dict[str, float | int | str]],
    window: int = 50,
    width: int = 980,
    height: int = 560,
    phase_change_round: int | None = None,
    title: str | None = None,
) -> str:
    """Return an SVG with rolling coverage and tau paths for one run."""

    groups = _group_records(records)
    if not groups:
        return "<svg></svg>"

    target = float(records[0]["target_coverage"])
    alpha = float(records[0]["alpha"])
    tau_axis_max = max(float(row["tau"]) for row in records)
    n_rounds = max(int(row["t"]) for row in records)

    margin_left = 78
    margin_right = 34
    panel_height = 185
    panel_gap = 70
    top_1 = 45
    top_2 = top_1 + panel_height + panel_gap
    legend_top = top_2 + panel_height + 55
    height = max(height, legend_top + 18 * len(groups) + 18)
    plot_width = width - margin_left - margin_right
    colors = ["#2364aa", "#d95f02", "#2a9d8f", "#6a4c93", "#8c564b"]

    def x_scale(t: int) -> float:
        if n_rounds <= 1:
            return margin_left
        return margin_left + (t - 1) * plot_width / (n_rounds - 1)

    def y_scale(value: float, top: int, y_min: float, y_max: float) -> float:
        value = min(max(value, y_min), y_max)
        return top + panel_height - (value - y_min) * panel_height / (y_max - y_min)

    tau_y_max = max(tau_axis_max, alpha, 0.05)
    x_ticks = _round_ticks(n_rounds)
    coverage_ticks = [0.0, 0.25, 0.50, 0.75, 1.0]
    tau_ticks = [0.0, tau_y_max / 2.0, tau_y_max]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        _svg_text(
            20,
            26,
            title or "One-run behavior: rolling coverage and played quantile",
            17,
        ),
    ]

    parts += _panel_axes(
        margin_left,
        top_1,
        plot_width,
        panel_height,
        "coverage",
        "round",
        x_ticks,
        coverage_ticks,
        x_scale,
        lambda value: y_scale(value, top_1, 0.0, 1.0),
    )
    parts += _panel_axes(
        margin_left,
        top_2,
        plot_width,
        panel_height,
        "tau",
        "round",
        x_ticks,
        tau_ticks,
        x_scale,
        lambda value: y_scale(value, top_2, 0.0, tau_y_max),
    )

    if phase_change_round is not None:
        x_phase = x_scale(phase_change_round)
        for top in [top_1, top_2]:
            parts.append(
                f'<line x1="{x_phase:.2f}" y1="{top}" '
                f'x2="{x_phase:.2f}" y2="{top + panel_height}" '
                'stroke="#111" stroke-dasharray="5 5" stroke-width="1.5"/>'
            )
        parts.append(_svg_text(x_phase + 6, top_1 + 16, "censor-only starts", 12))

    y_target = y_scale(target, top_1, 0.0, 1.0)
    parts.append(
        f'<line x1="{margin_left}" y1="{y_target:.2f}" '
        f'x2="{margin_left + plot_width}" y2="{y_target:.2f}" '
        'stroke="#666" stroke-dasharray="4 4" stroke-width="1.3"/>'
    )
    parts.append(_svg_text(width - 145, y_target - 7, f"target={target:.2f}", 12))

    y_alpha = y_scale(alpha, top_2, 0.0, tau_y_max)
    parts.append(
        f'<line x1="{margin_left}" y1="{y_alpha:.2f}" '
        f'x2="{margin_left + plot_width}" y2="{y_alpha:.2f}" '
        'stroke="#666" stroke-dasharray="4 4" stroke-width="1.3"/>'
    )
    parts.append(_svg_text(width - 125, y_alpha - 7, f"alpha={alpha:.2f}", 12))

    for index, (algorithm, group) in enumerate(groups.items()):
        color = colors[index % len(colors)]
        group = sorted(group, key=lambda row: int(row["t"]))
        coverage_points = []
        tau_points = []
        covered_window: list[float] = []
        for row in group:
            covered_window.append(float(row["covered"]))
            if len(covered_window) > window:
                covered_window.pop(0)
            rolling_coverage = _mean(covered_window)
            x = x_scale(int(row["t"]))
            coverage_points.append(
                f"{x:.2f},{y_scale(rolling_coverage, top_1, 0.0, 1.0):.2f}"
            )
            tau_points.append(
                f"{x:.2f},{y_scale(float(row['tau']), top_2, 0.0, tau_y_max):.2f}"
            )
        parts.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2.1" '
            f'points="{" ".join(coverage_points)}"/>'
        )
        parts.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2.1" '
            f'points="{" ".join(tau_points)}"/>'
        )
        legend_y = legend_top + 18 * index
        parts.append(
            f'<line x1="{margin_left}" y1="{legend_y}" x2="{margin_left + 26}" '
            f'y2="{legend_y}" stroke="{color}" stroke-width="3"/>'
        )
        parts.append(_svg_text(margin_left + 34, legend_y + 4, algorithm, 12))

    parts.append("</svg>")
    return "\n".join(parts)


def _group_records(records):
    groups = defaultdict(list)
    for row in records:
        groups[str(row["algorithm"])].append(row)
    return dict(groups)


def _mean(values) -> float:
    values = list(values)
    if not values:
        return math.nan
    return sum(float(value) for value in values) / len(values)


def _standard_error(values: list[float]) -> float:
    if len(values) <= 1:
        return math.nan
    mean_value = _mean(values)
    variance = sum((value - mean_value) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance / len(values))


def _format_value(value, digits: int) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.{digits}f}"
    return str(value)


def _svg_text(x: float, y: float, text: str, size: int) -> str:
    safe = html.escape(text)
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-family="DejaVu Sans, Arial, sans-serif" '
        f'font-size="{size}" fill="#222">{safe}</text>'
    )


def _round_ticks(n_rounds: int) -> list[int]:
    if n_rounds <= 1:
        return [1]
    ticks = [1]
    for fraction in [0.25, 0.50, 0.75]:
        ticks.append(max(1, round(1 + (n_rounds - 1) * fraction)))
    ticks.append(n_rounds)
    return sorted(set(ticks))


def _format_tick(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.2f}"


def _panel_axes(
    x: int,
    y: int,
    width: int,
    height: int,
    y_label: str,
    x_label: str,
    x_ticks: list[int],
    y_ticks: list[float],
    x_scale,
    y_scale,
) -> list[str]:
    parts = [
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" '
        'fill="#fafafa" stroke="#ddd"/>',
        f'<line x1="{x}" y1="{y + height}" x2="{x + width}" y2="{y + height}" '
        'stroke="#222" stroke-width="1"/>',
        f'<line x1="{x}" y1="{y}" x2="{x}" y2="{y + height}" '
        'stroke="#222" stroke-width="1"/>',
    ]

    for tick in x_ticks:
        tx = x_scale(tick)
        parts.append(
            f'<line x1="{tx:.2f}" y1="{y + height}" x2="{tx:.2f}" '
            f'y2="{y + height + 5}" stroke="#222" stroke-width="1"/>'
        )
        parts.append(_svg_text(tx - 12, y + height + 22, _format_tick(tick), 11))

    for tick in y_ticks:
        ty = y_scale(tick)
        parts.append(
            f'<line x1="{x - 5}" y1="{ty:.2f}" x2="{x}" y2="{ty:.2f}" '
            'stroke="#222" stroke-width="1"/>'
        )
        parts.append(_svg_text(x - 42, ty + 4, _format_tick(tick), 11))
        parts.append(
            f'<line x1="{x}" y1="{ty:.2f}" x2="{x + width}" y2="{ty:.2f}" '
            'stroke="#e4e4e4" stroke-width="0.8"/>'
        )

    parts.append(_svg_text(18, y + 18, y_label, 12))
    parts.append(_svg_text(x + width - 44, y + height + 40, x_label, 12))
    return parts
