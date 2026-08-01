"""Shared training, evaluation and bookkeeping used by every algorithm.

The four algorithms have to be compared on equal terms, so all of them go
through the same env factory, the same evaluation protocol and the same results
schema. The only thing that differs between runs is the hyperparameters under
test.

Evaluation protocol
-------------------
Every run is scored on a *fixed* block of seeds that never appears in training,
and on the same block for every algorithm. Reporting the training reward instead
would flatter whichever algorithm happened to explore least, and reporting on
fresh random seeds each time would make runs incomparable. The held-out block is
also what the generalisation test perturbs.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from environment.custom_env import EnvConfig, SubseaInspectionEnv

# The policies are small MLPs and the environments run in their own processes,
# so torch's intra-op thread pool only fights the env workers for cores. Pinning
# it to one thread measured ~1.9x faster end-to-end than the default of four.
torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = LOG_DIR / "results"

#: Seeds held out from training and used for every reported score.
EVAL_SEEDS: tuple[int, ...] = tuple(range(9_000, 9_030))
#: A second, disjoint block used only for the generalisation test.
GENERALIZATION_SEEDS: tuple[int, ...] = tuple(range(12_000, 12_030))


def make_env(
    seed: int | None = None,
    config: EnvConfig | None = None,
    monitor_path: str | Path | None = None,
    **kwargs,
) -> Callable[[], Monitor]:
    """Return a thunk building one monitored environment instance."""

    def _init() -> Monitor:
        env = SubseaInspectionEnv(config=config, **kwargs)
        env = Monitor(
            env,
            filename=str(monitor_path) if monitor_path else None,
            info_keywords=("success", "outcome"),
        )
        if seed is not None:
            env.reset(seed=seed)
        return env

    return _init


def make_vec_env(
    n_envs: int = 8,
    seed: int = 0,
    config: EnvConfig | None = None,
    subprocess: bool | None = None,
) -> VecEnv:
    """Build a vectorised env.

    Subprocesses only pay off once there are enough of them to cover the IPC
    cost, so a single env stays in-process.
    """
    if subprocess is None:
        subprocess = n_envs > 1
    thunks = [make_env(seed=seed + i, config=config) for i in range(n_envs)]
    if subprocess and n_envs > 1:
        return SubprocVecEnv(thunks, start_method="fork")
    return DummyVecEnv(thunks)


# --------------------------------------------------------------------- scoring


@dataclass
class EvalResult:
    """What one evaluation block tells us about a policy."""

    mean_return: float
    std_return: float
    success_rate: float
    mean_miss: float
    mean_flight_time: float
    mean_battery_left: float
    outcomes: dict[str, int] = field(default_factory=dict)

    def as_row(self, prefix: str = "") -> dict[str, Any]:
        return {
            f"{prefix}mean_return": round(self.mean_return, 2),
            f"{prefix}std_return": round(self.std_return, 2),
            f"{prefix}success_rate": round(self.success_rate, 4),
            f"{prefix}mean_miss": round(self.mean_miss, 3),
            f"{prefix}flight_time": round(self.mean_flight_time, 2),
            f"{prefix}battery_left": round(self.mean_battery_left, 3),
        }


def evaluate_policy(
    predict: Callable[[np.ndarray], int],
    seeds: Iterable[int] = EVAL_SEEDS,
    config: EnvConfig | None = None,
    env: SubseaInspectionEnv | None = None,
) -> EvalResult:
    """Score a policy over a fixed block of seeds.

    ``predict`` maps an observation to an action, so an SB3 model plugs in as
    ``lambda o: int(model.predict(o, deterministic=True)[0])``.
    """
    owned = env is None
    env = env or SubseaInspectionEnv(config=config)
    returns, misses, times, batteries, outcomes = [], [], [], [], []
    try:
        for seed in seeds:
            obs, _ = env.reset(seed=int(seed))
            total, done = 0.0, False
            info: dict[str, Any] = {}
            while not done:
                obs, reward, terminated, truncated, info = env.step(predict(obs))
                total += reward
                done = terminated or truncated
            returns.append(total)
            outcomes.append(info["outcome"])
            times.append(info["mission_time"])
            batteries.append(info["battery"])
            if not np.isnan(info["inspect_error"]):
                misses.append(info["inspect_error"])
    finally:
        if owned:
            env.close()

    n = max(1, len(returns))
    return EvalResult(
        mean_return=float(np.mean(returns)),
        std_return=float(np.std(returns)),
        success_rate=sum(o == "survey_complete" for o in outcomes) / n,
        mean_miss=float(np.mean(misses)) if misses else float("nan"),
        mean_flight_time=float(np.mean(times)),
        mean_battery_left=float(np.mean(batteries)),
        outcomes={o: outcomes.count(o) for o in sorted(set(outcomes))},
    )


class PeriodicEvalCallback(BaseCallback):
    """Score the policy on the held-out seeds at a fixed timestep interval.

    SB3's own ``EvalCallback`` reports only mean reward; the report needs the
    delivery rate and miss distance alongside it, and needs them sampled on the
    same schedule for all four algorithms so the convergence plots line up.
    """

    def __init__(
        self,
        eval_every: int,
        out_csv: Path,
        n_seeds: int = 10,
        config: EnvConfig | None = None,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.eval_every = eval_every
        self.out_csv = Path(out_csv)
        self.seeds = EVAL_SEEDS[:n_seeds]
        self.config = config
        self.history: list[dict[str, Any]] = []
        self._next_at = eval_every
        self._env: SubseaInspectionEnv | None = None
        self._reference_obs: np.ndarray | None = None

    def _on_training_start(self) -> None:
        self._env = SubseaInspectionEnv(config=self.config)
        self._reference_obs = self._collect_reference_obs()
        self._record(0)

    def _collect_reference_obs(self, n: int = 256) -> np.ndarray | None:
        """A fixed batch of states for tracking the value estimate over training.

        SB3's DQN logs its TD loss but never the Q-values themselves, and the
        report needs the objective curve. Holding the states fixed is what makes
        the curve comparable across checkpoints: a moving state distribution
        would confound "values grew" with "the agent went somewhere else".
        """
        if self._env is None:
            return None
        rng = np.random.default_rng(0)
        obs_list = []
        obs, _ = self._env.reset(seed=7_777)
        for _ in range(n):
            obs_list.append(obs.copy())
            obs, _, term, trunc, _ = self._env.step(int(rng.integers(0, self._env.action_space.n)))
            if term or trunc:
                obs, _ = self._env.reset()
        return np.asarray(obs_list, dtype=np.float32)

    def _mean_q(self) -> float | None:
        """Mean greedy Q-value on the reference states, for value-based methods."""
        q_net = getattr(self.model, "q_net", None)
        if q_net is None or self._reference_obs is None:
            return None
        with torch.no_grad():
            values = q_net(torch.as_tensor(self._reference_obs))
            return float(values.max(dim=1).values.mean().item())

    def _record(self, step: int) -> None:
        res = evaluate_policy(
            lambda o: int(self.model.predict(o, deterministic=True)[0]),
            seeds=self.seeds,
            env=self._env,
        )
        row = {"timesteps": step, **res.as_row()}
        mean_q = self._mean_q()
        if mean_q is not None:
            row["mean_q"] = round(mean_q, 3)
        self.history.append(row)
        if self.verbose:
            print(
                f"    [eval] {step:>8,} steps  return {res.mean_return:8.2f}"
                f"  survey {100 * res.success_rate:5.1f}%"
            )

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next_at:
            self._next_at += self.eval_every
            self._record(self.num_timesteps)
        return True

    def _on_training_end(self) -> None:
        self._record(self.num_timesteps)
        write_csv(self.out_csv, self.history)
        if self._env is not None:
            self._env.close()
            self._env = None


# ------------------------------------------------------------------ bookkeeping


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for k in row:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def append_result(algo: str, row: dict[str, Any]) -> None:
    """Append one sweep row to that algorithm's results table."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{algo.lower()}_sweep.csv"
    rows = read_results(algo)
    rows = [r for r in rows if r.get("run_id") != row.get("run_id")]
    rows.append(row)
    rows.sort(key=lambda r: str(r.get("run_id", "")))
    write_csv(path, rows)


def read_results(algo: str) -> list[dict[str, Any]]:
    path = RESULTS_DIR / f"{algo.lower()}_sweep.csv"
    if not path.exists():
        return []
    with path.open() as fh:
        return list(csv.DictReader(fh))


def convergence_step(history: list[dict[str, Any]], frac: float = 0.9) -> int | float:
    """First timestep at which the eval return reaches ``frac`` of the run's best.

    A single number for "how fast did this configuration get there", used as the
    convergence column of the hyperparameter tables.
    """
    if not history:
        return float("nan")
    returns = [float(h["mean_return"]) for h in history]
    best = max(returns)
    floor = min(returns)
    if best <= floor:
        return float("nan")
    threshold = floor + frac * (best - floor)
    for h, r in zip(history, returns):
        if r >= threshold:
            return int(h["timesteps"])
    return float("nan")


@dataclass
class RunSpec:
    """One row of a hyperparameter table."""

    run_id: str
    algo: str
    params: dict[str, Any]
    net_arch: tuple[int, ...] = (128, 128)
    total_timesteps: int = 200_000
    n_envs: int = 8
    seed: int = 0

    @property
    def tag(self) -> str:
        return f"{self.algo.lower()}_{self.run_id}"

    def dirs(self) -> tuple[Path, Path]:
        log = LOG_DIR / self.algo.lower() / self.run_id
        model = MODEL_DIR / ("dqn" if self.algo.lower() == "dqn" else "pg")
        log.mkdir(parents=True, exist_ok=True)
        model.mkdir(parents=True, exist_ok=True)
        return log, model


def finish_run(
    spec: RunSpec,
    result: EvalResult,
    history: list[dict[str, Any]],
    wall_time: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble and persist the results row for one sweep run."""
    row: dict[str, Any] = {
        "run_id": spec.run_id,
        "algo": spec.algo,
        **{k: _fmt(v) for k, v in spec.params.items()},
        "timesteps": spec.total_timesteps,
        **result.as_row(),
        "convergence_step": convergence_step(history),
        "wall_time_s": round(wall_time, 1),
    }
    if extra:
        row.update(extra)
    append_result(spec.algo, row)

    log_dir, _ = spec.dirs()
    (log_dir / "summary.json").write_text(
        json.dumps({"row": row, "outcomes": result.outcomes}, indent=2, default=str)
    )
    return row


def _fmt(v: Any) -> Any:
    if isinstance(v, (list, tuple)):
        return "x".join(str(x) for x in v)
    if callable(v):
        return getattr(v, "__name__", "fn")
    return v


def run_sb3(
    spec: RunSpec,
    model_cls,
    config: EnvConfig | None = None,
    eval_every: int | None = None,
    save_model: bool = True,
    verbose: int = 1,
) -> dict[str, Any]:
    """Train one SB3 configuration, score it and persist everything."""
    from stable_baselines3.common.logger import configure

    log_dir, model_dir = spec.dirs()
    eval_every = eval_every or max(spec.total_timesteps // 12, 2_000)

    env = make_vec_env(spec.n_envs, seed=spec.seed, config=config)
    try:
        model = model_cls(
            "MlpPolicy",
            env,
            seed=spec.seed,
            verbose=0,
            device="cpu",
            policy_kwargs={"net_arch": list(spec.net_arch)},
            **spec.params,
        )
        # CSV output is what the loss / entropy / Q-value curves are read back
        # from, so it has to be configured before learning starts.
        model.set_logger(configure(str(log_dir), ["csv"]))

        cb = PeriodicEvalCallback(
            eval_every=eval_every,
            out_csv=log_dir / "evals.csv",
            config=config,
            verbose=verbose,
        )
        watch = Stopwatch()
        with watch:
            model.learn(total_timesteps=spec.total_timesteps, callback=cb, progress_bar=False)

        result = evaluate_policy(
            lambda o: int(model.predict(o, deterministic=True)[0]), config=config
        )
        if save_model:
            model.save(str(model_dir / spec.tag))
    finally:
        env.close()

    row = finish_run(spec, result, cb.history, watch.seconds)
    if verbose:
        print(
            f"  -> {spec.run_id}: return {result.mean_return:8.2f}"
            f"  survey {100 * result.success_rate:5.1f}%"
            f"  scan-off {result.mean_miss:4.2f} m  ({watch.seconds:.0f}s)"
        )
    return row


def run_reinforce(
    spec: RunSpec,
    config: EnvConfig | None = None,
    eval_every: int | None = None,
    save_model: bool = True,
    verbose: int = 1,
) -> dict[str, Any]:
    """Train one REINFORCE configuration through the same protocol as SB3 runs."""
    from training.reinforce import REINFORCE

    log_dir, model_dir = spec.dirs()
    eval_every = eval_every or max(spec.total_timesteps // 12, 2_000)

    env = make_vec_env(spec.n_envs, seed=spec.seed, config=config)
    try:
        model = REINFORCE(
            env, hidden=spec.net_arch, seed=spec.seed, verbose=0, **spec.params
        )
        cb = PeriodicEvalCallback(
            eval_every=eval_every,
            out_csv=log_dir / "evals.csv",
            config=config,
            verbose=verbose,
        )
        watch = Stopwatch()
        with watch:
            model.learn(total_timesteps=spec.total_timesteps, callback=cb)

        result = evaluate_policy(
            lambda o: int(model.predict(o, deterministic=True)[0]), config=config
        )
        model.dump_train_log(log_dir / "train_log.json")
        if save_model:
            model.save(model_dir / spec.tag)
    finally:
        env.close()

    row = finish_run(spec, result, cb.history, watch.seconds)
    if verbose:
        print(
            f"  -> {spec.run_id}: return {result.mean_return:8.2f}"
            f"  survey {100 * result.success_rate:5.1f}%"
            f"  scan-off {result.mean_miss:4.2f} m  ({watch.seconds:.0f}s)"
        )
    return row


#: Where each algorithm's saved policy lives, and how to rebuild it.
ALGO_FAMILY = {"dqn": "dqn", "ppo": "pg", "a2c": "pg", "reinforce": "pg"}


def model_path(algo: str, run_id: str = "final") -> Path:
    return MODEL_DIR / ALGO_FAMILY[algo.lower()] / f"{algo.lower()}_{run_id}"


def best_sweep_run(algo: str) -> str | None:
    """The highest-scoring sweep configuration for an algorithm."""
    rows = [r for r in read_results(algo) if r.get("run_id") != "final"]
    if not rows:
        return None
    return max(rows, key=lambda r: float(r.get("mean_return", "-inf")))["run_id"]


def resolve_run_id(algo: str) -> str | None:
    """Pick the highest-scoring policy actually on disk.

    Ranked by held-out score rather than by "prefer the longer run": a longer
    budget is not automatically a better policy, and A2C's extended run in fact
    scored well below its best sweep configuration. Every sweep configuration is
    saved, so an algorithm whose extended run was never performed still has ten
    trained policies to choose from and stays in the comparison.
    """
    scored: list[tuple[float, str]] = []
    for row in read_results(algo):
        run_id = row.get("run_id")
        if not run_id:
            continue
        base = model_path(algo, run_id)
        if not any(base.with_suffix(sfx).exists() for sfx in (".zip", ".pt")):
            continue
        try:
            scored.append((float(row.get("mean_return", "-inf")), run_id))
        except ValueError:
            continue
    return max(scored)[1] if scored else None


def available_agents() -> dict[str, Path]:
    """Which trained policies are actually on disk."""
    found = {}
    for algo in ALGO_FAMILY:
        run_id = resolve_run_id(algo)
        if run_id is None:
            continue
        base = model_path(algo, run_id)
        for suffix in (".zip", ".pt"):
            if base.with_suffix(suffix).exists():
                found[algo] = base.with_suffix(suffix)
                break
    return found


def load_agent(algo: str, run_id: str | None = None, env=None):
    """Load a trained policy and return ``(predict_fn, label)``.

    ``predict_fn`` maps an observation to a greedy action, which is the same
    interface the scripted pilot and the evaluation helpers use.
    """
    algo = algo.lower()
    run_id = run_id or resolve_run_id(algo) or "final"
    base = model_path(algo, run_id)
    if algo == "reinforce":
        from training.reinforce import REINFORCE

        path = base.with_suffix(".pt")
        if not path.exists():
            raise FileNotFoundError(path)
        probe = env or SubseaInspectionEnv()
        model = REINFORCE.load(path, env=probe)
        if env is None:
            probe.close()
    else:
        from stable_baselines3 import A2C, DQN, PPO

        cls = {"dqn": DQN, "ppo": PPO, "a2c": A2C}[algo]
        path = base.with_suffix(".zip")
        if not path.exists():
            raise FileNotFoundError(path)
        model = cls.load(str(path), device="cpu")

    return (lambda obs: int(model.predict(obs, deterministic=True)[0])), f"{algo.upper()} ({run_id})"


def best_available_agent():
    """Pick the strongest trained policy on disk, or fall back to the pilot."""
    agents = available_agents()
    if not agents:
        from tests.scripted_pilot import scripted_policy

        return None, "scripted pilot (no trained model found)", scripted_policy

    scored = []
    for algo in agents:
        run_id = resolve_run_id(algo)
        rows = [r for r in read_results(algo) if r.get("run_id") == run_id]
        if not rows:
            rows = read_results(algo)
        score = max((float(r.get("mean_return", "-inf")) for r in rows), default=float("-inf"))
        scored.append((score, algo))
    scored.sort(reverse=True)
    algo = scored[0][1]
    predict, label = load_agent(algo)
    return predict, label, None


class Stopwatch:
    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self._t0

    @property
    def seconds(self) -> float:
        return getattr(self, "elapsed", time.perf_counter() - self._t0)


def print_header(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")
