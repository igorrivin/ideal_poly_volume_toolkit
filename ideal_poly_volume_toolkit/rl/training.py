"""
Problem-agnostic training and evaluation helpers for RL harness users.

Extracted from the old `rl_optimizer.py`.  The key generalization is that
`train_controller` takes an `env_factory: Callable[[], GenericOptEnv]`
rather than hard-coding the polytope env — any `GenericOptEnv` instance
works here (polytope, cage, Hadamard, future tasks).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, List, Optional, Union

import numpy as np

from .generic import GenericOptEnv, InnerOptimizer


# =====================================================================
# PPO training
# =====================================================================

def train_controller(
    env_factory: Callable[[], GenericOptEnv],
    *,
    total_timesteps: int = 500_000,
    save_path: Optional[Union[str, Path]] = None,
    seed: int = 0,
    policy_kwargs: Optional[dict] = None,
    tensorboard_log: Optional[Union[str, Path]] = None,
    device: str = "cpu",
    verbose: int = 1,
    n_steps: int = 2048,
    batch_size: int = 256,
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    ent_coef: float = 0.01,
):
    """Train a stable-baselines3 PPO policy on `env_factory()`.

    Returns the trained model.  If `save_path` is given, the model is
    saved to that path after training (SB3 automatically appends `.zip`).
    """
    from stable_baselines3 import PPO

    env = env_factory()
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=verbose,
        n_steps=n_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        gamma=gamma,
        ent_coef=ent_coef,
        seed=seed,
        device=device,
        policy_kwargs=policy_kwargs,
        tensorboard_log=(str(tensorboard_log) if tensorboard_log else None),
    )
    model.learn(total_timesteps=total_timesteps)
    if save_path is not None:
        model.save(str(save_path))
    return model


# =====================================================================
# Model evaluation
# =====================================================================

def evaluate_controller(
    model_or_path: Union[str, Path, "object"],
    env: GenericOptEnv,
    *,
    n_episodes: int = 10,
    deterministic: bool = True,
    seed: int = 0,
) -> dict:
    """Run a trained controller for `n_episodes` rollouts.

    `model_or_path` is either an SB3 model instance or a path to a saved
    `.zip`.  `env` is the evaluation env (typically narrower than the
    training env — e.g. a single (d, N) for the polytope case).

    Returns `{mean_value, std_value, max_value, mean_evals, results}`
    where `results` is a per-episode list.
    """
    if isinstance(model_or_path, (str, Path)):
        from stable_baselines3 import PPO
        model = PPO.load(str(model_or_path), env=env)
    else:
        model = model_or_path

    results: List[dict] = []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        total_reward = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            done = bool(terminated or truncated)
        results.append({
            "best_value": float(info["best_value"]),
            "total_evals": int(info["total_evals"]),
            "total_reward": float(total_reward),
        })

    values = np.array([r["best_value"] for r in results], dtype=float)
    return {
        "mean_value": float(values.mean()),
        "std_value": float(values.std()),
        "max_value": float(values.max()),
        "mean_evals": float(np.mean([r["total_evals"] for r in results])),
        "results": results,
    }


# =====================================================================
# Baseline comparison
# =====================================================================

def compare_with_baseline(
    env_factory: Callable[[], GenericOptEnv],
    *,
    n_trials: int = 20,
    budget_per_trial: int = 50,
    baselines: Optional[Iterable[InnerOptimizer]] = None,
    seed: int = 0,
) -> dict:
    """Run the configured inner optimizers in isolation on `env_factory()`
    tasks (no RL controller) to establish a baseline.

    Each trial constructs a fresh env (via `env_factory()`), resets it with
    a deterministic per-trial seed, and runs each baseline for
    `budget_per_trial` budget units on the task's initial random solution.

    If `baselines` is None, the env's own `inner_optimizers` are used.

    Returns `{baseline_name: {mean_value, std_value, max_value}, ...}`.
    """
    env = env_factory()
    baseline_list = list(baselines) if baselines is not None else list(env.inner_optimizers)

    summary: dict = {}
    for opt in baseline_list:
        per_trial = []
        for trial in range(n_trials):
            env = env_factory()
            env.reset(seed=seed + trial)
            task = env.task
            start_sol = task.copy_solution(env.root.best_solution)
            rng = np.random.default_rng(seed + 1_000_000 + trial)
            best_sol, values = opt.run(start_sol, task, budget_per_trial, rng)
            final = task.evaluate(best_sol)
            per_trial.append(final)
        arr = np.array(per_trial, dtype=float)
        summary[opt.name] = {
            "mean_value": float(arr.mean()),
            "std_value": float(arr.std()),
            "max_value": float(arr.max()),
            "n_trials": int(n_trials),
        }
    return summary
