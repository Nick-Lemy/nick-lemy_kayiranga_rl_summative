"""REINFORCE (Monte-Carlo policy gradient), written to look like an SB3 model.

Stable-Baselines3 ships DQN, A2C and PPO but not REINFORCE, so this is a
from-scratch implementation. It deliberately exposes the same surface as an SB3
algorithm - ``learn``, ``predict``, ``save``, ``load``, ``num_timesteps``,
``logger`` - so the shared evaluation callback, the sweep runner and the plotting
code treat all four algorithms identically and the comparison stays honest.

What makes this REINFORCE rather than A2C
-----------------------------------------
The gradient is estimated from **complete episodes** using the discounted
reward-to-go, with no bootstrapping anywhere. The optional value network is used
purely as a variance-reduction baseline subtracted from that Monte-Carlo return;
it never supplies a TD target. That distinction is the whole point of having
REINFORCE in the comparison: it is unbiased but high-variance, and the sweep is
set up to show exactly that trade-off against the bootstrapped methods.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


def _mlp(in_dim: int, out_dim: int, hidden: Sequence[int], activation: str = "tanh") -> nn.Module:
    act = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU}[activation]
    layers: list[nn.Module] = []
    last = in_dim
    for h in hidden:
        layers += [nn.Linear(last, h), act()]
        last = h
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


class _DummyLogger:
    """Stands in for SB3's logger so shared callbacks can be reused."""

    def record(self, *args, **kwargs) -> None:  # pragma: no cover
        pass

    def dump(self, *args, **kwargs) -> None:  # pragma: no cover
        pass


class REINFORCE:
    """Monte-Carlo policy gradient with an optional learned baseline.

    Parameters mirror the hyperparameters swept in the report:

    ``learning_rate``
        Adam step size for the policy.
    ``gamma``
        Discount applied to the reward-to-go.
    ``episodes_per_update``
        How many complete episodes are averaged into one gradient step. Larger
        batches trade sample efficiency for a lower-variance gradient.
    ``use_baseline``
        Subtract a learned state-value baseline from the return. This is the
        single biggest variance lever in the algorithm.
    ``normalize_returns``
        Standardise advantages within the batch before the gradient step.
    ``ent_coef``
        Entropy bonus; keeps the categorical policy from collapsing early.
    """

    def __init__(
        self,
        env,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        episodes_per_update: int = 16,
        hidden: Sequence[int] = (128, 128),
        activation: str = "tanh",
        use_baseline: bool = True,
        baseline_lr: float | None = None,
        normalize_returns: bool = True,
        ent_coef: float = 0.01,
        max_grad_norm: float | None = 0.5,
        seed: int = 0,
        device: str = "cpu",
        verbose: int = 0,
    ) -> None:
        self.env = env
        self.n_envs = getattr(env, "num_envs", 1)
        obs_space = env.observation_space
        act_space = env.action_space
        self.obs_dim = int(np.prod(obs_space.shape))
        self.n_actions = int(act_space.n)

        self.gamma = gamma
        self.episodes_per_update = episodes_per_update
        self.use_baseline = use_baseline
        self.normalize_returns = normalize_returns
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.device = torch.device(device)

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.policy = _mlp(self.obs_dim, self.n_actions, hidden, activation).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)

        self.value: nn.Module | None = None
        if use_baseline:
            self.value = _mlp(self.obs_dim, 1, hidden, activation).to(self.device)
            self.value_optimizer = torch.optim.Adam(
                self.value.parameters(), lr=baseline_lr or learning_rate
            )

        self.num_timesteps = 0
        self.logger = _DummyLogger()
        #: per-update diagnostics, used for the entropy and loss curves
        self.train_log: list[dict[str, float]] = []

    # ------------------------------------------------------------------ SB3 API

    def get_env(self):
        return self.env

    def predict(
        self, observation: np.ndarray, state=None, episode_start=None, deterministic: bool = False
    ):
        obs = np.asarray(observation, dtype=np.float32)
        single = obs.ndim == 1
        if single:
            obs = obs[None, :]
        with torch.no_grad():
            logits = self.policy(torch.as_tensor(obs, device=self.device))
            if deterministic:
                action = torch.argmax(logits, dim=-1)
            else:
                action = Categorical(logits=logits).sample()
        action_np = action.cpu().numpy()
        return (int(action_np[0]) if single else action_np), state

    def save(self, path: str | Path) -> None:
        path = Path(path).with_suffix(".pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy": self.policy.state_dict(),
                "value": self.value.state_dict() if self.value is not None else None,
                "config": {
                    "obs_dim": self.obs_dim,
                    "n_actions": self.n_actions,
                    "gamma": self.gamma,
                    "ent_coef": self.ent_coef,
                },
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, env=None, **kwargs) -> "REINFORCE":
        path = Path(path).with_suffix(".pt")
        blob = torch.load(path, map_location="cpu", weights_only=False)
        hidden = kwargs.pop("hidden", (128, 128))
        model = cls(env=env, hidden=hidden, **kwargs)
        model.policy.load_state_dict(blob["policy"])
        if blob.get("value") is not None and model.value is not None:
            model.value.load_state_dict(blob["value"])
        return model

    # ----------------------------------------------------------------- training

    def learn(
        self,
        total_timesteps: int,
        callback=None,
        progress_callback: Callable[[int], None] | None = None,
    ) -> "REINFORCE":
        if callback is not None:
            callback.init_callback(self)
            callback.on_training_start({}, {})

        env = self.env
        obs = env.reset()
        # one in-flight trajectory per parallel environment
        buffers: list[dict[str, list]] = [
            {"obs": [], "act": [], "rew": []} for _ in range(self.n_envs)
        ]
        finished: list[dict[str, np.ndarray]] = []

        while self.num_timesteps < total_timesteps:
            obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device)
            with torch.no_grad():
                logits = self.policy(obs_t)
                actions = Categorical(logits=logits).sample().cpu().numpy()

            next_obs, rewards, dones, infos = env.step(actions)
            self.num_timesteps += self.n_envs

            for i in range(self.n_envs):
                buffers[i]["obs"].append(np.asarray(obs[i], dtype=np.float32))
                buffers[i]["act"].append(int(actions[i]))
                buffers[i]["rew"].append(float(rewards[i]))
                if dones[i]:
                    finished.append(
                        {
                            "obs": np.array(buffers[i]["obs"], dtype=np.float32),
                            "act": np.array(buffers[i]["act"], dtype=np.int64),
                            "rew": np.array(buffers[i]["rew"], dtype=np.float32),
                        }
                    )
                    buffers[i] = {"obs": [], "act": [], "rew": []}

            obs = next_obs

            if callback is not None and not callback.on_step():
                break

            if len(finished) >= self.episodes_per_update:
                self._update(finished)
                finished = []

        if callback is not None:
            callback.on_training_end()
        return self

    def _update(self, episodes: list[dict[str, np.ndarray]]) -> None:
        """One policy-gradient step from a batch of complete episodes."""
        obs_all, act_all, ret_all = [], [], []
        ep_returns = []
        for ep in episodes:
            rewards = ep["rew"]
            # discounted reward-to-go, computed backwards
            returns = np.empty_like(rewards)
            running = 0.0
            for t in range(len(rewards) - 1, -1, -1):
                running = rewards[t] + self.gamma * running
                returns[t] = running
            obs_all.append(ep["obs"])
            act_all.append(ep["act"])
            ret_all.append(returns)
            ep_returns.append(float(rewards.sum()))

        obs = torch.as_tensor(np.concatenate(obs_all), device=self.device)
        actions = torch.as_tensor(np.concatenate(act_all), device=self.device)
        returns = torch.as_tensor(np.concatenate(ret_all), device=self.device)

        # --- baseline -------------------------------------------------------
        value_loss = torch.zeros((), device=self.device)
        if self.value is not None:
            values = self.value(obs).squeeze(-1)
            value_loss = nn.functional.mse_loss(values, returns)
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()
            with torch.no_grad():
                advantages = returns - self.value(obs).squeeze(-1)
        else:
            advantages = returns

        if self.normalize_returns and advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # --- policy gradient -------------------------------------------------
        logits = self.policy(obs)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        policy_loss = -(log_probs * advantages).mean()
        loss = policy_loss - self.ent_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        self.train_log.append(
            {
                "timesteps": self.num_timesteps,
                "policy_loss": float(policy_loss.item()),
                "value_loss": float(value_loss.item()),
                "entropy": float(entropy.item()),
                "ep_rew_mean": float(np.mean(ep_returns)),
                "episodes": len(episodes),
            }
        )
        if self.verbose and len(self.train_log) % 10 == 0:
            last = self.train_log[-1]
            print(
                f"    [reinforce] {self.num_timesteps:>8,} steps  "
                f"rew {last['ep_rew_mean']:8.2f}  entropy {last['entropy']:.3f}"
            )

    def dump_train_log(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.train_log, indent=1))


__all__ = ["REINFORCE"]
