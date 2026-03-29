"""
Gymnasium environment for generalised option pricing / optimal stopping.

Supports:
  - M independent GBM processes (M=1 → standard, M>1 → basket/Asian-style)
  - L exercise rights
  - delta refractory period between exercises (0 = next step allowed)
  - Put payoff on mean price across processes

The environment replays a *pre-generated* price path so that LSM and DQN
can be trained and evaluated on exactly the same paths.

Observation: [s_1/K, s_2/K, ..., s_M/K, t/N, rights_left/L]
  - normalised prices, normalised time, normalised rights remaining
Action: 0 = hold, 1 = exercise (if allowed)
Reward: discounted payoff on exercise; 0 on hold; terminal exercise at maturity
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from dataclasses import dataclass
from typing import Optional


@dataclass
class OptionParams:
    """Parameters for the generalised option."""
    S0: float = 100.0
    K: float = 100.0
    T: float = 1.0
    r: float = 0.05
    sigma: float = 0.3
    N: int = 50       # exercise dates
    M: int = 1        # number of processes
    L: int = 1        # exercise rights
    delta: int = 0    # refractory period (0 = next step) WILL ALWAYS BE ZERO FOR MY EXPERIMENTS

    @property
    def dt(self):
        return self.T / self.N

    @property
    def df(self):
        return np.exp(-self.r * self.dt)


def generate_gbm_paths(params: OptionParams, n_paths: int,
                        seed: Optional[int] = None) -> np.ndarray:
    """Generate GBM paths.  Shape: (n_paths, N+1, M)."""
    rng = np.random.RandomState(seed)
    dt = params.dt
    Z = rng.standard_normal((n_paths, params.N, params.M))
    drift = (params.r - 0.5 * params.sigma ** 2) * dt
    diffusion = params.sigma * np.sqrt(dt) * Z
    log_ret = np.cumsum(drift + diffusion, axis=1)
    paths = np.zeros((n_paths, params.N + 1, params.M))
    paths[:, 0, :] = params.S0
    paths[:, 1:, :] = params.S0 * np.exp(log_ret)
    return paths


def payoff_put(prices: np.ndarray, K: float) -> np.ndarray:
    """Put payoff on mean price.  prices shape (..., M)."""
    avg = prices.mean(axis=-1)
    return np.maximum(K - avg, 0.0)


class OptionPricingEnv(gym.Env):
    """
    Gymnasium environment that replays one pre-generated price path per episode.

    Usage
    -----
    paths = generate_gbm_paths(params, n_paths)
    env = OptionPricingEnv(params, paths)
    obs, info = env.reset()          # picks next path in round-robin
    obs, reward, terminated, truncated, info = env.step(action)
    """

    metadata = {"render_modes": []}

    def __init__(self, params: OptionParams, paths: np.ndarray):
        """
        Parameters
        ----------
        params : OptionParams
        paths  : ndarray of shape (n_paths, N+1, M)
        """
        super().__init__()
        self.params = params
        self.paths = paths
        self.n_paths = len(paths)
        self._path_idx = 0

        # obs = [s_1/K, ..., s_M/K,  t/N,  rights_left/L]
        obs_dim = params.M + 2
        self.observation_space = spaces.Box(
            low=0.0, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(2)  # 0=hold, 1=exercise

        # internal state
        self._t: int = 0
        self._rights: int = params.L
        self._path: np.ndarray = np.empty(0)
        self._cumulative_reward: float = 0.0
        self._cooldown: int = 0  # remaining cooldown steps

    # ------------------------------------------------------------------
    def _obs(self) -> np.ndarray:
        prices_norm = self._path[self._t] / self.params.K
        t_norm = np.float32(self._t / self.params.N)
        r_norm = np.float32(self._rights / self.params.L)
        return np.concatenate([prices_norm.astype(np.float32),
                               [t_norm, r_norm]])

    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # round-robin through paths
        self._path = self.paths[self._path_idx % self.n_paths]
        self._path_idx += 1

        self._t = 1                 # first possible exercise date
        self._rights = self.params.L
        self._cumulative_reward = 0.0
        self._cooldown = 0
        return self._obs(), {}

    def reset_to_path(self, idx: int):
        """Reset to a specific path index (for evaluation)."""
        self._path = self.paths[idx]
        self._t = 1
        self._rights = self.params.L
        self._cumulative_reward = 0.0
        self._cooldown = 0
        return self._obs(), {}

    # ------------------------------------------------------------------
    def step(self, action: int):
        p = self.params
        reward = 0.0
        terminated = False
        truncated = False

        if self._cooldown > 0:
            # in refractory period — forced hold regardless of action
            action = 0
            self._cooldown -= 1

        if action == 1 and self._rights > 0:
            # exercise one right
            pay = payoff_put(self._path[self._t].reshape(1, -1), p.K).item()
            reward = pay * (p.df ** self._t)
            self._cumulative_reward += reward
            self._rights -= 1
            self._cooldown = p.delta   # delta extra hold steps

        # advance time
        self._t += 1

        if self._t > p.N or self._rights <= 0:
            # maturity or no rights left
            if self._t <= p.N and self._rights <= 0:
                pass  # no rights — done
            elif self._t > p.N and self._rights > 0:
                # auto-exercise at maturity if ITM (1 right)
                pay = payoff_put(self._path[p.N].reshape(1, -1), p.K).item()
                if pay > 0:
                    reward += pay * (p.df ** p.N)
                    self._cumulative_reward += pay * (p.df ** p.N)
                    self._rights -= 1
            terminated = True
            return self._obs() if self._t <= p.N else self._terminal_obs(), \
                   reward, terminated, truncated, \
                   {"cumulative_reward": self._cumulative_reward}

        return self._obs(), reward, terminated, truncated, \
               {"cumulative_reward": self._cumulative_reward}

    def _terminal_obs(self):
        """Observation when past maturity (terminal padding)."""
        prices_norm = self._path[self.params.N] / self.params.K
        t_norm = np.float32(1.0)
        r_norm = np.float32(self._rights / self.params.L)
        return np.concatenate([prices_norm.astype(np.float32),
                               [t_norm, r_norm]])
