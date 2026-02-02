"""Model-Based RL: Learn transition matrix from data, solve with DP."""
import numpy as np
from typing import Optional, Tuple


class ModelBasedAgent:
    """
    Model-based agent that learns P(s'|s) from experience and solves via DP.
    """
    
    def __init__(self, env, seed: Optional[int] = None):
        self.env = env
        self.n_states = env.n_states
        self.max_time = env.max_time
        self.max_stops = env.max_stops
        self.rng = np.random.default_rng(seed)
        
        # Transition counts: N[s, s'] = count of s -> s' transitions
        self.transition_counts = np.zeros((self.n_states, self.n_states))
        self.total_transitions = 0
        
        # Learned model and policy
        self.P_hat = None  # Empirical TPM
        self.V = None
        self.Q = None
        self.policy = None
        self.trained = False
    
    def update_counts(self, s: int, s_next: int):
        """Update transition counts."""
        self.transition_counts[s, s_next] += 1
        self.total_transitions += 1
    
    def get_empirical_tpm(self) -> np.ndarray:
        """Compute empirical transition matrix from counts."""
        P = np.zeros((self.n_states, self.n_states))
        for s in range(self.n_states):
            row_sum = self.transition_counts[s].sum()
            if row_sum > 0:
                P[s] = self.transition_counts[s] / row_sum
            else:
                P[s] = 1.0 / self.n_states  # Uniform prior
        return P
    
    def collect_transitions(self, n_episodes: int, verbose: int = 0):
        """Collect transitions by random exploration."""
        for ep in range(n_episodes):
            obs, _ = self.env.reset()
            done = False
            while not done:
                s = obs['offer']
                action = self.rng.integers(0, 2)  # Random action
                obs_next, _, terminated, truncated, _ = self.env.step(action)
                s_next = obs_next['offer']
                self.update_counts(s, s_next)
                obs = obs_next
                done = terminated or truncated
            
            if verbose and (ep + 1) % (n_episodes // 10) == 0:
                print(f"  Collected {ep + 1}/{n_episodes} episodes")
    
    def solve_dp(self, P: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Solve optimal policy via DP given transition matrix P."""
        V = np.zeros((self.n_states, self.max_time + 1, self.max_stops + 1))
        Q = np.zeros((self.n_states, self.max_time + 1, self.max_stops + 1, 2))
        policy = np.zeros((self.n_states, self.max_time + 1, self.max_stops + 1), dtype=int)
        
        cost = self.env.holding_cost_per_step
        prices = self.env.offer_values
        
        # Base cases
        V[:, 1, 1] = prices  # Must sell at time_left=1
        policy[:, 1, 1] = 1
        
        # Backward induction
        for t in range(2, self.max_time + 1):
            # stops_left = 1: Sell decision
            for s in range(self.n_states):
                value_sell = prices[s]
                value_wait = -cost + np.dot(P[s], V[:, t-1, 1])
                Q[s, t, 1, 0] = value_wait
                Q[s, t, 1, 1] = value_sell
                if value_sell > value_wait:
                    V[s, t, 1] = value_sell
                    policy[s, t, 1] = 1
                else:
                    V[s, t, 1] = value_wait
                    policy[s, t, 1] = 0
            
            # stops_left = 2: Buy decision
            for s in range(self.n_states):
                value_buy = -prices[s] - cost + np.dot(P[s], V[:, t-1, 1])
                value_wait = np.dot(P[s], V[:, t-1, 2])
                Q[s, t, 2, 0] = value_wait
                Q[s, t, 2, 1] = value_buy
                if value_buy >= value_wait:
                    V[s, t, 2] = value_buy
                    policy[s, t, 2] = 1
                else:
                    V[s, t, 2] = value_wait
                    policy[s, t, 2] = 0
        
        return V, Q, policy
    
    def train(self, n_episodes: int, verbose: int = 1):
        """Collect data and solve DP."""
        if verbose:
            print(f"Collecting {n_episodes} episodes...")
        self.collect_transitions(n_episodes, verbose)
        
        self.P_hat = self.get_empirical_tpm()
        self.V, self.Q, self.policy = self.solve_dp(self.P_hat)
        self.trained = True
        
        if verbose:
            print(f"Model-based training complete.")
            print(f"  Total transitions: {self.total_transitions}")
    
    def get_policy(self) -> np.ndarray:
        if not self.trained:
            raise ValueError("Agent not trained.")
        return self.policy.copy()
    
    def tpm_error(self) -> float:
        """Compute Frobenius norm of (P_hat - P_true)."""
        if self.P_hat is None:
            return np.nan
        return np.linalg.norm(self.P_hat - self.env.P, 'fro')
