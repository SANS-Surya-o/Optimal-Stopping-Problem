"""
Accept Reward Environment: Simple single-stopping problem where you receive value.

This is a classic secretary/house-hunting problem where:
- You observe offers over time
- You can accept an offer once and receive its value as reward
- Or reject and continue to the next offer
- Goal: Maximize expected reward

This is useful for testing Q-Learning approaches without the complication of buy-sell pairs.
"""
from typing import Optional, Dict, Any, Tuple
import numpy as np
from base_env import BaseStoppingEnv


class AcceptRewardEnv(BaseStoppingEnv):
    """
    Simple accept-reward environment: single stopping problem.
    
    - max_stops is always 1 (single decision)
    - Accepting gives reward = current offer value
    - No holding costs or complicated dynamics
    - Classic optimal stopping problem
    """
    
    def __init__(
        self,
        n_states: int,
        max_time: int,
        offer_values: Optional[np.ndarray] = None,
        transition_model: str = 'random_walk',
        transition_params: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize Accept-Reward environment.
        
        Args:
            n_states: Number of discrete offer states
            max_time: Maximum time horizon
            offer_values: Array mapping state index to actual offer value
            transition_model: Type of stochastic process
            transition_params: Parameters for transition model
            seed: Random seed
        """
        # Always single stopping
        super().__init__(
            n_states=n_states,
            max_time=max_time,
            max_stops=1,  # Always 1 for this environment
            transition_model=transition_model,
            transition_params=transition_params,
            seed=seed
        )
        
        # Map state indices to actual offer values
        if offer_values is None:
            # Default: linearly spaced values between 0 and n_states
            self.offer_values = np.linspace(0, n_states, n_states)
        else:
            assert len(offer_values) == n_states, \
                f"offer_values must have length {n_states}"
            self.offer_values = offer_values
        
        # Track accepted offer
        self.accepted_value = None
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, int], Dict[str, Any]]:
        """Reset environment."""
        obs, info = super().reset(seed=seed, options=options)
        self.accepted_value = None
        return obs, info
    
    def _compute_reward(self, action: int) -> float:
        """
        Compute reward for current action.
        
        Reward structure:
        - Accept (action=1): Receive current offer value
        - Reject (action=0): Get 0 reward, continue searching
        
        Args:
            action: 0 = continue/reject, 1 = stop/accept
            
        Returns:
            Reward value
        """
        current_value = self.offer_values[self.current_offer]
        
        if action == 1:  # Accept
            self.accepted_value = current_value
            return current_value
        else:  # Reject
            return 0.0
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional information."""
        info = super()._get_info()
        info.update({
            'accepted_value': self.accepted_value,
        })
        return info
    
    def get_state_value(self, state_idx: int) -> float:
        """
        Get the actual value corresponding to a state index.
        
        Args:
            state_idx: State index
            
        Returns:
            Actual offer value
        """
        return self.offer_values[state_idx]
    
    def solve_optimal_policy(self, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute optimal value function and policy using dynamic programming.
        
        Classic optimal stopping problem:
        - V(s, t, 1) = max(s, E[V(s', t-1, 1)])  where s is the offer value
        - Accept if current value >= expected continuation value
        
        Args:
            verbose: Whether to print progress information
            
        Returns:
            Tuple of (V, policy) where:
            - V[s, t, n]: Optimal value with offer s, time_left t, stops_left n
            - policy[s, t, n]: Optimal action (0=reject, 1=accept)
        """
        # Initialize
        V = np.zeros((self.n_states, self.max_time + 1, 2))  # stops_left is 0 or 1
        policy = np.zeros((self.n_states, self.max_time + 1, 2), dtype=int)
        P = self.P  # Transition probabilities
        
        # Base case: time_left = 0 (terminal, no value)
        V[:, 0, :] = 0.0
        
        # Base case: time_left = 1, stops_left = 0 (already used stop, no value)
        V[:, 1, 0] = 0.0
        policy[:, 1, 0] = 0
        
        # Base case: time_left = 1, stops_left = 1 (must accept now or get nothing)
        V[:, 1, 1] = self.offer_values  # Accept current offer
        policy[:, 1, 1] = 1
        
        # Backward induction
        for t in range(2, self.max_time + 1):
            # stops_left = 0 (already stopped, no value)
            V[:, t, 0] = 0.0
            policy[:, t, 0] = 0
            
            # stops_left = 1 (haven't stopped yet, can choose)
            for s in range(self.n_states):
                # Accept: receive current offer value, then terminal
                value_accept = self.offer_values[s]
                
                # Reject: continue, expect V(s', t-1, 1)
                value_reject = np.dot(P[s, :], V[:, t-1, 1])
                
                if value_accept >= value_reject:
                    V[s, t, 1] = value_accept
                    policy[s, t, 1] = 1
                else:
                    V[s, t, 1] = value_reject
                    policy[s, t, 1] = 0
        
        if verbose:
            print("✓ Optimal policy computed via DP")
            print(f"  Max stops: {self.max_stops} (accept-reward problem)")
            print(f"  Expected value at start: {V[:, self.max_time, 1].mean():.4f}")
            
            # Compute threshold (lowest value that should be accepted)
            accept_threshold = np.zeros(self.max_time + 1)
            for t in range(1, self.max_time + 1):
                accepting_states = np.where(policy[:, t, 1] == 1)[0]
                if len(accepting_states) > 0:
                    accept_threshold[t] = self.offer_values[accepting_states[0]]
                else:
                    accept_threshold[t] = np.inf
            
            print(f"  Accept thresholds (by time_left):")
            sample_times = [self.max_time, self.max_time//2, max(1, self.max_time//4), 1]
            for t in sample_times:
                if t <= self.max_time:
                    print(f"    t={t:2d}: {accept_threshold[t]:.2f}")
            
            accept_rate = policy[:, 1:, 1].mean()
            print(f"  Overall accept rate: {accept_rate:.2%}")
        
        return V, policy
