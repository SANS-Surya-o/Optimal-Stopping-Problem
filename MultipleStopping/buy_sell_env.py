"""
Buy-Sell Environment: Multiple-stop optimal stopping problem.

Generic environment supporting any number of stops:
- Odd stops (1, 3, 5, ...): Buying actions
- Even stops (2, 4, 6, ...): Selling actions
- Holding cost incurred while holding an asset (after buy, before sell)

Examples:
- max_stops=1: Single stopping (e.g., just buy a house)
- max_stops=2: Buy-Sell once
- max_stops=4: Buy-Sell twice
"""
from typing import Optional, Dict, Any, Tuple, List
import numpy as np
from base_env import BaseStoppingEnv


class BuySellEnv(BaseStoppingEnv):
    """
    Buy-Sell environment with configurable number of stops.
    
    Stop interpretation:
    - Odd stops (1, 3, 5, ...): Buy actions (acquire asset at current price)
    - Even stops (2, 4, 6, ...): Sell actions (liquidate asset at current price)
    
    Holding cost is incurred per time step while holding an asset
    (i.e., between odd and even stops).
    """
    
    def __init__(
        self,
        n_states: int,
        max_time: int,
        max_stops: int = 2,
        holding_cost_per_step: float = 0.1,
        offer_values: Optional[np.ndarray] = None,
        transition_model: str = 'random_walk',
        transition_params: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize Buy-Sell environment.
        
        Args:
            n_states: Number of discrete offer states
            max_time: Maximum time horizon
            max_stops: Maximum number of stops allowed (default=2 for buy-sell)
            holding_cost_per_step: Cost incurred per time step while holding asset
            offer_values: Array mapping state index to actual offer value
            transition_model: Type of stochastic process
            transition_params: Parameters for transition model
            seed: Random seed
        """
        # Initialize with configurable max_stops
        super().__init__(
            n_states=n_states,
            max_time=max_time,
            max_stops=max_stops,
            transition_model=transition_model,
            transition_params=transition_params,
            seed=seed
        )
        
        self.holding_cost_per_step = holding_cost_per_step
        
        # Map state indices to actual offer values
        if offer_values is None:
            # Default: linearly spaced values between 0 and n_states
            self.offer_values = np.linspace(0, n_states-1, n_states)
        else:
            assert len(offer_values) == n_states, \
                f"offer_values must have length {n_states}"
            self.offer_values = offer_values
        
        # Track transaction history (buy/sell prices and times)
        self.transaction_history = []
        self.currently_holding = False
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, int], Dict[str, Any]]:
        """Reset environment and clear transaction tracking."""
        obs, info = super().reset(seed=seed, options=options)
        
        self.transaction_history = []
        self.currently_holding = False
        
        return obs, info
    
    def _compute_reward(self, action: int) -> float:
        """
        Compute reward for current action.
        
        Reward structure (generalized for any number of stops):
        - Odd stops (buy): Pay -price and incur -holding_cost
        - Even stops (sell): Receive +price (no holding cost that step)
        - While holding (after buy, before sell): Pay -holding_cost per step
        - Not holding: No cost
        
        Args:
            action: 0 = continue/reject, 1 = stop/accept
            
        Returns:
            Reward value
        """
        current_value = self.offer_values[self.current_offer]
        stops_remaining = self.max_stops - self.stops_made
        
        # Determine if this would be a buy (odd) or sell (even) stop
        next_stop_number = self.stops_made + 1  # 1-indexed
        is_buy_action = (next_stop_number % 2 == 1)  # Odd = buy
        
        if action == 1:  # Accept/Stop
            # Record transaction
            self.transaction_history.append({
                'stop_number': next_stop_number,
                'action_type': 'buy' if is_buy_action else 'sell',
                'price': current_value,
                'time': self.current_time
            })
            
            if is_buy_action:
                # Buy action: Pay price and holding cost
                self.currently_holding = True
                return -current_value - self.holding_cost_per_step
            else:
                # Sell action: Receive price (no holding cost this step)
                self.currently_holding = False
                return current_value
        else:  # action == 0: Reject/Continue
            # If currently holding an asset, pay holding cost
            if self.currently_holding:
                return -self.holding_cost_per_step
            else:
                # Not holding, no cost
                return 0.0
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional information including transaction details."""
        info = super()._get_info()
        info.update({
            'transaction_history': self.transaction_history,
            'currently_holding': self.currently_holding
        })
        
        # For backward compatibility, add buy/sell prices if max_stops == 2
        if self.max_stops == 2:
            buy_price = None
            sell_price = None
            buy_time = None
            sell_time = None
            
            for txn in self.transaction_history:
                if txn['action_type'] == 'buy':
                    buy_price = txn['price']
                    buy_time = txn['time']
                elif txn['action_type'] == 'sell':
                    sell_price = txn['price']
                    sell_time = txn['time']
            
            info.update({
                'buy_price': buy_price,
                'buy_time': buy_time,
                'sell_price': sell_price,
                'sell_time': sell_time
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
        
        Generalized for any number of stops:
        - Odd stops (1, 3, 5, ...): Buy actions
        - Even stops (2, 4, 6, ...): Sell actions
        - Holding cost incurred while holding (between buy and sell)
        
        Key logic:
        - Buy at time t: Pay -price - holding_cost, transition to (s', t-1, stops-1)
        - Sell at time t: Receive +price (no holding cost), transition to (s', t-1, stops-1)
        - Reject while holding: Pay -holding_cost, transition to (s', t-1, same stops)
        - Reject while not holding: No cost, transition to (s', t-1, same stops)
        
        Args:
            verbose: Whether to print progress information
            
        Returns:
            Tuple of (V, policy) where:
            - V[s, t, n]: Optimal value with offer s, time_left t, stops_left n
            - policy[s, t, n]: Optimal action (0=reject, 1=accept)
        """
        # Initialize
        V = np.zeros((self.n_states, self.max_time + 1, self.max_stops + 1))
        policy = np.zeros((self.n_states, self.max_time + 1, self.max_stops + 1), dtype=int)
        P = self.P  # Transition probabilities
        cost = self.holding_cost_per_step
        
        # Base case: time_left = 0 (no time remaining, all forced terminal)
        for n in range(self.max_stops + 1):
            V[:, 0, n] = 0.0  # Terminal state, no value
        
        # Note: This solver currently only handles max_stops=2 (single buy-sell pair)
        if self.max_stops != 2:
            raise NotImplementedError(f"solve_optimal_policy currently only supports max_stops=2, got {self.max_stops}")
        
        # For max_stops=2: First stop is buy (stops_left=2), second stop is sell (stops_left=1)
        
        # Base case: time_left = 1
        # stops_left = 0: Already done, no value
        V[:, 1, 0] = 0.0
        policy[:, 1, 0] = 0
        
        # stops_left = 1: Must sell now (last chance)
        V[:, 1, 1] = self.offer_values  # Receive sell price
        policy[:, 1, 1] = 1
        
        # stops_left = 2: Can't complete buy-sell in 1 step, so don't buy
        V[:, 1, 2] = 0  # Impossible to complete profitably
        policy[:, 1, 2] = 0
        
        # Backward induction for t >= 2
        for t in range(2, self.max_time + 1):
            # stops_left = 0: Already completed both stops
            V[:, t, 0] = 0.0
            policy[:, t, 0] = 0
            
            # stops_left = 1: Need to sell (currently holding)
            for s in range(self.n_states):
                # Accept: Sell at current price, no more holding cost
                value_accept = self.offer_values[s]
                
                # Reject: Pay holding cost and continue
                value_reject = -cost + np.dot(P[s, :], V[:, t-1, 1])
                
                if value_accept > value_reject:
                    V[s, t, 1] = value_accept
                    policy[s, t, 1] = 1
                else:
                    V[s, t, 1] = value_reject
                    policy[s, t, 1] = 0
            
            # stops_left = 2: Need to buy (not yet holding)
            for s in range(self.n_states):
                # Accept: Buy at current price, pay holding cost, then continue with stops_left=1
                value_accept = -self.offer_values[s] - cost + np.dot(P[s, :], V[:, t-1, 1])
                
                # Reject: Don't buy, no cost, continue with stops_left=2
                value_reject = np.dot(P[s, :], V[:, t-1, 2])
                
                if value_accept >= value_reject:
                    V[s, t, 2] = value_accept
                    policy[s, t, 2] = 1
                else:
                    V[s, t, 2] = value_reject
                    policy[s, t, 2] = 0
        
        if verbose:
            print("✓ Optimal policy computed via DP")
            print(f"  Max stops: {self.max_stops}")
            print(f"  Expected value at start: {V[:, self.max_time, self.max_stops].mean():.4f}")
            
            # Print accept rates for each stop type
            for n in range(self.max_stops, 0, -1):
                stops_to_make = self.max_stops - n + 1
                action_type = "Buy" if stops_to_make % 2 == 1 else "Sell"
                accept_rate = policy[:, 1:, n].mean()
                print(f"  {action_type} (stop {stops_to_make}) accept rate: {accept_rate:.2%}")
        
        return V, policy
