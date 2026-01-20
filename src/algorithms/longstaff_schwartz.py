"""
Longstaff-Schwartz Algorithm for Multiple Optimal Stopping Problems.

For buy-sell problems with K stops:
1. Work backwards from the last stop (sell) to the first stop (buy)
2. At each stop level, use regression to estimate continuation values
3. The value of completing stop k becomes the "reward" for stop k-1

Reference: Longstaff & Schwartz (2001) for single stopping,
extended here for multiple stopping.
"""
import numpy as np
from typing import Optional, Dict, Any, Tuple, Callable, List
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class LSMResult:
    """Results from LSM training/evaluation."""
    mean_value: float
    std_error: float
    values: np.ndarray
    exercise_times: Dict[int, np.ndarray]  # stop_level -> exercise times
    training_time: float = 0.0


class LongstaffSchwartzAgentBuySell:
    """
    Longstaff-Schwartz agent for multiple optimal stopping (buy-sell problem).
    
    For a buy-sell problem (max_stops=2):
    - stops_left=2: Need to buy (pay price + holding cost)
    - stops_left=1: Need to sell (receive price)
    
    Algorithm:
    1. Generate paths from the environment's transition model
    2. Phase 1: Solve optimal SELL policy (stops_left=1) via backward induction
    3. Phase 2: Solve optimal BUY policy (stops_left=2) using sell values from Phase 1
    """
    
    def __init__(
        self,
        n_states: Optional[int] = None,
        max_time: Optional[int] = None,
        max_stops: Optional[int] = None,
        env=None,
        poly_degree: int = 3,
        holding_cost: float = 0.0,
        seed: Optional[int] = None
    ):
        """
        Initialize LSM agent.
        
        Args:
            n_states: Number of discrete states
            max_time: Maximum time horizon
            max_stops: Number of stops (2 for buy-sell)
            env: Environment to extract parameters from
            poly_degree: Degree of polynomial basis functions
            holding_cost: Cost per time step while holding
            seed: Random seed
        """
        # Extract from env if provided
        if env is not None:
            n_states = env.n_states
            max_time = env.max_time
            max_stops = env.max_stops
            if hasattr(env, 'holding_cost_per_step'):
                holding_cost = env.holding_cost_per_step
            if hasattr(env, 'offer_values'):
                self.offer_values = env.offer_values
            else:
                self.offer_values = np.arange(n_states, dtype=float)
            if hasattr(env, 'P'):
                self.P = env.P  # Transition matrix
            else:
                self.P = None
        else:
            self.offer_values = np.arange(n_states, dtype=float)
            self.P = None
        
        if n_states is None or max_time is None or max_stops is None:
            raise ValueError("Must provide n_states, max_time, max_stops or env")
        
        self.n_states = n_states
        self.max_time = max_time
        self.max_stops = max_stops
        self.poly_degree = poly_degree
        self.holding_cost = holding_cost
        
        self.rng = np.random.default_rng(seed)
        
        # Regression coefficients: {(stops_left, time_left): coefficients}
        self.regression_models: Dict[Tuple[int, int], np.ndarray] = {}
        
        # Stored policy: policy[state, time_left, stops_left] = action
        self.policy = None
        self.trained = False
        
    def _basis_functions(self, states: np.ndarray) -> np.ndarray:
        """
        Compute polynomial basis functions for regression.
        
        Args:
            states: Array of state indices, shape (n_paths,)
            
        Returns:
            Basis matrix, shape (n_paths,n_basis)
        """
        # Convert state indices to values
        values = self.offer_values[states]
        
        # Normalize to [0, 1] for numerical stability
        v_min, v_max = self.offer_values.min(), self.offer_values.max()
        if v_max > v_min:
            values_norm = (values - v_min) / (v_max - v_min)
        else:
            values_norm = values * 0
        
        # Polynomial basis: 1, x, x^2, ..., x^d
        n_paths = len(states)
        basis = np.zeros((n_paths, self.poly_degree + 1))
        for d in range(self.poly_degree + 1):
            basis[:, d] = values_norm ** d
        
        return basis
    
    def _generate_paths(self, n_paths: int, seed: Optional[int] = None) -> np.ndarray:
        """
        Generate state paths using the transition matrix.
        
        Args:
            n_paths: Number of paths to generate
            seed: Random seed
            
        Returns:
            paths: Array of shape (n_paths, max_time + 1) with state indices
                   paths[:, 0] = initial state (time_left = max_time)
                   paths[:, t] = state at time_left = max_time - t
        """
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = self.rng
        
        paths = np.zeros((n_paths, self.max_time + 1), dtype=int)
        
        # Initial state distribution (uniform or from transition matrix)
        paths[:, 0] = rng.integers(0, self.n_states, size=n_paths)
        
        # Generate paths using transition matrix
        if self.P is not None:
            for t in range(self.max_time):
                for i in range(n_paths):
                    current_state = paths[i, t]
                    paths[i, t + 1] = rng.choice(self.n_states, p=self.P[current_state])
        else:
            # Random walk if no transition matrix
            for t in range(self.max_time):
                moves = rng.choice([-1, 0, 1], size=n_paths)
                paths[:, t + 1] = np.clip(paths[:, t] + moves, 0, self.n_states - 1)
        
        return paths
    
    def _train_on_paths(self, paths: np.ndarray, verbose: bool = False) -> float:
        """
        Train the LSM policy on pre-generated paths.
        
        This is the same as train() but uses externally provided paths instead of
        generating new ones. Useful for online regret experiments where we accumulate
        paths over time and periodically retrain.
        
        Args:
            paths: Array of shape (n_paths, max_time+1) with state indices
                   OR (n_paths, max_time) - will be handled appropriately
            verbose: Whether to show progress bars
            
        Returns:
            training_time: Time taken to train
        """
        import time
        start_time = time.time()
        
        if self.max_stops != 2:
            raise NotImplementedError("LSM currently only supports max_stops=2 (buy-sell)")
        
        # Handle both (n_paths, max_time) and (n_paths, max_time+1) shapes
        # Convention: paths[i, t] is state at step t, where time_left = max_time - t
        # So paths should have max_time+1 columns (steps 0 to max_time)
        n_paths = paths.shape[0]
        if paths.shape[1] == self.max_time:
            # Paths have max_time columns (steps 0 to max_time-1)
            # This means time_left goes from max_time down to 1
            # We need to adjust: time_left = max_time - t for t in [0, max_time-1]
            path_length = self.max_time
        elif paths.shape[1] == self.max_time + 1:
            # Paths have max_time+1 columns (steps 0 to max_time)
            # time_left = max_time - t for t in [0, max_time]
            path_length = self.max_time + 1
        else:
            raise ValueError(f"Expected paths shape (n_paths, {self.max_time}) or (n_paths, {self.max_time+1}), got {paths.shape}")
        
        prices = self.offer_values[paths]  # prices[i, t] = price on path i at step t
        
        # Initialize policy array
        self.policy = np.zeros((self.n_states, self.max_time + 1, self.max_stops + 1), dtype=int)
        
        # ============================================================
        # PHASE 1: Solve for sell decision (stops_left=1)
        # ============================================================
        
        if verbose:
            print("Phase 1: Computing optimal SELL policy (stops_left=1)...")
        
        V_sell = np.zeros((n_paths, path_length))
        sell_time = np.full(n_paths, path_length - 1)
        
        # Terminal condition: at last step (t = path_length - 1), time_left = max_time - (path_length - 1)
        # For path_length = max_time: time_left = 1, must sell
        # For path_length = max_time + 1: time_left = 0, terminal (no value)
        last_valid_t = path_length - 1 if path_length == self.max_time else path_length - 2
        V_sell[:, last_valid_t] = prices[:, last_valid_t]
        
        # Policy at time_left=1: always sell
        for s in range(self.n_states):
            self.policy[s, 1, 1] = 1
        
        # Backward induction
        time_range = range(last_valid_t - 1, -1, -1) if not verbose else tqdm(range(last_valid_t - 1, -1, -1), desc="  Sell phase")
        for t in time_range:
            time_left = self.max_time - t  # Correct: at t=0, time_left=max_time
                
            current_states = paths[:, t]
            immediate_sell = prices[:, t]
            future_values = V_sell[:, t + 1].copy()
            continuation = -self.holding_cost + future_values
            
            # Regression
            X = self._basis_functions(current_states)
            try:
                coeffs, _, _, _ = np.linalg.lstsq(X, continuation, rcond=None)
                self.regression_models[(1, time_left)] = coeffs
                continuation_estimate = X @ coeffs
            except np.linalg.LinAlgError:
                self.regression_models[(1, time_left)] = None
                continuation_estimate = continuation
            
            exercise = immediate_sell >= continuation_estimate
            V_sell[:, t] = np.where(exercise, immediate_sell, continuation)
            sell_time = np.where(exercise, t, sell_time)
            
            # Build policy
            for s in range(self.n_states):
                mask = current_states == s
                if np.any(mask):
                    self.policy[s, time_left, 1] = int(np.mean(exercise[mask]) >= 0.5)
        
        # ============================================================
        # PHASE 2: Solve for buy decision (stops_left=2)
        # ============================================================
        
        if verbose:
            print("Phase 2: Computing optimal BUY policy (stops_left=2)...")
        
        V_buy = np.zeros((n_paths, path_length))
        buy_time = np.full(n_paths, -1)
        
        # Terminal conditions
        self.policy[:, 0, 2] = 0
        self.policy[:, 1, 2] = 0
        
        # Backward induction - need time_left >= 2 to complete buy-sell
        time_range = range(last_valid_t - 1, -1, -1) if not verbose else tqdm(range(last_valid_t - 1, -1, -1), desc="  Buy phase")
        for t in time_range:
            time_left = self.max_time - t
            if time_left < 2:
                continue
                
            current_states = paths[:, t]
            buy_now_value = -prices[:, t] - self.holding_cost + V_sell[:, t + 1]
            future_values = V_buy[:, t + 1].copy()
            continuation = future_values
            
            # Regression
            X = self._basis_functions(current_states)
            try:
                coeffs, _, _, _ = np.linalg.lstsq(X, continuation, rcond=None)
                self.regression_models[(2, time_left)] = coeffs
                continuation_estimate = X @ coeffs
            except np.linalg.LinAlgError:
                self.regression_models[(2, time_left)] = None
                continuation_estimate = continuation
            
            exercise = buy_now_value >= continuation_estimate
            V_buy[:, t] = np.where(exercise, buy_now_value, continuation)
            buy_time = np.where(exercise & (buy_time < 0), t, buy_time)
            
            # Build policy
            for s in range(self.n_states):
                mask = current_states == s
                if np.any(mask):
                    self.policy[s, time_left, 2] = int(np.mean(exercise[mask]) >= 0.5)
        
        # Refine policy
        self._refine_policy_from_regression()
        
        self.trained = True
        training_time = time.time() - start_time
        
        return training_time

    def train(self, n_paths: int = 10000, seed: Optional[int] = None, verbose: bool = True) -> float:
        """
        Train the LSM policy using backward induction for buy-sell problem.
        
        For buy-sell (max_stops=2):
        1. Phase 1: Solve optimal SELL policy (stops_left=1) - backward induction
        2. Phase 2: Solve optimal BUY policy (stops_left=2) using sell values
        
        Args:
            n_paths: Number of simulation paths
            seed: Random seed
            verbose: Whether to show progress bars
            
        Returns:
            training_time: Time taken to train
        """
        import time
        start_time = time.time()
        
        if self.max_stops != 2:
            raise NotImplementedError("LSM currently only supports max_stops=2 (buy-sell)")
        
        # Generate paths: paths[i, t] = state at time step t
        # t=0 is the initial state (time_left = max_time)
        # t=max_time is the final state (time_left = 0)
        paths = self._generate_paths(n_paths, seed)
        prices = self.offer_values[paths]  # prices[i, t] = price on path i at step t
        
        # Initialize policy array: policy[state, time_left, stops_left]
        self.policy = np.zeros((self.n_states, self.max_time + 1, self.max_stops + 1), dtype=int)
        
        # ============================================================
        # PHASE 1: Solve for sell decision (stops_left=1)
        # After buying, we hold the asset and need to decide when to sell
        # Value of selling = price
        # Value of continuing = -holding_cost + E[V(next_state, time_left-1, 1)]
        # ============================================================
        
        if verbose:
            print("Phase 1: Computing optimal SELL policy (stops_left=1)...")
        
        # V_sell[i, t] = value of optimal selling strategy on path i starting at time step t
        # given that we're currently holding (have bought but not sold)
        V_sell = np.zeros((n_paths, self.max_time + 1))
        sell_time = np.full(n_paths, self.max_time)  # When we sell on each path
        
        # Terminal condition: at t=max_time (time_left=0), forced to end, value = 0
        # At t=max_time-1 (time_left=1), must sell: V = price
        V_sell[:, self.max_time - 1] = prices[:, self.max_time - 1]
        sell_time[:] = self.max_time - 1  # Initialize: sell at last step
        
        # Policy at time_left=1: always sell
        for s in range(self.n_states):
            self.policy[s, 1, 1] = 1
        
        # Backward induction for time_left >= 2
        time_range = range(self.max_time - 2, -1, -1) if not verbose else tqdm(range(self.max_time - 2, -1, -1), desc="  Sell phase")
        for t in time_range:
            time_left = self.max_time - t
            current_states = paths[:, t]
            
            # Immediate value of selling now
            immediate_sell = prices[:, t]
            
            # Continuation value: -holding_cost + future sell value
            # Future sell value comes from path at t+1 with their optimal decisions
            future_values = V_sell[:, t + 1].copy()
            continuation = -self.holding_cost + future_values
            
            # Regression: fit continuation value as function of current state
            X = self._basis_functions(current_states)
            try:
                coeffs, _, _, _ = np.linalg.lstsq(X, continuation, rcond=None)
                self.regression_models[(1, time_left)] = coeffs
                continuation_estimate = X @ coeffs
            except np.linalg.LinAlgError:
                self.regression_models[(1, time_left)] = None
                continuation_estimate = continuation
            
            # Decision: sell if immediate >= continuation estimate
            exercise = immediate_sell >= continuation_estimate
            
            # Update values and sell times
            V_sell[:, t] = np.where(exercise, immediate_sell, continuation)
            sell_time = np.where(exercise, t, sell_time)
            
            # Build policy for this time_left
            for s in range(self.n_states):
                mask = current_states == s
                if np.any(mask):
                    # Policy = 1 (sell) if majority of paths with this state sell
                    self.policy[s, time_left, 1] = int(np.mean(exercise[mask]) >= 0.5)
        
        # ============================================================
        # PHASE 2: Solve for buy decision (stops_left=2)
        # Before buying, we don't hold anything
        # Value of buying = -price - holding_cost + V_sell(next_state, time_left-1, 1)
        # Value of continuing = E[V(next_state, time_left-1, 2)]
        # ============================================================
        
        if verbose:
            print("Phase 2: Computing optimal BUY policy (stops_left=2)...")
        
        # V_buy[i, t] = value of optimal buy-sell strategy on path i starting at time step t
        V_buy = np.zeros((n_paths, self.max_time + 1))
        buy_time = np.full(n_paths, -1)  # When we buy on each path (-1 = never)
        
        # Terminal conditions:
        # At time_left=0: no time, value = 0
        # At time_left=1: can't complete buy-sell (need 2 steps), value = 0
        V_buy[:, self.max_time] = 0
        V_buy[:, self.max_time - 1] = 0
        self.policy[:, 0, 2] = 0
        self.policy[:, 1, 2] = 0
        
        # Backward induction for time_left >= 2
        time_range = range(self.max_time - 2, -1, -1) if not verbose else tqdm(range(self.max_time - 2, -1, -1), desc="  Buy phase")
        for t in time_range:
            time_left = self.max_time - t
            current_states = paths[:, t]
            
            # Value of buying now:
            # Pay price, pay holding cost this step, then get sell value from next step
            buy_now_value = -prices[:, t] - self.holding_cost + V_sell[:, t + 1]
            
            # Continuation value: wait and decide later (with stops_left=2 still)
            future_values = V_buy[:, t + 1].copy()
            continuation = future_values
            
            # Regression: fit continuation value as function of current state
            X = self._basis_functions(current_states)
            try:
                coeffs, _, _, _ = np.linalg.lstsq(X, continuation, rcond=None)
                self.regression_models[(2, time_left)] = coeffs
                continuation_estimate = X @ coeffs
            except np.linalg.LinAlgError:
                self.regression_models[(2, time_left)] = None
                continuation_estimate = continuation
            
            # Decision: buy if buy_value >= continuation estimate
            exercise = buy_now_value >= continuation_estimate
            
            # Update values
            V_buy[:, t] = np.where(exercise, buy_now_value, continuation)
            buy_time = np.where(exercise & (buy_time < 0), t, buy_time)
            
            # Build policy for this time_left
            for s in range(self.n_states):
                mask = current_states == s
                if np.any(mask):
                    self.policy[s, time_left, 2] = int(np.mean(exercise[mask]) >= 0.5)
        
        # Refine policy using regression coefficients for smoother decisions
        self._refine_policy_from_regression()
        
        self.trained = True
        training_time = time.time() - start_time
        
        if verbose:
            print(f"Training complete in {training_time:.2f}s")
            buy_rate = self.policy[:, 2:, 2].mean()
            sell_rate = self.policy[:, 1:, 1].mean()
            print(f"  Buy policy accept rate: {buy_rate:.2%}")
            print(f"  Sell policy accept rate: {sell_rate:.2%}")
        
        return training_time
    
    def _refine_policy_from_regression(self):
        """
        Refine policy using regression coefficients for deterministic decisions.
        This provides smoother policy boundaries than path-based averaging.
        """
        for stops_left in [1, 2]:
            is_holding = (stops_left == 1)  # Holding if need to sell
            
            for time_left in range(2, self.max_time + 1):
                key = (stops_left, time_left)
                
                if key not in self.regression_models or self.regression_models[key] is None:
                    continue
                
                coeffs = self.regression_models[key]
                
                for state in range(self.n_states):
                    # Compute continuation value estimate
                    basis = self._basis_functions(np.array([state]))
                    continuation = (basis @ coeffs)[0]
                    
                    # Compute immediate value
                    price = self.offer_values[state]
                    
                    if stops_left == 1:  # Sell decision
                        immediate = price  # Value of selling
                    else:  # Buy decision (stops_left == 2)
                        # Value of buying: need to estimate expected sell value
                        # Use regression at (1, time_left-1) for next step sell value
                        sell_key = (1, time_left - 1)
                        if sell_key in self.regression_models and self.regression_models[sell_key] is not None:
                            # Expected sell value given this state transitions
                            expected_sell = 0.0
                            if self.P is not None:
                                for next_state in range(self.n_states):
                                    next_basis = self._basis_functions(np.array([next_state]))
                                    next_sell_cont = (next_basis @ self.regression_models[sell_key])[0]
                                    # At next step, either sell or continue
                                    next_sell_imm = self.offer_values[next_state]
                                    next_sell_val = max(next_sell_imm, next_sell_cont)
                                    expected_sell += self.P[state, next_state] * next_sell_val
                            else:
                                expected_sell = price  # Rough approximation
                            immediate = -price - self.holding_cost + expected_sell
                        else:
                            # Fallback: use simple estimate
                            immediate = -price - self.holding_cost + price  # Break-even estimate
                    
                    # Decision
                    self.policy[state, time_left, stops_left] = int(immediate >= continuation)
    
    def predict(self, state: int, time_left: int, stops_left: int) -> int:
        """
        Predict action for given state.
        
        Args:
            state: Current state index
            time_left: Time remaining
            stops_left: Stops remaining
            
        Returns:
            action: 0 = continue, 1 = stop
        """
        if not self.trained:
            raise ValueError("Agent not trained. Call train() first.")
        
        if time_left <= 0:
            return 1 if stops_left > 0 else 0
        
        if stops_left <= 0:
            return 0
        
        return self.policy[state, time_left, stops_left]
    
    def get_policy(self) -> np.ndarray:
        """
        Get the learned policy array.
        
        Returns:
            policy: Array of shape (n_states, max_time+1, max_stops+1)
        """
        if not self.trained:
            raise ValueError("Agent not trained. Call train() first.")
        
        return self.policy.copy()
    
    def evaluate(
        self,
        env,
        n_episodes: int = 1000,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Evaluate the trained policy on the environment.
        
        Args:
            env: Environment to evaluate on
            n_episodes: Number of episodes
            seed: Random seed
            
        Returns:
            Evaluation results dictionary
        """
        if not self.trained:
            raise ValueError("Agent not trained. Call train() first.")
        
        if seed is not None:
            np.random.seed(seed)
        
        episode_rewards = []
        episode_lengths = []
        
        for _ in range(n_episodes):
            obs, info = env.reset()
            total_reward = 0.0
            steps = 0
            done = False
            
            while not done:
                state = obs['offer']
                time_left = obs['time_left']
                stops_left = obs['stops_left']
                
                action = self.predict(state, time_left, stops_left)
                
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                steps += 1
                done = terminated or truncated
            
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
        
        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'std_error': np.std(episode_rewards) / np.sqrt(n_episodes),
            'episode_rewards': np.array(episode_rewards),
            'episode_lengths': np.array(episode_lengths)
        }
    
    def save(self, filepath: str):
        """Save trained model."""
        np.savez(
            filepath,
            policy=self.policy,
            n_states=self.n_states,
            max_time=self.max_time,
            max_stops=self.max_stops,
            offer_values=self.offer_values,
            poly_degree=self.poly_degree,
            holding_cost=self.holding_cost
        )
    
    def load(self, filepath: str):
        """Load trained model."""
        data = np.load(filepath)
        self.policy = data['policy']
        self.n_states = int(data['n_states'])
        self.max_time = int(data['max_time'])
        self.max_stops = int(data['max_stops'])
        self.offer_values = data['offer_values']
        self.poly_degree = int(data['poly_degree'])
        self.holding_cost = float(data['holding_cost'])
        self.trained = True
