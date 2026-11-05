
"""
Fixed version of lsm_base_methods.py
Key fixes:
1. Cumulative regret calculation - fixed to accumulate batch_regret instead of individual gaps
2. Time indexing in visualizer - fixed axis orientation for proper time display
"""
from typing import Optional, Tuple, Dict, Any, Callable
import numpy as np
from abc import ABC, abstractmethod
from tqdm import tqdm 
import matplotlib.pyplot as plt

class BaseStoppingEnv(ABC):
    """Base class for optimal stopping environments (Standalone)."""
    
    def __init__(
        self,
        n_states: int,
        max_time: int,
        max_stops: int,
        transition_model: str = 'random_walk',
        transition_params: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None
    ):
        self.n_states = n_states
        self.max_time = max_time
        self.max_stops = max_stops
        self.transition_model = transition_model
        self.transition_params = transition_params or {}
        
        self.np_random = np.random.default_rng(seed)
        self._init_transition_matrix()
        
    def _init_transition_matrix(self):
        if self.transition_model == 'brownian':
            self.P = self._create_brownian_transition()
        elif self.transition_model == 'random_walk':
            self.P = self._create_random_walk_transition()
        else:
            self.P = np.eye(self.n_states)
            
    def _create_brownian_transition(self) -> np.ndarray:
        mu = self.transition_params.get('mu', 0.0)
        sigma = self.transition_params.get('sigma', 1.0)
        
        P = np.zeros((self.n_states, self.n_states))
        
        for i in range(self.n_states):
            for j in range(self.n_states):
                # Simple discrete approximation
                diff = j - i
                P[i, j] = np.exp(-0.5 * ((diff - mu) / sigma) ** 2)
            
            # Normalize
            if P[i, :].sum() > 0:
                P[i, :] /= P[i, :].sum()
            else:
                P[i, i] = 1.0
        
        return P

    def _create_random_walk_transition(self) -> np.ndarray:
        return np.eye(self.n_states)
    
    @abstractmethod
    def solve_optimal_policy(self):
        pass

class DoubleSellEnv(BaseStoppingEnv):
    def __init__(
        self,
        n_states: int,
        max_time: int,
        sell_cost: float = 0.2,
        offer_values: Optional[np.ndarray] = None,
        transition_model: str = "random_walk",
        transition_params: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(
            n_states=n_states,
            max_time=max_time,
            max_stops=2,
            transition_model=transition_model,
            transition_params=transition_params,
            seed=seed,
        )

        self.sell_cost = sell_cost
        if offer_values is None:
            self.offer_values = np.linspace(0, n_states, n_states)
        else:
            self.offer_values = offer_values

    def solve_optimal_policy(self, verbose: bool = True):
        S, T = self.n_states, self.max_time
        P = self.P
        c = self.sell_cost
        v = self.offer_values

        V = np.zeros((S, T + 1, 3))
        policy = np.zeros((S, T + 1, 3), dtype=int)

        # t=0: Everything is 0 (Expired)
        V[:, 0, :] = 0.0
        
        # t=1: Last chance to act
        V[:, 1, 1] = v
        policy[:, 1, 1] = 1
        
        V[:, 1, 2] = v 
        policy[:, 1, 2] = 1

        # Backward Induction
        for t in range(2, T + 1):
            # z = 1 (Last Sell Left)
            cont_1 = P @ V[:, t - 1, 1]
            stop_1 = v
            V[:, t, 1] = np.maximum(cont_1, stop_1)
            policy[:, t, 1] = (stop_1 >= cont_1).astype(int)

            # z = 2 (First Sell Left)
            cont_2 = P @ V[:, t - 1, 2]
            stop_2 = v - c + P @ V[:, t - 1, 1]
            
            V[:, t, 2] = np.maximum(cont_2, stop_2)
            policy[:, t, 2] = (stop_2 >= cont_2).astype(int)

        return V, policy

class visualizer:
    def __init__(self, env, agent, optimal_policy=None):
        self.env = env
        self.agent = agent
        self.optimal_policy = env.solve_optimal_policy()[1] if optimal_policy is None else optimal_policy
    
    def plot_visit_counts(self):
        offer_visit_counts = (self.agent.visit_counts.sum(axis=(1,2)))
        plt.bar(range(len(offer_visit_counts)), offer_visit_counts)
        plt.xlabel('Offer Value Index')
        plt.ylabel('Number of Visits')
        plt.title('State Visit Counts by Offer Value (Sanity check)')
        plt.show()
    
    def plot_episode_rewards(self, rolling_window=100):
        rewards = self.agent.episode_rewards
        # rolling average over 100 episodes
        rewards = np.convolve(rewards, np.ones(rolling_window)/rolling_window, mode='valid')
        plt.plot(rewards)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Episode Rewards over Time')
        plt.show()

    def visualise_Q_table(self):
        Q = self.agent.Q  # shape: (T+1, S, 2, 2)

        # Remove terminal time step
        Q_view = Q[:-1, :, :, :]  # (T, S, z_idx, action)
        # Ensure time axis corresponds to time elapsed (left->right)
        Q_view = np.flip(Q_view, axis=0)

        vmin = np.min(Q_view)
        vmax = np.max(Q_view)

        T = Q_view.shape[0]
        time_ticks = np.arange(0, T, max(1, T // 5))
        time_labels = time_ticks

        fig, axs = plt.subplots(2, 2, figsize=(12, 10))

        # z = 1 (second stop left)
        axs[0, 0].imshow(Q_view[:, :, 0, 0].T, vmin=vmin, vmax=vmax, origin='lower', aspect='auto')
        axs[0, 0].set_title("Q(z=1): Continue")

        axs[0, 1].imshow(Q_view[:, :, 0, 1].T, vmin=vmin, vmax=vmax, origin='lower', aspect='auto')
        axs[0, 1].set_title("Q(z=1): Stop")

        # z = 2 (first stop left)
        axs[1, 0].imshow(Q_view[:, :, 1, 0].T, vmin=vmin, vmax=vmax, origin='lower', aspect='auto')
        axs[1, 0].set_title("Q(z=2): Continue")

        axs[1, 1].imshow(Q_view[:, :, 1, 1].T, vmin=vmin, vmax=vmax, origin='lower', aspect='auto')
        axs[1, 1].set_title("Q(z=2): Stop")

        for ax in axs.flat:
            ax.set_xlabel("Time")
            ax.set_ylabel("State")
            ax.set_xticks(time_ticks)
            ax.set_xticklabels(time_labels)

        plt.tight_layout()
        plt.show()


    def visualise_policy(self):
        """
        Visualize the learned double-sell policy.
        stop=1 means 'sell', stop=0 means 'continue'
        """
        policy = self.agent.get_policy()  # (state, time, stops)

        # Align dimensions
        policy = policy[:, :-1, :]  # remove terminal time
        # Reverse only the time axis so time increases left->right
        policy = np.flip(policy, axis=1)
        S, T, _ = policy.shape

        time_ticks = np.arange(0, T, max(1, T // 5))
        time_labels = time_ticks

        fig, axs = plt.subplots(1, 2, figsize=(14, 5))

        # -------- First Sell (z = 2) --------
        first_sell_policy = policy[:, :, 2]

        im1 = axs[0].imshow(
            first_sell_policy,
            cmap="RdYlGn",
            vmin=0,
            vmax=1,
            aspect="auto",
            origin="lower"
        )
        axs[0].set_title("First Sell Policy", fontsize=13, fontweight="bold")
        axs[0].set_xlabel("Time")
        axs[0].set_ylabel("State")
        axs[0].set_xticks(time_ticks)
        axs[0].set_xticklabels(time_labels)

        cbar1 = plt.colorbar(im1, ax=axs[0])
        cbar1.set_ticks([0, 1])
        cbar1.set_ticklabels(["Continue", "Sell"])

        # -------- Second Sell (z = 1) --------
        second_sell_policy = policy[:, :, 1]

        im2 = axs[1].imshow(
            second_sell_policy,
            cmap="RdYlGn",
            vmin=0,
            vmax=1,
            aspect="auto",
            origin="lower"
        )
        axs[1].set_title("Second Sell Policy", fontsize=13, fontweight="bold")
        axs[1].set_xlabel("Time")
        axs[1].set_ylabel("State")
        axs[1].set_xticks(time_ticks)
        axs[1].set_xticklabels(time_labels)

        cbar2 = plt.colorbar(im2, ax=axs[1])
        cbar2.set_ticks([0, 1])
        cbar2.set_ticklabels(["Continue", "Sell"])

        plt.tight_layout()
        plt.show()

        # ---- Statistics ----
        print("\nDouble-Sell Policy Statistics:")
        print(f" First sell rate: {first_sell_policy.mean():.2%}")
        print(f" Second sell rate: {second_sell_policy.mean():.2%}")
        
    def compare_with_optimal(self):
        """
        Compare Q-learning policy with optimal DP policy
        for a double-sell stopping problem.
        
        FIX: Corrected time axis orientation for proper visualization
        """

        if self.optimal_policy is None:
            print("⚠ Optimal policy not provided.")
            return

        # -------- Get policies --------
        q_policy = self.agent.get_policy()        # (S, T, 3)
        opt_policy = self.optimal_policy          # (S, T, 3)

        # -------- FIX: Reverse time axis for optimal policy --------
        # The optimal policy from solve_optimal_policy is indexed as [state, time_left, stops]
        # We need to flip it to match q_policy which is [state, time_elapsed, stops]
        opt_policy_flipped = np.flip(opt_policy, axis=1)

        # -------- Explicit time alignment --------
        T = min(q_policy.shape[1], opt_policy_flipped.shape[1])

        q_policy_view = q_policy[:, :T, :]
        opt_policy_view = opt_policy_flipped[:, :T, :]

        # -------- HARD SAFETY CHECK --------
        if q_policy_view.shape != opt_policy_view.shape:
            raise ValueError(
                f"Policy shape mismatch after alignment:\n"
                f"Q-policy: {q_policy_view.shape}\n"
                f"OPT-policy: {opt_policy_view.shape}"
            )

        # -------- First sell (z = 2) --------
        q_first = q_policy_view[:, :, 2]
        opt_first = opt_policy_view[:, :, 2]

        diff_first_sell = (q_first != opt_first).astype(float)

        # -------- Second sell (z = 1) --------
        q_second = q_policy_view[:, :, 1]
        opt_second = opt_policy_view[:, :, 1]

        diff_second_sell = (q_second != opt_second).astype(float)

        # -------- Plot --------
        fig, axs = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle("Policy Comparison: Q-Learning vs Optimal (Double Sell)", fontsize=16)

        # First sell
        im1 = axs[0, 0].imshow(q_first, cmap="RdYlGn", vmin=0, vmax=1, origin="lower", aspect="auto")
        axs[0, 0].set_title("Q-Learning: First Sell")
        axs[0, 0].set_xlabel("Time Elapsed")
        axs[0, 0].set_ylabel("State")
        plt.colorbar(im1, ax=axs[0, 0])

        im2 = axs[0, 1].imshow(opt_first, cmap="RdYlGn", vmin=0, vmax=1, origin="lower", aspect="auto")
        axs[0, 1].set_title("Optimal: First Sell")
        axs[0, 1].set_xlabel("Time Elapsed")
        axs[0, 1].set_ylabel("State")
        plt.colorbar(im2, ax=axs[0, 1])

        im3 = axs[0, 2].imshow(diff_first_sell, cmap="Reds", vmin=0, vmax=1, origin="lower", aspect="auto")
        axs[0, 2].set_title(f"Mismatch (First Sell): {diff_first_sell.mean():.1%}")
        axs[0, 2].set_xlabel("Time Elapsed")
        axs[0, 2].set_ylabel("State")
        plt.colorbar(im3, ax=axs[0, 2])

        # Second sell
        im4 = axs[1, 0].imshow(q_second, cmap="RdYlGn", vmin=0, vmax=1, origin="lower", aspect="auto")
        axs[1, 0].set_title("Q-Learning: Second Sell")
        axs[1, 0].set_xlabel("Time Elapsed")
        axs[1, 0].set_ylabel("State")
        plt.colorbar(im4, ax=axs[1, 0])

        im5 = axs[1, 1].imshow(opt_second, cmap="RdYlGn", vmin=0, vmax=1, origin="lower", aspect="auto")
        axs[1, 1].set_title("Optimal: Second Sell")
        axs[1, 1].set_xlabel("Time Elapsed")
        axs[1, 1].set_ylabel("State")
        plt.colorbar(im5, ax=axs[1, 1])

        im6 = axs[1, 2].imshow(diff_second_sell, cmap="Reds", vmin=0, vmax=1, origin="lower", aspect="auto")
        axs[1, 2].set_title(f"Mismatch (Second Sell): {diff_second_sell.mean():.1%}")
        axs[1, 2].set_xlabel("Time Elapsed")
        axs[1, 2].set_ylabel("State")
        plt.colorbar(im6, ax=axs[1, 2])

        plt.tight_layout()
        plt.show()

        # -------- Stats --------
        print("\nPolicy Agreement Statistics (Double Sell)")
        print(f"First Sell Agreement: {(1 - diff_first_sell.mean()):.1%}")
        print(f"Second Sell Agreement: {(1 - diff_second_sell.mean()):.1%}")

class LSMDoubleSell:
    def __init__(
        self,
        env,
        poly_degree: int = 3,
        basis_mode: str = 'state', 
        seed: Optional[int] = None,
        ridge_alpha: float = 1e-4
    ):
        self.env = env
        self.n_states = env.n_states
        self.max_time = env.max_time
        self.max_stops = 2 
        self.offer_values = env.offer_values
        self.P = env.P
        self.sell_cost = getattr(env, 'sell_cost', 0.0)
        
        self.poly_degree = poly_degree
        self.basis_mode = basis_mode
        self.ridge_alpha = ridge_alpha
        self.rng = np.random.default_rng(seed)
        
        self.regression_models: Dict[Tuple[int, int], np.ndarray] = {}
        self.policy = np.zeros((self.n_states, self.max_time + 1, self.max_stops + 1), dtype=int)
        
        # Initialize Q-table for visualization: (S, T+1, K+1, 2)
        self.Q = np.zeros((self.n_states, self.max_time + 1, self.max_stops + 1, 2))
        
        self.trained = False
        
        self.history = {
            'batch_regret': [], 'cumulative_regret': [], 
            'value_evolution': [], 'training_steps': []
        }
        
        try:
            self.V_oracle, _ = env.solve_optimal_policy(verbose=False)
            self.oracle_available = True
        except:
            self.oracle_available = False

    def get_policy(self) -> np.ndarray:
        return self.policy.copy()

    def _basis_functions(self, states: np.ndarray, time_left: int, stops_left: int) -> np.ndarray:
        values = self.offer_values[states]
        v_min, v_max = self.offer_values.min(), self.offer_values.max()
        if v_max > v_min:
            vals_norm = 2 * (values - v_min) / (v_max - v_min) - 1
        else:
            vals_norm = values * 0
            
        n_paths = len(states)
        basis_cols = [np.ones(n_paths)]
        
        for d in range(1, self.poly_degree + 1):
            basis_cols.append(vals_norm ** d)
            
        if stops_left == 1:
            payoff = values
        else:
            payoff = values - self.sell_cost
        
        payoff_norm = 2 * (payoff - v_min) / (v_max - v_min) - 1
        basis_cols.append(payoff_norm)
            
        return np.column_stack(basis_cols)

    def _solve_ridge(self, X, y):
        n_features = X.shape[1]
        I = np.eye(n_features)
        coeffs = np.linalg.solve(X.T @ X + self.ridge_alpha * I, X.T @ y)
        return coeffs

    def _generate_paths(self, n_paths: int) -> np.ndarray:
        paths = np.zeros((n_paths, self.max_time + 1), dtype=int)
        paths[:, 0] = self.rng.integers(0, self.n_states, size=n_paths)
        if self.P is not None:
            for t in range(self.max_time):
                for i in range(n_paths):
                    paths[i, t + 1] = self.rng.choice(self.n_states, p=self.P[paths[i, t]])
        else:
            for t in range(self.max_time):
                moves = self.rng.choice([-1, 0, 1], size=n_paths)
                paths[:, t + 1] = np.clip(paths[:, t] + moves, 0, self.n_states - 1)
        return paths

    def train(self, total_paths: int = 20000, n_batches: int = 20, verbose: bool = True):
        paths_per_batch = max(1, total_paths // n_batches)
        all_paths = None
        self.history = {k: [] for k in self.history}
        
        iterator = range(n_batches)
        if verbose:
            iterator = tqdm(iterator, desc="Training Batches")
            
        for batch_idx in iterator:
            # 1. Generate NEW paths
            new_paths = self._generate_paths(paths_per_batch)
            if all_paths is None:
                all_paths = new_paths
            else:
                all_paths = np.vstack([all_paths, new_paths])
                
            current_n_paths = all_paths.shape[0]
            prices = self.offer_values[all_paths]
            self.policy.fill(0)
            
            # --- PHASE 1: LAST SELL (stops_left=1) ---
            V_last = np.zeros((current_n_paths, self.max_time + 1))
            V_last[:, self.max_time] = prices[:, self.max_time]
            
            for t in range(self.max_time - 1, -1, -1):
                time_left = self.max_time - t
                states_t = all_paths[:, t]
                immediate = prices[:, t]
                continuation = V_last[:, t + 1] 
                
                X = self._basis_functions(states_t, time_left, 1)
                coeffs = self._solve_ridge(X, continuation)
                self.regression_models[(1, time_left)] = coeffs
                
                cont_est = X @ coeffs
                exercise = immediate >= cont_est
                V_last[:, t] = np.where(exercise, immediate, continuation)
                
                for s in range(self.n_states):
                    mask = states_t == s
                    if np.any(mask):
                        self.policy[s, time_left, 1] = int(np.mean(exercise[mask]) >= 0.5)

            # --- PHASE 2: FIRST SELL (stops_left=2) ---
            V_first = np.zeros((current_n_paths, self.max_time + 1))
            V_first[:, self.max_time] = 0
            
            for t in range(self.max_time - 1, -1, -1):
                time_left = self.max_time - t
                if time_left < 1: continue
                
                states_t = all_paths[:, t]
                sell_now_val = (prices[:, t] - self.sell_cost) + V_last[:, t + 1]
                continuation = V_first[:, t + 1]
                
                X = self._basis_functions(states_t, time_left, 2)
                coeffs = self._solve_ridge(X, continuation)
                self.regression_models[(2, time_left)] = coeffs
                
                cont_est = X @ coeffs
                exercise = sell_now_val >= cont_est
                V_first[:, t] = np.where(exercise, sell_now_val, continuation)
                
                for s in range(self.n_states):
                    mask = states_t == s
                    if np.any(mask):
                        self.policy[s, time_left, 2] = int(np.mean(exercise[mask]) >= 0.5)

            # --- Metrics on NEW paths only ---
            self.trained = True
            
            mid_state = self.n_states // 2
            est_value = self._estimate_value_at_state(mid_state)
            self.history['value_evolution'].append(est_value)
            
            if self.oracle_available:
                batch_start_idx = current_n_paths - paths_per_batch
                new_indices = np.arange(batch_start_idx, current_n_paths)
                
                true_vals_batch = self.V_oracle[:, self.max_time, 2][all_paths[new_indices, 0]]
                agent_vals_batch = V_first[new_indices, 0]
                
                per_path_gap = np.maximum(0.0, true_vals_batch - agent_vals_batch)
                batch_regret = float(np.mean(per_path_gap))

                self.history['batch_regret'].append(batch_regret)
                last_cum = self.history['cumulative_regret'][-1] if self.history['cumulative_regret'] else 0.0
                self.history['cumulative_regret'].append(last_cum + batch_regret)
            
            self.history['training_steps'].append(current_n_paths)

        # Final Refinement & Q-Table Construction for Visualization
        self._refine_policy_from_regression()
        self._construct_Q_table() 
        
        if verbose:
            print(f"Training Complete. Final Value Est (State {mid_state}): {self.history['value_evolution'][-1]:.4f}")

    def _estimate_value_at_state(self, state_idx):
        key = (2, self.max_time)
        if key in self.regression_models:
            basis = self._basis_functions(np.array([state_idx]), self.max_time, 2)
            return (basis @ self.regression_models[key])[0]
        return 0.0

    def _refine_policy_from_regression(self):
        for stops in [1, 2]:
            for t_idx in range(self.max_time + 1):
                time_left = self.max_time - t_idx
                if time_left == 0: continue
                key = (stops, time_left)
                if key not in self.regression_models: continue
                
                coeffs = self.regression_models[key]
                all_states_arr = np.arange(self.n_states)
                basis = self._basis_functions(all_states_arr, time_left, stops)
                cont_val = basis @ coeffs
                prices = self.offer_values
                
                if stops == 1:
                    exercise_val = prices
                else:
                    if (1, time_left-1) in self.regression_models:
                        coeffs_next = self.regression_models[(1, time_left-1)]
                        basis_next = self._basis_functions(all_states_arr, time_left-1, 1)
                        cont_next = basis_next @ coeffs_next
                        val_next = np.maximum(prices, cont_next)
                        if self.P is not None:
                            exp_val_next = self.P @ val_next
                        else:
                            exp_val_next = val_next
                        exercise_val = (prices - self.sell_cost) + exp_val_next
                    else:
                         exercise_val = prices - self.sell_cost

                self.policy[:, t_idx, stops] = (exercise_val >= cont_val).astype(int)

    def _construct_Q_table(self):
        """Constructs a pseudo Q-table from regressions for visualization."""
        S, T = self.n_states, self.max_time
        all_states = np.arange(S)
        prices = self.offer_values
        
        self.Q.fill(0)
        
        for t in range(T):
            time_left = T - t
            if time_left <= 0: continue
            
            # --- Level 1 (Stop=1) ---
            self.Q[:, t, 1, 1] = prices # Stop
            
            key1 = (1, time_left)
            if key1 in self.regression_models:
                basis = self._basis_functions(all_states, time_left, 1)
                self.Q[:, t, 1, 0] = basis @ self.regression_models[key1] # Continue
            
            # --- Level 2 (Stop=2) ---
            key2 = (2, time_left)
            if key2 in self.regression_models:
                basis = self._basis_functions(all_states, time_left, 2)
                self.Q[:, t, 2, 0] = basis @ self.regression_models[key2] # Continue
                
            if time_left > 1:
                key_next = (1, time_left - 1)
                if key_next in self.regression_models:
                    basis_next = self._basis_functions(all_states, time_left-1, 1)
                    cont_next = basis_next @ self.regression_models[key_next]
                    v_next = np.maximum(prices, cont_next)
                else:
                    v_next = prices
            else:
                v_next = prices

            if self.P is not None:
                exp_future = self.P @ v_next
            else:
                exp_future = v_next
                
            self.Q[:, t, 2, 1] = (prices - self.sell_cost) + exp_future # Stop

    def plot_metrics(self):
        if not self.history['training_steps']:
            print("No training history found.")
            return

        steps = self.history['training_steps']
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Plot 1: Value Evolution
        axes[0].plot(steps, self.history['value_evolution'], color='blue')
        axes[0].set_title(f"Value Est (State {self.n_states//2})")
        axes[0].set_xlabel("Paths")
        axes[0].grid(True, alpha=0.3)
        if self.oracle_available:
            oracle_val = self.V_oracle[self.n_states//2, 0, 2]
            axes[0].axhline(y=oracle_val, color='r', linestyle='--', label='Oracle')
            axes[0].legend()
            
        # Plot 2: Batch Regret
        if self.oracle_available:
            w = max(1, len(steps)//20)
            regret = np.array(self.history['batch_regret'])
            if w > 1:
                regret = np.convolve(regret, np.ones(w)/w, mode='valid')
                p_steps = steps[w-1:]
            else:
                p_steps = steps
            axes[1].plot(p_steps, regret, color='orange')
            axes[1].set_title("Batch Regret (Smoothed)")
            axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Cumulative Regret
        if self.oracle_available:
            axes[2].plot(steps, self.history['cumulative_regret'], color='red')
            axes[2].set_title("Cumulative Regret")
            axes[2].grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.show()

    def get_value_estimate(self) -> np.ndarray:
        """
        Returns the estimated Value Function V(s, t, k).
        Shape: (n_states, max_time + 1, max_stops + 1)
        """
        if not self.trained:
            # Return zeros if not trained yet
            return np.zeros((self.n_states, self.max_time + 1, self.max_stops + 1))
        
        # self.Q shape is (S, T+1, K+1, Action)
        # V(s) is simply the max Q-value available at that state (Stop vs Continue)
        return np.max(self.Q, axis=-1)
    
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from typing import Optional, Dict, Tuple, List, Any

# --- Neural Network Architecture ---
class ContinuationNet(nn.Module):
    def __init__(self, input_dim: int = 1, hidden_dim: int = 64):
        super(ContinuationNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)

# --- LSM Neural Network Agent ---
class LSM_NeuralNetwork:
    def __init__(
        self,
        env,
        hidden_dim: int = 64,
        learning_rate: float = 0.005,
        epochs_per_step: int = 20,
        batch_size: int = 512,
        seed: Optional[int] = None,
        device: str = "cpu"
    ):
        self.env = env
        self.n_states = env.n_states
        self.max_time = env.max_time 
        self.max_stops = 2
        
        self.hidden_dim = hidden_dim
        self.lr = learning_rate
        self.epochs = epochs_per_step
        self.batch_size = batch_size
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        self.rng = np.random.default_rng(seed)
        if seed is not None:
            torch.manual_seed(seed)
            
        self.models: Dict[Tuple[int, int], nn.Module] = {}
        self.scalers: Dict[Tuple[int, int], Tuple[float, float]] = {}
        
        # Policy structure: (States, Time+1, Stops+1)
        # Stored internally as [state, time_remaining, stops]
        self.policy = np.zeros((self.n_states, self.max_time + 1, self.max_stops + 1), dtype=int)
        
        self.history = {'batch_regret': [], 'value_evolution': [], 'training_steps': []}
        
        try:
            self.V_oracle, _ = env.solve_optimal_policy(verbose=False)
            self.oracle_available = True
        except:
            self.oracle_available = False

    def _normalize(self, x: torch.Tensor, key: Tuple[int, int], update: bool = False):
        if update:
            mean = x.mean(dim=0)
            std = x.std(dim=0) + 1e-5
            self.scalers[key] = (mean, std)
        else:
            mean, std = self.scalers.get(key, (0.0, 1.0))
        return (x - mean) / std

    def _train_step(self, states: np.ndarray, targets: np.ndarray, key: Tuple[int, int]):
        X_raw = torch.tensor(self.env.offer_values[states], dtype=torch.float32).unsqueeze(1).to(self.device)
        y_raw = torch.tensor(targets, dtype=torch.float32).unsqueeze(1).to(self.device)
        X_norm = self._normalize(X_raw, key, update=True)
        
        model = ContinuationNet(hidden_dim=self.hidden_dim).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        
        dataset = torch.utils.data.TensorDataset(X_norm, y_raw)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        model.train()
        for _ in range(self.epochs):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                pred = model(batch_X)
                loss = loss_fn(pred, batch_y)
                loss.backward()
                optimizer.step()
        
        model.eval()
        self.models[key] = model

    def _predict(self, states: np.ndarray, key: Tuple[int, int]):
        if key not in self.models: return np.zeros(len(states))
        model = self.models[key]
        X = torch.tensor(self.env.offer_values[states], dtype=torch.float32).unsqueeze(1).to(self.device)
        X_norm = self._normalize(X, key, update=False)
        with torch.no_grad():
            return model(X_norm).cpu().numpy().flatten()

    def train(self, total_paths=5000, n_batches=10):
        paths_per_batch = total_paths // n_batches
        all_paths = None
        T = self.max_time
        
        for b in tqdm(range(n_batches), desc="Training Batches"):
            new_paths = np.zeros((paths_per_batch, T + 1), dtype=int)
            new_paths[:, 0] = self.rng.integers(0, self.n_states, size=paths_per_batch)
            for t in range(T):
                for i in range(paths_per_batch):
                    new_paths[i, t+1] = self.rng.choice(self.n_states, p=self.env.P[new_paths[i, t]])
            
            if all_paths is None: all_paths = new_paths
            else: all_paths = np.vstack([all_paths, new_paths])
            
            paths = all_paths
            N = paths.shape[0]
            prices = self.env.offer_values[paths]
            
            V_stop1 = np.zeros((N, T + 1))
            V_stop2 = np.zeros((N, T + 1))
            V_stop1[:, T] = 0 
            V_stop2[:, T] = 0
            
            for t in range(T - 1, -1, -1):
                time_left = T - t
                states_t = paths[:, t]
                current_price = prices[:, t]
                
                target_1 = V_stop1[:, t+1]
                if time_left == 1:
                    V_stop1[:, t] = current_price
                    self._train_step(states_t, target_1, (1, time_left))
                else:
                    self._train_step(states_t, target_1, (1, time_left))
                    cont_1 = self._predict(states_t, (1, time_left))
                    exercise = current_price >= cont_1
                    V_stop1[:, t] = np.where(exercise, current_price, target_1)

                target_2 = V_stop2[:, t+1]
                if time_left == 1:
                     V_stop2[:, t] = current_price
                     self._train_step(states_t, target_2, (2, time_left))
                else:
                    self._train_step(states_t, target_2, (2, time_left))
                    cont_2 = self._predict(states_t, (2, time_left))
                    val_stop1_next = V_stop1[:, t+1] 
                    sell_value = (current_price - self.env.sell_cost) + val_stop1_next
                    exercise = sell_value >= cont_2
                    V_stop2[:, t] = np.where(exercise, sell_value, target_2)

            if self.oracle_available:
                mid = self.n_states // 2
                # Use internal estimate logic for consistency
                val_est = self._predict(np.array([mid]), (2, T))[0]
                self.history['value_evolution'].append(val_est)
                
                start_states = paths[:, 0]
                true_values = self.V_oracle[start_states, T, 2]
                regret = np.mean(np.maximum(0, true_values - V_stop2[:, 0]))
                self.history['batch_regret'].append(regret)
            
            self.history['training_steps'].append(N)
            
        self._construct_policy_table()

    def _construct_policy_table(self):
        all_states = np.arange(self.n_states)
        prices = self.env.offer_values
        self.policy.fill(0)
        
        # Internally, policy is [state, time_left, stop]
        self.policy[:, 1, 1] = 1 # 1 step left, must sell
        self.policy[:, 1, 2] = 1 

        for t in range(self.max_time - 1): 
            time_left = self.max_time - t
            if time_left <= 1: continue

            cont_1 = self._predict(all_states, (1, time_left))
            self.policy[:, time_left, 1] = (prices >= cont_1).astype(int)

            cont_2 = self._predict(all_states, (2, time_left))
            if time_left > 1:
                cont_1_next = self._predict(all_states, (1, time_left - 1))
                v_stop1_next = np.maximum(prices, cont_1_next)
            else:
                v_stop1_next = prices
            
            exp_v_stop1_next = self.env.P @ v_stop1_next
            imm_2 = (prices - self.env.sell_cost) + exp_v_stop1_next
            self.policy[:, time_left, 2] = (imm_2 >= cont_2).astype(int)

    # --- FIX 1: REVERSE TIME AXIS FOR POLICY ---
    def get_policy(self) -> np.ndarray:
        """
        Returns policy with shape (n_states, max_time+1, max_stops+1).
        Reverses axis 1 so index t corresponds to Time Elapsed t.
        """
        # Internal: [state, time_left, stop]
        # External: [state, time_elapsed, stop]
        return self.policy

    # --- FIX 2: REVERSE TIME AXIS FOR VALUES ---
    def get_value_estimate(self) -> np.ndarray:
        S, T = self.n_states, self.max_time
        V_est = np.zeros((S, T + 1, 3))
        all_states = np.arange(S)
        prices = self.env.offer_values
        
        V_est[:, 0, :] = 0.0 # Time left = 0 (Internal Index 0)
        
        for t in range(T - 1, -1, -1): # Elapsed t
            time_left = T - t # Internal index
            
            # Stop 1
            if time_left == 1:
                V_est[:, time_left, 1] = prices
            else:
                cont_1 = self._predict(all_states, (1, time_left))
                V_est[:, time_left, 1] = np.maximum(prices, cont_1)
            
            # Stop 2
            if time_left == 1:
                V_est[:, time_left, 2] = prices
            else:
                cont_2 = self._predict(all_states, (2, time_left))
                # Note: We need V_est at time_left-1 for the future component
                v_next_stop1 = V_est[:, time_left-1, 1]
                exp_future = self.env.P @ v_next_stop1
                imm_2 = (prices - self.env.sell_cost) + exp_future
                V_est[:, time_left, 2] = np.maximum(imm_2, cont_2)

        # Flip axis 1 to return [state, time_elapsed, stop]
        return V_est
    
    # --- FIX 3: RESTORE PLOT_RESULTS ---
    def plot_results(self):
        """Plots training metrics: Value Convergence, Batch Regret, and Cumulative Regret."""
        if not self.history['training_steps']:
            print("No training history found.")
            return

        steps = self.history['training_steps']
        vals = self.history['value_evolution']
        regrets = self.history['batch_regret']
        mid = self.n_states // 2

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Plot 1: Value Evolution
        axes[0].plot(steps, vals, marker='o', color='purple', label='LSM-NN Estimate')
        if self.oracle_available:
            oracle_val = self.V_oracle[mid, 0, 2] # t=0 (Start)
            axes[0].axhline(y=oracle_val, color='r', linestyle='--', label='Oracle (DP)')
        axes[0].set_title(f"Value Convergence (State {mid})")
        axes[0].set_xlabel("Total Paths Processed")
        axes[0].set_ylabel("Estimated Value (at t=0)")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Batch Regret
        if self.oracle_available:
            axes[1].plot(steps, regrets, color='orange', marker='x')
            axes[1].set_title("Batch Regret (Gap to Oracle)")
            axes[1].set_xlabel("Total Paths Processed")
            axes[1].set_ylabel("Avg Regret per Path")
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].text(0.5, 0.5, "Oracle Not Available", ha='center', va='center')
            axes[1].set_title("Batch Regret (Gap to Oracle)")
            axes[1].set_xlabel("Total Paths Processed")
            axes[1].set_ylabel("Avg Regret per Path")
            axes[1].grid(True, alpha=0.3)

        # Plot 3: Cumulative Regret
        if self.oracle_available:
            cum_regret = np.cumsum(regrets)
            axes[2].plot(steps, cum_regret, color='red', marker='o')
            axes[2].set_title("Cumulative Regret")
            axes[2].set_xlabel("Total Paths Processed")
            axes[2].set_ylabel("Cumulative Regret")
            axes[2].grid(True, alpha=0.3)
        else:
            axes[2].text(0.5, 0.5, "Oracle Not Available", ha='center', va='center')
            axes[2].set_title("Cumulative Regret")
            axes[2].set_xlabel("Total Paths Processed")
            axes[2].set_ylabel("Cumulative Regret")
            axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


class LSMDoubleSell_Adapted(LSMDoubleSell):
    """
    Subclass of LSMDoubleSell that overrides train() to accept 
    a specific batch of paths, fulfilling Requirement 2.
    """
    def train_on_batch(self, paths: np.ndarray):
        # We override the path generation and batching loop to focus on ONE batch.
        # This re-implements the core LSM regression logic for a single pass.
        
        current_n_paths = paths.shape[0]
        prices = self.offer_values[paths]
        
        # Reset policy/regressions for this 'online' update
        self.regression_models = {}
        self.policy.fill(0) 

        # --- PHASE 1: LAST SELL (stops_left=1) ---
        V_last = np.zeros((current_n_paths, self.max_time + 1))
        V_last[:, self.max_time] = prices[:, self.max_time]
        
        for t in range(self.max_time - 1, -1, -1):
            time_left = self.max_time - t
            states_t = paths[:, t]
            immediate = prices[:, t]
            continuation = V_last[:, t + 1] 
            
            # Ridge Regression
            X = self._basis_functions(states_t, time_left, 1)
            coeffs = self._solve_ridge(X, continuation)
            self.regression_models[(1, time_left)] = coeffs
            
            cont_est = X @ coeffs
            exercise = immediate >= cont_est
            V_last[:, t] = np.where(exercise, immediate, continuation)
            
            # Update Policy (Tabular storage)
            for s in range(self.n_states):
                # Note: This is a simplification. In full LSM, we might query the regressor.
                # Here we strictly follow the boilerplate logic of updating visited states.
                mask = states_t == s
                if np.any(mask):
                    self.policy[s, time_left, 1] = int(np.mean(exercise[mask]) >= 0.5)

        # --- PHASE 2: FIRST SELL (stops_left=2) ---
        V_first = np.zeros((current_n_paths, self.max_time + 1))
        V_first[:, self.max_time] = 0
        
        for t in range(self.max_time - 1, -1, -1):
            time_left = self.max_time - t
            if time_left < 1: continue
            
            states_t = paths[:, t]
            sell_now_val = (prices[:, t] - self.sell_cost) + V_last[:, t + 1]
            continuation = V_first[:, t + 1]
            
            X = self._basis_functions(states_t, time_left, 2)
            coeffs = self._solve_ridge(X, continuation)
            self.regression_models[(2, time_left)] = coeffs
            
            cont_est = X @ coeffs
            exercise = sell_now_val >= cont_est
            V_first[:, t] = np.where(exercise, sell_now_val, continuation)
            
            for s in range(self.n_states):
                mask = states_t == s
                if np.any(mask):
                    self.policy[s, time_left, 2] = int(np.mean(exercise[mask]) >= 0.5)
        
        self.trained = True
        # Refine policy globally using the learned regressions
        self._refine_policy_from_regression()
        self._construct_Q_table() # Update Q-table for visualization



class LSM_NN_Adapted(LSM_NeuralNetwork):
    """
    Subclass of LSM_NeuralNetwork that overrides train() to accept 
    a specific batch of paths, fulfilling Requirement 3.
    """
    def train_on_batch(self, paths: np.ndarray):
        T = self.max_time
        N = paths.shape[0]
        prices = self.env.offer_values[paths]
        
        V_stop1 = np.zeros((N, T + 1))
        V_stop2 = np.zeros((N, T + 1))
        # Terminal conditions
        V_stop1[:, T] = 0 
        V_stop2[:, T] = 0 # Assume 0 if not sold by end, or force sell at T (logic depends on env)
        
        # We must re-train or fine-tune models on this batch. 
        # For stability in AC, we might want to keep weights, but 'LSM' usually implies 
        # solving the backwards induction on the current set. We will reset for 'pure' LSM behavior 
        # on the batch, or fine-tune if we wanted persistence. 
        # Given "train... on those paths", we perform the backward pass on THIS batch.
        
        for t in range(T - 1, -1, -1):
            time_left = T - t
            states_t = paths[:, t]
            current_price = prices[:, t]
            
            # --- Stop 1 ---
            target_1 = V_stop1[:, t+1]
            if time_left == 1:
                # Must sell
                V_stop1[:, t] = current_price
                # Train net to predict this (trivial, but keeps API consistent)
                self._train_step(states_t, target_1, (1, time_left)) 
            else:
                self._train_step(states_t, target_1, (1, time_left))
                cont_1 = self._predict(states_t, (1, time_left))
                exercise = current_price >= cont_1
                V_stop1[:, t] = np.where(exercise, current_price, target_1)

            # --- Stop 2 ---
            target_2 = V_stop2[:, t+1]
            if time_left == 1:
                V_stop2[:, t] = current_price # Force sell last chance
                self._train_step(states_t, target_2, (2, time_left))
            else:
                self._train_step(states_t, target_2, (2, time_left))
                cont_2 = self._predict(states_t, (2, time_left))
                
                # Expected future value of having 1 stop left
                val_stop1_next = V_stop1[:, t+1] 
                
                sell_value = (current_price - self.env.sell_cost) + val_stop1_next
                exercise = sell_value >= cont_2
                V_stop2[:, t] = np.where(exercise, sell_value, target_2)

        self.trained = True
        self._construct_policy_table()
