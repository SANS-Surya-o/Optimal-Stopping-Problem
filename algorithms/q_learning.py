"""
Modular and extensible Q-Learning implementation for optimal stopping problems.
Supports tunable and time-varying hyperparameters.
"""
from typing import Optional, Dict, Any, Callable, List, Tuple, Union
import numpy as np
from collections import defaultdict
import json
from pathlib import Path
from tqdm.auto import tqdm


class QLearningAgent:
    """
    Q-Learning agent with support for:
    - Time-varying epsilon (exploration rate)
    - Time-varying alpha (learning rate)
    - Customizable discount factor (gamma)
    - Training callbacks and monitoring
    Q(offer, time_left, stops_left, action)
    """
    
    def __init__(
        self,
        n_states: Optional[int] = None,
        n_actions: Optional[int] = None,
        max_time: Optional[int] = None,
        max_stops: Optional[int] = None,
        env = None,
        Q_init: Optional[np.ndarray] = None,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.1,
        alpha_schedule: Optional[Callable[[int], float]] = None,
        epsilon_schedule: Optional[Callable[[int], float]] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize Q-Learning agent.
        
        Args:
            n_states: Number of offer states (extracted from env if not provided)
            n_actions: Number of actions (extracted from env if not provided)
            max_time: Maximum time horizon (extracted from env if not provided)
            max_stops: Maximum number of stops (extracted from env if not provided)
            env: Environment instance to extract parameters from (if other params not provided)
            alpha: Initial learning rate
            gamma: Discount factor
            epsilon: Initial exploration rate
            alpha_schedule: Function(episode) -> alpha for time-varying learning rate
            epsilon_schedule: Function(episode) -> epsilon for time-varying exploration
            seed: Random seed
        """
        # Extract parameters from environment if not provided
        if env is not None:
            if n_states is None:
                n_states = env.n_states
            if n_actions is None:
                n_actions = env.action_space.n
            if max_time is None:
                max_time = env.max_time
            if max_stops is None:
                max_stops = env.max_stops

       

        # Validate that all required parameters are available
        if n_states is None or n_actions is None or max_time is None or max_stops is None:
            raise ValueError(
                "Either provide all of (n_states, n_actions, max_time, max_stops) "
                "or provide an env to extract these parameters from."
            )
        
        self.n_states = n_states
        self.n_actions = n_actions
        self.max_time = max_time
        self.max_stops = max_stops
        
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        self.alpha_schedule = alpha_schedule
        self.epsilon_schedule = epsilon_schedule
        
        self.rng = np.random.default_rng(seed)
        
        # Q-table: Q[offer, time_left, stops_left, action]
        if Q_init is not None:
            self.Q = Q_init
        else:
            # Initialize Q-table
            self.Q = np.zeros((n_states, max_time + 1, max_stops + 1, n_actions))
        
        # Initialize Q-values for accept action if environment provides offer_values
        if env is not None and hasattr(env, 'offer_values'):
            for s in range(n_states):
                for t in range(max_time + 1):
                    self.Q[s, t, 1, 1] = env.offer_values[s]

        # Visit counts for diagnostics
        self.visit_counts = np.zeros((n_states, max_time + 1, max_stops + 1, n_actions), dtype=int)
        
        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_losses = []
        
    def get_state_tuple(self, obs: Dict[str, int]) -> Tuple[int, int, int]:
        """Convert observation dict to state tuple."""
        return (obs['offer'], obs['time_left'], obs['stops_left'])
    
    def get_q_value(self, obs: Dict[str, int], action: int) -> float:
        """Get Q-value for state-action pair."""
        state = self.get_state_tuple(obs)
        return self.Q[state][action]
    
    def get_max_q_value(self, obs: Dict[str, int]) -> float:
        """Get maximum Q-value for a state."""
        state = self.get_state_tuple(obs)
        return np.max(self.Q[state])
    
    def get_best_action(self, obs: Dict[str, int]) -> int:
        """Get greedy action for a state."""
        state = self.get_state_tuple(obs)
        return np.argmax(self.Q[state])
    
    def select_action(
        self,
        obs: Dict[str, int],
        epsilon: Optional[float] = None
    ) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            obs: Current observation
            epsilon: Exploration rate (uses self.epsilon if None)
            
        Returns:
            Selected action
        """
        if epsilon is None:
            epsilon = self.epsilon
        
        if self.rng.random() < epsilon:
            # Explore: random action
            return self.rng.integers(0, self.n_actions)
        else:
            # Exploit: best action
            return self.get_best_action(obs)
    
    def update(
        self,
        obs: Dict[str, int],
        action: int,
        reward: float,
        next_obs: Dict[str, int],
        terminated: bool,
        alpha: Optional[float] = None
    ) -> float:
        """
        Update Q-value using Q-learning update rule.
        
        Args:
            obs: Current observation
            action: Action taken
            reward: Reward received
            next_obs: Next observation
            terminated: Whether episode terminated
            alpha: Learning rate (uses self.alpha if None)
            
        Returns:
            TD error (for monitoring)
        """
        if alpha is None:
            alpha = self.alpha
        
        state = self.get_state_tuple(obs)
        current_q = self.Q[state][action]
        
        if terminated:
            target = reward
        else:
            next_max_q = self.get_max_q_value(next_obs)
            target = reward + self.gamma * next_max_q
        
        td_error = target - current_q
        self.Q[state][action] += alpha * td_error
        
        # Update visit count
        self.visit_counts[state][action] += 1

        return abs(td_error)


    def train_episode(
        self,
        env,
        episode_num: int,
        callbacks: Optional[List[Callable]] = None
    ) -> Dict[str, Any]:
        """
        Train for one episode.
        
        Args:
            env: Environment instance
            episode_num: Current episode number
            callbacks: List of callback functions
            
        Returns:
            Episode statistics
        """
        # Update hyperparameters if schedules are provided
        if self.alpha_schedule is not None:
            current_alpha = self.alpha_schedule(episode_num)
        else:
            current_alpha = self.alpha
        
        if self.epsilon_schedule is not None:
            current_epsilon = self.epsilon_schedule(episode_num)
        else:
            current_epsilon = self.epsilon
        
        # Reset environment
        obs, info = env.reset()
        
        episode_reward = 0.0
        episode_length = 0
        td_errors = []
        
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            # Select action
            action = self.select_action(obs, epsilon=current_epsilon)
            
            # Take step
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # Update Q-value
            td_error = self.update(
                obs, action, reward, next_obs,
                terminated or truncated,
                alpha=current_alpha
            )
            
            td_errors.append(td_error)
            episode_reward += reward
            episode_length += 1
            
            obs = next_obs

        # Store statistics
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        self.training_losses.append(np.mean(td_errors))
        
        # Execute callbacks
        if callbacks is not None:
            for callback in callbacks:
                callback(self, env, episode_num, {
                    'episode_reward': episode_reward,
                    'episode_length': episode_length,
                    'td_error': np.mean(td_errors),
                    'alpha': current_alpha,
                    'epsilon': current_epsilon
                })
        
        return {
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            'mean_td_error': np.mean(td_errors),
            'alpha': current_alpha,
            'epsilon': current_epsilon
        }
    
    def train(
        self,
        env,
        n_episodes: int,
        callbacks: Optional[List[Callable]] = None,
        verbose: int = 1,
        log_interval: int = 100,
        use_tqdm: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train agent for multiple episodes.
        
        Args:
            env: Environment instance
            n_episodes: Number of episodes to train
            callbacks: List of callback functions
            verbose: Verbosity level (0=silent, 1=progress, 2=detailed)
            log_interval: How often to print progress
            use_tqdm: Whether to use tqdm progress bar
            
        Returns:
            Training history
        """
        episode_iterator = range(n_episodes)
        if use_tqdm:
            episode_iterator = tqdm(
                episode_iterator, 
                desc="Training",
                unit="episode",
                ncols=100
            )
        
        for episode in episode_iterator:
            stats = self.train_episode(env, episode, callbacks)
            
            # Update tqdm postfix with current stats
            if use_tqdm and isinstance(episode_iterator, tqdm):
                postfix_dict = {
                    'reward': f"{stats['episode_reward']:.3f}",
                    'eps': f"{stats['epsilon']:.3f}"
                }
                if len(self.episode_rewards) >= 100:
                    postfix_dict['avg_reward'] = f"{np.mean(self.episode_rewards[-100:]):.3f}"
                episode_iterator.set_postfix(postfix_dict)
            
            if verbose > 0 and (episode + 1) % log_interval == 0:
                avg_reward = np.mean(self.episode_rewards[-log_interval:])
                avg_length = np.mean(self.episode_lengths[-log_interval:])
                
                if not use_tqdm:  # Only print if not using tqdm
                    print(f"Episode {episode + 1}/{n_episodes} | "
                          f"Avg Reward: {avg_reward:.4f} | "
                          f"Avg Length: {avg_length:.2f} | "
                          f"Epsilon: {stats['epsilon']:.4f} | "
                          f"Alpha: {stats['alpha']:.4f}")
                
                if verbose > 1:
                    print(f"  TD Error: {stats['mean_td_error']:.6f}")
        
        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'training_losses': self.training_losses
        }
    
    def evaluate(
        self,
        env,
        n_episodes: int = 100,
        seed: Optional[int] = None,
        use_tqdm: bool = True,
        verbose: int = 0
    ) -> Dict[str, Any]:
        """
        Evaluate agent (greedy policy, no exploration).
        
        Args:
            env: Environment instance
            n_episodes: Number of episodes to evaluate
            seed: Random seed for evaluation
            use_tqdm: Whether to use tqdm progress bar
            
        Returns:
            Evaluation statistics
        """
        if seed is not None:
            eval_rng = np.random.default_rng(seed)
        
        episode_rewards = []
        episode_lengths = []
        
        episode_iterator = range(n_episodes)
        if use_tqdm:
            episode_iterator = tqdm(
                episode_iterator,
                desc="Evaluating",
                unit="episode",
                ncols=100
            )
        
        for episode in episode_iterator:
            obs, info = env.reset(seed=seed + episode if seed else None)
            
            episode_reward = 0.0
            episode_length = 0
            
            terminated = False
            truncated = False
            
            while not (terminated or truncated):
                # Greedy action
                action = self.get_best_action(obs)
                
                obs, reward, terminated, truncated, info = env.step(action)
                
                episode_reward += reward
                episode_length += 1
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # Update tqdm postfix
            if use_tqdm and isinstance(episode_iterator, tqdm):
                episode_iterator.set_postfix({
                    'reward': f"{episode_reward:.3f}",
                    'avg': f"{np.mean(episode_rewards):.3f}"
                })
        if verbose:
            return {
                'mean_reward': np.mean(episode_rewards),
                'std_reward': np.std(episode_rewards),
                'mean_length': np.mean(episode_lengths),
                'std_length': np.std(episode_lengths),
                'min_reward': np.min(episode_rewards),
                'max_reward': np.max(episode_rewards),
                'episode_rewards': episode_rewards
            }
        else:
            return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards)
        }
    
    def save(self, filepath: str):
        """Save Q-table and training statistics."""
        save_dict = {
            'Q': self.Q.tolist(),
            'visit_counts': self.visit_counts.tolist(),
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'training_losses': self.training_losses,
            'hyperparameters': {
                'n_states': self.n_states,
                'n_actions': self.n_actions,
                'max_time': self.max_time,
                'max_stops': self.max_stops,
                'alpha': self.alpha,
                'gamma': self.gamma,
                'epsilon': self.epsilon
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(save_dict, f)
        
        print(f"Agent saved to {filepath}")
    
    def load(self, filepath: str):
        """Load Q-table and training statistics."""
        with open(filepath, 'r') as f:
            save_dict = json.load(f)
        
        self.Q = np.array(save_dict['Q'])
        self.visit_counts = np.array(save_dict['visit_counts'])
        self.episode_rewards = save_dict['episode_rewards']
        self.episode_lengths = save_dict['episode_lengths']
        self.training_losses = save_dict['training_losses']
        
        print(f"Agent loaded from {filepath}")
    
    def get_policy(self) -> np.ndarray:
        """
        Extract greedy policy from Q-table.
        
        Returns:
            Policy array of shape (n_states, max_time+1, max_stops+1)
        """
        return np.argmax(self.Q, axis=- 1)
    
class QLearningAgentVanilla(QLearningAgent):
    """
    Just a placeholder for a vanilla Q-learning agent.
    DIfferent name, same functionality as QLearningAgent.
    """
    pass

class QLearningAgentConstraint(QLearningAgent):
    """
    Q-Learning agent with monotonicity constraint in time.
    
    Enforces Q(s, t1, n, a) >= Q(s, t2, n, a) when t1 > t2
    (having more time left should be at least as valuable)
    
    After each episode, Q-values are adjusted to satisfy this constraint
    by propagating the maximum Q-value seen so far as time increases.
    """
    
    def _apply_monotonicity_constraint(self):
        """
        Apply monotonicity constraint: Q-values should be non-decreasing in time_left.
        For each (state, stops_left, action), ensure Q values increase with time_left.
        """
        for s in range(self.n_states):
            for n in range(self.max_stops + 1):
                for a in range(self.n_actions):
                    max_so_far = -np.inf
                    for t in range(self.max_time + 1):
                        max_so_far = max(max_so_far, self.Q[s, t, n, a])
                        if self.Q[s, t, n, a] < max_so_far:
                            self.Q[s, t, n, a] = max_so_far
    
    def train_episode(
        self,
        env,
        episode_num: int,
        callbacks: Optional[List[Callable]] = None
    ) -> Dict[str, Any]:
        """
        Train for one episode with monotonicity constraint applied after updates.
        """
        # Call parent's train_episode logic
        # Update hyperparameters if schedules are provided
        if self.alpha_schedule is not None:
            current_alpha = self.alpha_schedule(episode_num)
        else:
            current_alpha = self.alpha
        
        if self.epsilon_schedule is not None:
            current_epsilon = self.epsilon_schedule(episode_num)
        else:
            current_epsilon = self.epsilon
        
        # Reset environment
        obs, info = env.reset()
        
        episode_reward = 0.0
        episode_length = 0
        td_errors = []
        
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            # Select action
            action = self.select_action(obs, epsilon=current_epsilon)
            
            # Take step
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # Update Q-value
            td_error = self.update(
                obs, action, reward, next_obs,
                terminated or truncated,
                alpha=current_alpha
            )
            
            td_errors.append(td_error)
            episode_reward += reward
            episode_length += 1
            
            obs = next_obs

        # Store statistics
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        self.training_losses.append(np.mean(td_errors))

        # Apply monotonicity constraint after episode
        self._apply_monotonicity_constraint()
        
        # Execute callbacks
        if callbacks is not None:
            for callback in callbacks:
                callback(self, env, episode_num, {
                    'episode_reward': episode_reward,
                    'episode_length': episode_length,
                    'td_error': np.mean(td_errors),
                    'alpha': current_alpha,
                    'epsilon': current_epsilon
                })
        
        return {
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            'mean_td_error': np.mean(td_errors),
            'alpha': current_alpha,
            'epsilon': current_epsilon
        }


from collections import deque
import numpy as np
from typing import Optional, List, Callable, Dict, Any

# -----------------------------------------------------------
# 1. Off-Policy Weighted Importance Sampling (WIS) Agent
#    - Good for off-policy learning.
#    - Uses Importance Sampling ratios to correct for exploration.
#    - Uses 'C' table to stabilize variance (bias removal).
#    - RISK: "Trace Cutting" (updates become 0 if exploration happens deep in the trace).
# -----------------------------------------------------------

class OffPolicyWISAgent(QLearningAgent):
    """
    Q-Learning agent using n-step returns with Weighted Importance Sampling (WIS).
    """
    
    def __init__(self, n_step: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.n_step = n_step
        # C-table for Weighted Importance Sampling (accumulated weights)
        self.C = np.zeros_like(self.Q) 

    def get_behavior_prob(self, action: int, best_action: int, epsilon: float) -> float:
        """Calculate probability of taking 'action' under epsilon-greedy policy."""
        if action == best_action:
            return (1.0 - epsilon) + (epsilon / self.n_actions)
        else:
            return epsilon / self.n_actions

    def _compute_wis_return(self, buffer, final_next_obs, terminated):
        """Computes n-step return G and importance sampling ratio rho."""
        rho = 1.0
        
        # Bootstrap with Greedy Target Policy (max Q)
        if terminated:
            G = 0.0
        else:
            G = self.get_max_q_value(final_next_obs)
        
        # Iterate backwards to compute n-step return and importance ratio
        for i in reversed(range(len(buffer))):
            obs, action, reward, mu_prob = buffer[i]
            G = reward + self.gamma * G
            
            # Target policy is deterministic greedy
            greedy_action = self.get_best_action(obs)
            pi_prob = 1.0 if action == greedy_action else 0.0
                
            ratio = pi_prob / mu_prob
            rho *= ratio
            
            if rho == 0.0:  # Trace cut by non-greedy action
                break
                
        return G, rho

    def update_wis(self, obs, action, G, rho):
        """Apply Weighted Importance Sampling update rule."""
        if rho == 0.0: return 0.0
            
        state = self.get_state_tuple(obs)
        self.C[state][action] += rho
        
        current_q = self.Q[state][action]
        td_error = G - current_q
        
        # Effective alpha is rho / C
        learning_rate = rho / self.C[state][action]
        self.Q[state][action] += learning_rate * td_error

        self.visit_counts[state][action] += 1
        return abs(td_error)

    def train_episode(self, env, episode_num: int, callbacks: Optional[List[Callable]] = None) -> Dict[str, Any]:
        # Update schedules
        curr_alpha = self.alpha_schedule(episode_num) if self.alpha_schedule else self.alpha
        curr_eps = self.epsilon_schedule(episode_num) if self.epsilon_schedule else self.epsilon
            
        obs, info = env.reset()
        episode_reward = 0.0
        episode_length = 0
        td_errors = []
        buffer = deque(maxlen=self.n_step)
        
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            best_action = self.get_best_action(obs)
            action = self.select_action(obs, epsilon=curr_eps)
            mu_prob = self.get_behavior_prob(action, best_action, curr_eps)
            
            next_obs, reward, terminated, truncated, info = env.step(action)
            buffer.append((obs, action, reward, mu_prob))
            
            if len(buffer) == self.n_step:
                # Update oldest state in buffer
                upd_obs, upd_act, _, _ = buffer[0]
                G, rho = self._compute_wis_return(list(buffer), next_obs, terminated or truncated)
                err = self.update_wis(upd_obs, upd_act, G, rho)
                td_errors.append(err)

            episode_reward += reward
            episode_length += 1
            obs = next_obs
            
        # Flush buffer
        while len(buffer) > 0:
            upd_obs, upd_act, _, _ = buffer[0]
            G, rho = self._compute_wis_return(list(buffer), None, True)
            err = self.update_wis(upd_obs, upd_act, G, rho)
            td_errors.append(err)
            buffer.popleft()

        self._log_episode(episode_reward, episode_length, td_errors) # Helper to log stats
        return self._make_stats_dict(episode_reward, episode_length, td_errors, curr_alpha, curr_eps)

    def _log_episode(self, r, l, errs):
        self.episode_rewards.append(r)
        self.episode_lengths.append(l)
        self.training_losses.append(np.mean(errs) if errs else 0.0)

    def _make_stats_dict(self, r, l, errs, alpha, eps):
         return {
            'episode_reward': r,
            'mean_td_error': np.mean(errs) if errs else 0.0,
            'epsilon': eps,
            'alpha': alpha
        }


# -----------------------------------------------------------
# 2. Tree Backup Agent
#    - THE RECOMMENDED APPROACH.
#    - Robust to off-policy exploration without trace cutting.
#    - Dynamically shortens lookahead when non-greedy actions occur.
# -----------------------------------------------------------

class TreeBackupAgent(QLearningAgent):
    """
    Q-Learning agent using n-step Tree Backup algorithm.
    """
    
    def __init__(self, n_step: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.n_step = n_step

    def _compute_tree_backup_return(self, buffer, final_next_obs, terminated):
        """
        Computes the recursive Tree Backup return G.
        """
        # 1. Start with the value at the horizon (t+n)
        if terminated:
            G = 0.0
        else:
            G = self.get_max_q_value(final_next_obs)
            
        # 2. Iterate backwards from t+n-1 down to t
        for i in reversed(range(len(buffer))):
            obs, action, reward, _ = buffer[i]
            
            # Identify the greedy action for this state
            greedy_action = self.get_best_action(obs)
            
            if action == greedy_action:
                # Case A: We took the greedy action.
                # The tree continues deeper. Add reward and discount existing G.
                G = reward + self.gamma * G
            else:
                # Case B: We took an exploratory action.
                # Cut the deep trace and bootstrap from the *next* state.
                
                # Determine value of S_{t+1}
                if i == len(buffer) - 1:
                    # The next state is outside the buffer
                    if terminated:
                        next_val = 0.0
                    else:
                        # SAFEGUARD: Ensure final_next_obs is not None before using
                        next_val = self.get_max_q_value(final_next_obs) if final_next_obs is not None else 0.0
                else:
                    # The next state is the next item in the buffer
                    next_obs_in_buffer = buffer[i+1][0]
                    next_val = self.get_max_q_value(next_obs_in_buffer)
                
                G = reward + self.gamma * next_val
                
        return G

    def update_standard(self, obs, action, G, alpha):
        """Standard Q-learning update using the calculated Return G."""
        state = self.get_state_tuple(obs)
        current_q = self.Q[state][action]
        td_error = G - current_q
        
        self.Q[state][action] += alpha * td_error
        self.visit_counts[state][action] += 1
        return abs(td_error)

    def train_episode(self, env, episode_num: int, callbacks: Optional[List[Callable]] = None) -> Dict[str, Any]:
        curr_alpha = self.alpha_schedule(episode_num) if self.alpha_schedule else self.alpha
        curr_eps = self.epsilon_schedule(episode_num) if self.epsilon_schedule else self.epsilon
            
        obs, info = env.reset()
        episode_reward = 0.0
        episode_length = 0
        td_errors = []
        buffer = deque(maxlen=self.n_step)
        
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            action = self.select_action(obs, epsilon=curr_eps)
            
            next_obs, reward, terminated, truncated, info = env.step(action)
            # Tree Backup doesn't need mu_prob, storing 0.0
            buffer.append((obs, action, reward, 0.0))
            
            if len(buffer) == self.n_step:
                upd_obs, upd_act, _, _ = buffer[0]
                G = self._compute_tree_backup_return(list(buffer), next_obs, terminated or truncated)
                err = self.update_standard(upd_obs, upd_act, G, curr_alpha)
                td_errors.append(err)

            episode_reward += reward
            episode_length += 1
            obs = next_obs
            
        # Flush buffer
        while len(buffer) > 0:
            upd_obs, upd_act, _, _ = buffer[0]
            # Passing None for final_next_obs because we are flushing at terminal state
            # The safeguards in _compute_tree_backup_return handle this.
            G = self._compute_tree_backup_return(list(buffer), None, True)
            err = self.update_standard(upd_obs, upd_act, G, curr_alpha)
            td_errors.append(err)
            buffer.popleft()

        # Log stats
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        mean_loss = np.mean(td_errors) if td_errors else 0.0
        self.training_losses.append(mean_loss)
        
        return {
            'episode_reward': episode_reward,
            'mean_td_error': mean_loss,
            'epsilon': curr_eps,
            'alpha': curr_alpha
        }


class DoubleQLearningAgent(QLearningAgent):
    """
    Double Q-Learning agent to overcome overestimation bias.
    
    Double Q-Learning maintains two separate Q-value estimates (Q1 and Q2) and randomly
    selects which one to update at each step. When updating, it uses one Q-table to 
    select the best action and the other Q-table to evaluate that action. This 
    decoupling reduces the overestimation bias inherent in standard Q-learning.
    
    Reference: van Hasselt, H. (2010). Double Q-learning. NeurIPS.
    """
    
    def __init__(
        self,
        n_states: Optional[int] = None,
        n_actions: Optional[int] = None,
        max_time: Optional[int] = None,
        max_stops: Optional[int] = None,
        env = None,
        Q_init: Optional[np.ndarray] = None,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.1,
        alpha_schedule: Optional[Callable[[int], float]] = None,
        epsilon_schedule: Optional[Callable[[int], float]] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize Double Q-Learning agent.
        
        Args:
            n_states: Number of offer states (extracted from env if not provided)
            n_actions: Number of actions (extracted from env if not provided)
            max_time: Maximum time horizon (extracted from env if not provided)
            max_stops: Maximum number of stops (extracted from env if not provided)
            env: Environment instance to extract parameters from
            Q_init: Optional initial Q-table (used to initialize both Q1 and Q2)
            alpha: Initial learning rate
            gamma: Discount factor
            epsilon: Initial exploration rate
            alpha_schedule: Function(episode) -> alpha for time-varying learning rate
            epsilon_schedule: Function(episode) -> epsilon for time-varying exploration
            seed: Random seed
        """
        # Initialize parent class
        super().__init__(
            n_states=n_states,
            n_actions=n_actions,
            max_time=max_time,
            max_stops=max_stops,
            env=env,
            Q_init=Q_init,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            alpha_schedule=alpha_schedule,
            epsilon_schedule=epsilon_schedule,
            seed=seed
        )
        
        # Rename parent Q-table to Q1
        self.Q1 = self.Q
        
        # Initialize second Q-table (Q2)
        if Q_init is not None:
            self.Q2 = Q_init.copy()
        else:
            self.Q2 = np.zeros((self.n_states, self.max_time + 1, self.max_stops + 1, self.n_actions))
        
        # Initialize Q2-values for accept action if environment provides offer_values
        if env is not None and hasattr(env, 'offer_values'):
            for s in range(self.n_states):
                for t in range(self.max_time + 1):
                    self.Q2[s, t, 1, 1] = env.offer_values[s]
        
        # Visit counts for Q1 and Q2 (for diagnostics)
        self.visit_counts_q1 = np.zeros((self.n_states, self.max_time + 1, self.max_stops + 1, self.n_actions), dtype=int)
        self.visit_counts_q2 = np.zeros((self.n_states, self.max_time + 1, self.max_stops + 1, self.n_actions), dtype=int)
        
    def get_q_value(self, obs: Dict[str, int], action: int) -> float:
        """Get average Q-value for state-action pair from both Q-tables."""
        state = self.get_state_tuple(obs)
        return (self.Q1[state][action] + self.Q2[state][action]) / 2.0
    
    def get_max_q_value(self, obs: Dict[str, int]) -> float:
        """Get maximum average Q-value for a state."""
        state = self.get_state_tuple(obs)
        avg_q = (self.Q1[state] + self.Q2[state]) / 2.0
        return np.max(avg_q)
    
    def get_best_action(self, obs: Dict[str, int]) -> int:
        """Get greedy action based on average Q-values from both tables."""
        state = self.get_state_tuple(obs)
        avg_q = (self.Q1[state] + self.Q2[state]) / 2.0
        return np.argmax(avg_q)
    
    def update(
        self,
        obs: Dict[str, int],
        action: int,
        reward: float,
        next_obs: Dict[str, int],
        terminated: bool,
        alpha: Optional[float] = None
    ) -> float:
        """
        Update Q-value using Double Q-learning update rule.
        
        With probability 0.5:
            - Update Q1 using action selected by Q1 but evaluated by Q2
        Otherwise:
            - Update Q2 using action selected by Q2 but evaluated by Q1
        
        Args:
            obs: Current observation
            action: Action taken
            reward: Reward received
            next_obs: Next observation
            terminated: Whether episode terminated
            alpha: Learning rate (uses self.alpha if None)
            
        Returns:
            TD error (for monitoring)
        """
        if alpha is None:
            alpha = self.alpha
        
        state = self.get_state_tuple(obs)
        next_state = self.get_state_tuple(next_obs)
        
        # Randomly choose which Q-table to update
        if self.rng.random() < 0.5:
            # Update Q1: use Q1 to select action, Q2 to evaluate
            current_q = self.Q1[state][action]
            
            if terminated:
                target = reward
            else:
                # Q1 selects best action
                best_next_action = np.argmax(self.Q1[next_state])
                # Q2 evaluates that action
                next_q = self.Q2[next_state][best_next_action]
                target = reward + self.gamma * next_q
            
            td_error = target - current_q
            self.Q1[state][action] += alpha * td_error
            
            # Update visit count
            self.visit_counts_q1[state][action] += 1
            
        else:
            # Update Q2: use Q2 to select action, Q1 to evaluate
            current_q = self.Q2[state][action]
            
            if terminated:
                target = reward
            else:
                # Q2 selects best action
                best_next_action = np.argmax(self.Q2[next_state])
                # Q1 evaluates that action
                next_q = self.Q1[next_state][best_next_action]
                target = reward + self.gamma * next_q
            
            td_error = target - current_q
            self.Q2[state][action] += alpha * td_error
            
            # Update visit count
            self.visit_counts_q2[state][action] += 1
        
        # Also update parent class visit_counts for compatibility
        self.visit_counts[state][action] += 1
        
        return abs(td_error)
    
    def get_q_tables(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get both Q-tables.
        
        Returns:
            Tuple of (Q1, Q2)
        """
        return self.Q1, self.Q2
    
    def get_q_difference(self) -> np.ndarray:
        """
        Get the difference between Q1 and Q2.
        Useful for analyzing estimation variance.
        
        Returns:
            Absolute difference |Q1 - Q2|
        """
        return np.abs(self.Q1 - self.Q2)
    
    def get_overestimation_stats(self) -> Dict[str, float]:
        """
        Get statistics about Q-value differences between the two tables.
        
        Returns:
            Dictionary with mean, max, and std of |Q1 - Q2|
        """
        diff = self.get_q_difference()
        return {
            'mean_diff': np.mean(diff),
            'max_diff': np.max(diff),
            'std_diff': np.std(diff),
            'q1_mean': np.mean(self.Q1),
            'q2_mean': np.mean(self.Q2)
        }


# Hyperparameter schedules
def exponential_decay_schedule(
    initial_value: float,
    final_value: float,
    decay_episodes: int
) -> Callable[[int], float]:
    """
    Create exponential decay schedule.
    
    Args:
        initial_value: Starting value
        final_value: Final value (asymptote)
        decay_episodes: Number of episodes over which to decay
        
    Returns:
        Schedule function
    """
    def schedule(episode: int) -> float:
        decay_rate = -np.log(final_value / initial_value) / decay_episodes
        value = initial_value * np.exp(-decay_rate * episode)
        return max(value, final_value)
    
    return schedule


def linear_decay_schedule(
    initial_value: float,
    final_value: float,
    decay_episodes: int
) -> Callable[[int], float]:
    """
    Create linear decay schedule.
    
    Args:
        initial_value: Starting value
        final_value: Final value
        decay_episodes: Number of episodes over which to decay
        
    Returns:
        Schedule function
    """
    def schedule(episode: int) -> float:
        if episode >= decay_episodes:
            return final_value
        decay_amount = (initial_value - final_value) / decay_episodes
        return initial_value - decay_amount * episode
    
    return schedule


def step_decay_schedule(
    initial_value: float,
    decay_factor: float,
    decay_interval: int
) -> Callable[[int], float]:
    """
    Create step decay schedule.
    
    Args:
        initial_value: Starting value
        decay_factor: Factor to multiply by at each step
        decay_interval: Episodes between decay steps
        
    Returns:
        Schedule function
    """
    def schedule(episode: int) -> float:
        return initial_value * (decay_factor ** (episode // decay_interval))
    
    return schedule
