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

        # Visit counts for diagnostics
        self.visit_counts = np.zeros((n_states, max_time + 1, max_stops + 1))
        
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
        self.visit_counts[state] += 1
        
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
