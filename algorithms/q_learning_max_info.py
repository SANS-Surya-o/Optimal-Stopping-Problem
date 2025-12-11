"""
Q-Learning variant that maximizes information gain per episode.

For single-stopping problems (e.g., buying a house), we know:
- Q(offer, time_left, accept) = offer (deterministic, no learning needed)
- Q(offer, time_left, reject) = must be learned from experience

Strategy: Run episodes where we always reject until the final time step.
This lets us observe the full trajectory of offers and learn reject values better.
"""
from typing import Optional, Dict, Any, Callable, List
import numpy as np
from tqdm.auto import tqdm
from algorithms.q_learning import QLearningAgent


class QLearningMaxInfo(QLearningAgent):
    """
    Q-Learning variant that rejects all offers until the last time step.
    This maximizes the information gained about future offers per episode.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.total_forced_rejects = 0
    
    def train_episode(
        self,
        env,
        episode_num: int,
        callbacks: Optional[List[Callable]] = None
    ) -> Dict[str, Any]:
        """
        Train for one episode, forcing rejection until the last time step.
        """
        # Update hyperparameters
        if self.alpha_schedule is not None:
            current_alpha = self.alpha_schedule(episode_num)
        else:
            current_alpha = self.alpha
        
        if self.epsilon_schedule is not None:
            current_epsilon = self.epsilon_schedule(episode_num)
        else:
            current_epsilon = self.epsilon
        
        # Run episode
        obs, info = env.reset()
        
        episode_reward = 0.0
        episode_length = 0
        td_errors = []
        forced_rejects = 0
        
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            time_left = obs['time_left']
            
            # Force rejection until last time step (time_left = 1)
            if time_left > 1:
                action = 0  # Reject
                forced_rejects += 1
            else:
                # On last step, select action normally (epsilon-greedy)
                action = self.select_action(obs, epsilon=current_epsilon)
            
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
        
        self.total_forced_rejects += forced_rejects
        
        # Store statistics
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        self.training_losses.append(np.mean(td_errors) if td_errors else 0.0)
        
        # Execute callbacks
        if callbacks is not None:
            for callback in callbacks:
                callback(self, env, episode_num, {
                    'episode_reward': episode_reward,
                    'episode_length': episode_length,
                    'td_error': np.mean(td_errors) if td_errors else 0.0,
                    'alpha': current_alpha,
                    'epsilon': current_epsilon,
                    'forced_rejects': forced_rejects
                })
        
        return {
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            'mean_td_error': np.mean(td_errors) if td_errors else 0.0,
            'alpha': current_alpha,
            'epsilon': current_epsilon,
            'forced_rejects': forced_rejects
        }
