"""
Training callbacks and monitoring utilities for Q-Learning.
Supports logging, visualization, and optional W&B integration.
"""
from typing import Optional, Dict, Any, List, Tuple, Callable
import numpy as np
from pathlib import Path
import json
from datetime import datetime


class TrainingMonitor:
    """
    Monitor and log training progress.
    Supports multiple callbacks and optional W&B integration.
    """
    
    def __init__(
        self,
        log_dir: Optional[str] = None,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize training monitor.
        
        Args:
            log_dir: Directory to save logs
            use_wandb: Whether to use Weights & Biases
            wandb_project: W&B project name
            wandb_config: W&B configuration dict
        """
        self.log_dir = Path(log_dir) if log_dir else Path("logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_wandb = use_wandb
        self.wandb_run = None
        
        if use_wandb:
            try:
                import wandb
                self.wandb = wandb
                self.wandb_run = wandb.init(
                    project=wandb_project or "optimal_stopping_rl",
                    config=wandb_config or {}
                )
                print(f"W&B logging enabled: {self.wandb_run.url}")
            except ImportError:
                print("Warning: wandb not installed. Install with: pip install wandb")
                self.use_wandb = False
        
        # Create log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"training_log_{timestamp}.jsonl"
        
        # Store metrics
        self.metrics_history = []
    
    def log_episode(
        self,
        agent,
        env,
        episode: int,
        stats: Dict[str, Any]
    ):
        """
        Log statistics for an episode.
        
        Args:
            agent: Q-learning agent
            env: Environment
            episode: Episode number
            stats: Episode statistics
        """
        # Add episode number to stats
        stats['episode'] = episode
        
        # Log to file
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(stats) + '\n')
        
        # Log to W&B
        if self.use_wandb and self.wandb_run is not None:
            self.wandb.log(stats, step=episode)
        
        # Store in history
        self.metrics_history.append(stats)
    
    def log_evaluation(
        self,
        episode: int,
        eval_stats: Dict[str, Any]
    ):
        """
        Log evaluation statistics.
        
        Args:
            episode: Episode number
            eval_stats: Evaluation statistics
        """
        eval_stats['episode'] = episode
        eval_stats['type'] = 'evaluation'
        
        # Log to file
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(eval_stats) + '\n')
        
        # Log to W&B with eval prefix
        if self.use_wandb and self.wandb_run is not None:
            wandb_stats = {f'eval/{k}': v for k, v in eval_stats.items() if k != 'episode_rewards'}
            self.wandb.log(wandb_stats, step=episode)
        
        print(f"\nEvaluation at episode {episode}:")
        print(f"  Mean Reward: {eval_stats['mean_reward']:.4f} ± {eval_stats['std_reward']:.4f}")
        print(f"  Mean Length: {eval_stats['mean_length']:.2f} ± {eval_stats['std_length']:.2f}")
    
    def finish(self):
        """Clean up and finish logging."""
        if self.use_wandb and self.wandb_run is not None:
            self.wandb_run.finish()
        
        print(f"\nTraining logs saved to: {self.log_file}")


class ProgressCallback:
    """Callback to print progress during training."""
    
    def __init__(self, log_interval: int = 100):
        """
        Initialize progress callback.
        
        Args:
            log_interval: How often to print progress
        """
        self.log_interval = log_interval
        self.recent_rewards = []
    
    def __call__(
        self,
        agent,
        env,
        episode: int,
        stats: Dict[str, Any]
    ):
        """Called after each episode."""
        self.recent_rewards.append(stats['episode_reward'])
        
        if (episode + 1) % self.log_interval == 0:
            avg_reward = np.mean(self.recent_rewards[-self.log_interval:])
            print(f"Episode {episode + 1}: Avg Reward = {avg_reward:.4f}")


class CheckpointCallback:
    """Callback to save agent checkpoints."""
    
    def __init__(
        self,
        save_dir: str,
        save_interval: int = 1000,
        keep_best: bool = True
    ):
        """
        Initialize checkpoint callback.
        
        Args:
            save_dir: Directory to save checkpoints
            save_interval: How often to save checkpoints
            keep_best: Whether to keep track of best model
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.save_interval = save_interval
        self.keep_best = keep_best
        self.best_reward = -np.inf
    
    def __call__(
        self,
        agent,
        env,
        episode: int,
        stats: Dict[str, Any]
    ):
        """Called after each episode."""
        # Regular checkpoint
        if (episode + 1) % self.save_interval == 0:
            filepath = self.save_dir / f"checkpoint_ep{episode + 1}.json"
            agent.save(str(filepath))
        
        # Best checkpoint
        if self.keep_best:
            if len(agent.episode_rewards) >= 100:
                recent_avg = np.mean(agent.episode_rewards[-100:])
                if recent_avg > self.best_reward:
                    self.best_reward = recent_avg
                    filepath = self.save_dir / "best_model.json"
                    agent.save(str(filepath))
                    print(f"New best model saved! Avg reward: {recent_avg:.4f}")


class EvaluationCallback:
    """Callback to evaluate agent during training."""
    
    def __init__(
        self,
        eval_env,
        eval_interval: int = 1000,
        n_eval_episodes: int = 100,
        monitor: Optional[TrainingMonitor] = None
    ):
        """
        Initialize evaluation callback.
        
        Args:
            eval_env: Environment for evaluation
            eval_interval: How often to evaluate
            n_eval_episodes: Number of episodes per evaluation
            monitor: Training monitor for logging
        """
        self.eval_env = eval_env
        self.eval_interval = eval_interval
        self.n_eval_episodes = n_eval_episodes
        self.monitor = monitor
    
    def __call__(
        self,
        agent,
        env,
        episode: int,
        stats: Dict[str, Any]
    ):
        """Called after each episode."""
        if (episode + 1) % self.eval_interval == 0:
            eval_stats = agent.evaluate(
                self.eval_env,
                n_episodes=self.n_eval_episodes,
                seed=42
            )
            
            if self.monitor is not None:
                self.monitor.log_evaluation(episode + 1, eval_stats)
            else:
                print(f"\nEvaluation at episode {episode + 1}:")
                print(f"  Mean Reward: {eval_stats['mean_reward']:.4f}")


class MetricsTracker:
    """Track and compute various training metrics."""
    
    def __init__(self, window_size: int = 100):
        """
        Initialize metrics tracker.
        
        Args:
            window_size: Window size for moving averages
        """
        self.window_size = window_size
        self.rewards = []
        self.lengths = []
        self.td_errors = []
    
    def update(self, stats: Dict[str, Any]):
        """Update metrics with episode statistics."""
        self.rewards.append(stats.get('episode_reward', 0))
        self.lengths.append(stats.get('episode_length', 0))
        self.td_errors.append(stats.get('td_error', 0))
    
    def get_metrics(self) -> Dict[str, float]:
        """Get current metrics."""
        if len(self.rewards) == 0:
            return {}
        
        # Use last window_size episodes
        recent_rewards = self.rewards[-self.window_size:]
        recent_lengths = self.lengths[-self.window_size:]
        recent_td_errors = self.td_errors[-self.window_size:]
        
        return {
            'avg_reward': np.mean(recent_rewards),
            'std_reward': np.std(recent_rewards),
            'avg_length': np.mean(recent_lengths),
            'avg_td_error': np.mean(recent_td_errors)
        }
    
    def __call__(
        self,
        agent,
        env,
        episode: int,
        stats: Dict[str, Any]
    ):
        """Called after each episode."""
        self.update(stats)


def create_default_callbacks(
    save_dir: str = "checkpoints",
    log_dir: str = "logs",
    use_wandb: bool = False,
    wandb_project: Optional[str] = None,
    wandb_config: Optional[Dict[str, Any]] = None,
    eval_env = None,
    eval_interval: int = 1000
) -> Tuple[List[Callable], TrainingMonitor]:
    """
    Create default set of callbacks.
    
    Args:
        save_dir: Directory for checkpoints
        log_dir: Directory for logs
        use_wandb: Whether to use W&B
        wandb_project: W&B project name
        wandb_config: W&B configuration
        eval_env: Environment for evaluation
        eval_interval: Evaluation interval
        
    Returns:
        List of callbacks and training monitor
    """
    monitor = TrainingMonitor(
        log_dir=log_dir,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_config=wandb_config
    )
    
    callbacks = [
        lambda agent, env, ep, stats: monitor.log_episode(agent, env, ep, stats),
        CheckpointCallback(save_dir=save_dir, save_interval=1000),
        MetricsTracker(window_size=100)
    ]
    
    if eval_env is not None:
        callbacks.append(
            EvaluationCallback(
                eval_env=eval_env,
                eval_interval=eval_interval,
                monitor=monitor
            )
        )
    
    return callbacks, monitor
