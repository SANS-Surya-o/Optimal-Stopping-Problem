"""
Deep Q-Network (DQN) implementation compatible with the RL framework.
Supports experience replay, target networks, and time-varying hyperparameters.
"""
from typing import Optional, Dict, Any, Callable, List, Tuple
import numpy as np
from collections import deque
import random
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm


class QNetwork(nn.Module):
    """Neural network for Q-value approximation."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [128, 128]
    ):
        """
        Initialize Q-Network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Number of actions
            hidden_dims: List of hidden layer dimensions
        """
        super(QNetwork, self).__init__()
        
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(state)


class ReplayBuffer:
    """Experience replay buffer for DQN."""
    
    def __init__(self, capacity: int, seed: Optional[int] = None):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum buffer size
            seed: Random seed
        """
        self.buffer = deque(maxlen=capacity)
        if seed is not None:
            random.seed(seed)
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Add experience to buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        """Sample a batch of experiences."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)


class DQNAgent:
    """
    Deep Q-Network agent with:
    - Experience replay
    - Target network
    - Time-varying hyperparameters
    - Compatible with Gymnasium environments
    """
    
    def __init__(
        self,
        state_dim: Optional[int] = None,
        action_dim: Optional[int] = None,
        env = None,
        hidden_dims: List[int] = [128, 128],
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_schedule: Optional[Callable[[int], float]] = None,
        buffer_capacity: int = 100000,
        batch_size: int = 64,
        target_update_freq: int = 100,
        device: Optional[str] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize DQN agent.
        
        Args:
            state_dim: Dimension of state space (flattened)
            action_dim: Number of actions
            env: Environment instance to extract parameters from
            hidden_dims: List of hidden layer dimensions
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_schedule: Function(episode) -> epsilon for time-varying exploration
            buffer_capacity: Replay buffer capacity
            batch_size: Batch size for training
            target_update_freq: Frequency of target network updates (in episodes)
            device: Device for PyTorch ('cpu', 'cuda', or None for auto)
            seed: Random seed
        """
        # Extract parameters from environment if provided
        if env is not None:
            if state_dim is None:
                state_dim = self._infer_state_dim(env)
            if action_dim is None:
                if hasattr(env, 'get_action_space_size'):
                    action_dim = env.get_action_space_size()
                else:
                    action_dim = env.action_space.n
        
        if state_dim is None or action_dim is None:
            raise ValueError("Must provide state_dim and action_dim or env")
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_schedule = epsilon_schedule
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Initialize networks
        self.policy_net = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity, seed=seed)
        
        # Training stats
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []
        self.training_step = 0
        
        # Set random seed
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
    
    def _infer_state_dim(self, env) -> int:
        """Infer state dimension from environment."""
        if hasattr(env, 'observation_space'):
            if isinstance(env.observation_space, dict) or hasattr(env.observation_space, 'spaces'):
                # Dict observation space - flatten
                obs, _ = env.reset()
                return self._flatten_observation(obs).shape[0]
            else:
                return env.observation_space.n
        else:
            raise ValueError("Cannot infer state_dim from environment")
    
    def _flatten_observation(self, obs: Dict[str, int]) -> np.ndarray:
        """Flatten dictionary observation to numpy array."""
        if isinstance(obs, dict):
            return np.array([obs[key] for key in sorted(obs.keys())], dtype=np.float32)
        return np.array(obs, dtype=np.float32)
    
    def select_action(self, state: np.ndarray, episode: Optional[int] = None) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            episode: Current episode (for epsilon schedule)
            
        Returns:
            Selected action
        """
        # Update epsilon if schedule provided
        if self.epsilon_schedule is not None and episode is not None:
            epsilon = self.epsilon_schedule(episode)
        else:
            epsilon = self.epsilon
        
        # Epsilon-greedy
        if np.random.random() < epsilon:
            return np.random.randint(self.action_dim)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax(dim=1).item()
    
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Store transition in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def update(self) -> Optional[float]:
        """
        Perform one update step using a batch from replay buffer.
        
        Returns:
            Loss value if update performed, None otherwise
        """
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.training_step += 1
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network with policy network weights."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def train(
        self,
        env,
        n_episodes: int,
        verbose: int = 1,
        use_tqdm: bool = True,
        callbacks: Optional[List] = None
    ):
        """
        Train the DQN agent.
        
        Args:
            env: Environment to train on
            n_episodes: Number of episodes
            verbose: Verbosity level (0=silent, 1=progress, 2=detailed)
            use_tqdm: Whether to use tqdm progress bar
            callbacks: List of callback functions
        """
        iterator = tqdm(range(n_episodes), desc="Training DQN") if use_tqdm else range(n_episodes)
        
        for episode in iterator:
            obs, _ = env.reset()
            state = self._flatten_observation(obs)
            
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done:
                # Select action
                action = self.select_action(state, episode)
                
                # Take step
                next_obs, reward, terminated, truncated, _ = env.step(action)
                next_state = self._flatten_observation(next_obs)
                done = terminated or truncated
                
                # Store transition
                self.store_transition(state, action, reward, next_state, done)
                
                # Update
                loss = self.update()
                if loss is not None:
                    self.losses.append(loss)
                
                episode_reward += reward
                episode_length += 1
                state = next_state
            
            # Update target network periodically
            if (episode + 1) % self.target_update_freq == 0:
                self.update_target_network()
            
            # Record episode stats
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            
            # Callbacks
            if callbacks is not None:
                for callback in callbacks:
                    callback(episode, episode_reward, episode_length, self)
            
            # Logging
            if verbose == 2 or (verbose == 1 and (episode + 1) % 100 == 0):
                avg_reward = np.mean(self.episode_rewards[-100:])
                avg_loss = np.mean(self.losses[-100:]) if self.losses else 0
                epsilon = self.epsilon_schedule(episode) if self.epsilon_schedule else self.epsilon
                
                if use_tqdm:
                    iterator.set_postfix({
                        'avg_reward': f'{avg_reward:.2f}',
                        'epsilon': f'{epsilon:.3f}',
                        'loss': f'{avg_loss:.4f}'
                    })
                elif verbose >= 1:
                    print(f"Episode {episode+1}/{n_episodes} - "
                          f"Avg Reward: {avg_reward:.2f}, "
                          f"Epsilon: {epsilon:.3f}, "
                          f"Loss: {avg_loss:.4f}")
    
    def get_policy(self) -> np.ndarray:
        """
        Extract greedy policy from Q-network.
        
        For tabular environments, returns policy array.
        For large state spaces, this may not be feasible.
        """
        raise NotImplementedError(
            "Policy extraction not implemented for DQN with continuous state spaces. "
            "Use evaluate_policy() instead for evaluation."
        )
    
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """
        Get Q-values for a given state.
        
        Args:
            state: State vector
            
        Returns:
            Q-values for all actions
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.cpu().numpy()[0]
    
    def save(self, filepath: str):
        """Save model weights."""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'losses': self.losses
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model weights."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.episode_rewards = checkpoint['episode_rewards']
        self.episode_lengths = checkpoint['episode_lengths']
        self.losses = checkpoint['losses']
        print(f"Model loaded from {filepath}")


def linear_decay_schedule(
    initial_value: float,
    final_value: float,
    decay_episodes: int
) -> Callable[[int], float]:
    """
    Create linear decay schedule for epsilon.
    
    Args:
        initial_value: Starting value
        final_value: Final value
        decay_episodes: Number of episodes to decay over
        
    Returns:
        Schedule function
    """
    def schedule(episode: int) -> float:
        if episode >= decay_episodes:
            return final_value
        return initial_value - (initial_value - final_value) * episode / decay_episodes
    
    return schedule
