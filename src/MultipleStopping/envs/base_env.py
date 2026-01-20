"""
Base environment for multiple optimal stopping problems.
Compatible with Gymnasium interface for future extensibility.
"""
from typing import Optional, Tuple, Dict, Any, Callable
import numpy as np
from abc import ABC, abstractmethod
import gymnasium as gym
from gymnasium import spaces


class BaseStoppingEnv(gym.Env, ABC):
    """
    Base class for multiple optimal stopping problems.
    
    State space: (offer, time_left, stops_left)
    Action space: {0: continue/reject, 1: stop/accept}
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(
        self,
        n_states: int,
        max_time: int,
        max_stops: int,
        transition_model: str = 'random_walk',
        transition_params: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize base stopping environment.
        
        Args:
            n_states: Number of discrete offer states
            max_time: Maximum time horizon
            max_stops: Maximum number of stops allowed
            transition_model: Type of stochastic process. Options:
                - 'iid': IID uniform or custom distribution
                - 'iid_gaussian': IID Gaussian distribution
                - 'iid_geometric': IID Geometric distribution
                - 'iid_poisson': IID Poisson distribution
                - 'iid_exponential': IID Exponential distribution (discretized)
                - 'iid_binomial': IID Binomial distribution
                - 'random_walk': Random walk with configurable probabilities
                - 'brownian': Discrete Brownian motion
            transition_params: Parameters for transition model. Examples:
                - iid_gaussian: {'mean': 5.0, 'std': 2.0}
                - iid_geometric: {'p': 0.3}
                - iid_poisson: {'lambda': 3.0}
                - iid_exponential: {'rate': 1.0}
                - iid_binomial: {'n': 10, 'p': 0.5}
                - random_walk: {'p_up': 0.5, 'p_down': 0.5}
                - brownian: {'mu': 0.0, 'sigma': 1.0}
            seed: Random seed for reproducibility
        """
        super().__init__()
        
        self.n_states = n_states
        self.max_time = max_time
        self.max_stops = max_stops
        self.transition_model = transition_model
        self.transition_params = transition_params or {}
        self.offline_paths = None

        # Set random seed
        self.np_random = np.random.default_rng(seed)
        
        # Define observation space: (offer, time_left, stops_left)
        self.observation_space = spaces.Dict({
            'offer': spaces.Discrete(n_states),
            'time_left': spaces.Discrete(max_time + 1),
            'stops_left': spaces.Discrete(max_stops + 1)
        })
        
        # Define action space: 0 = continue, 1 = stop
        self.action_space = spaces.Discrete(2)
        
        # Initialize transition probability matrix
        self._init_transition_matrix()
        
        # Current state
        self.current_offer = 0
        self.current_time = 0
        self.stops_made = 0
        
        # Episode tracking
        self.episode_history = []
        
    def _init_transition_matrix(self):
        """Initialize transition probability matrix based on model type."""
        if self.transition_model == 'iid_gaussian':
            self.P = self._create_iid_gaussian_transition()
        elif self.transition_model == 'iid':
            self.P = self._create_iid_transition()
        elif self.transition_model == 'iid_geometric':
            self.P = self._create_iid_geometric_transition()
        elif self.transition_model == 'iid_poisson':
            self.P = self._create_iid_poisson_transition()
        elif self.transition_model == 'iid_exponential':
            self.P = self._create_iid_exponential_transition()
        elif self.transition_model == 'iid_binomial':
            self.P = self._create_iid_binomial_transition()
        elif self.transition_model == 'random_walk':
            self.P = self._create_random_walk_transition()
        elif self.transition_model == 'brownian':
            self.P = self._create_brownian_transition()
        elif self.transition_model == 'up_down':
            self.P = self._create_up_down_transition()
        elif self.transition_model == 'custom':
            if 'tpm' in self.transition_params:
                self.P = self.transition_params['tpm']
            else:
                raise ValueError("Custom transition model requires 'tpm' in transition_params.")
        else:
            raise ValueError(f"Unknown transition model: {self.transition_model}")
        
    def _create_iid_gaussian_transition(self) -> np.ndarray:
        """
        Create IID Gaussian transition matrix.
        Each state transitions to any state according to a Gaussian distribution
        centered at some mean with given standard deviation.
        """
        mean = self.transition_params.get('mean', self.n_states / 2.0)
        std = self.transition_params.get('std', self.n_states / 4.0)
        
        # Create probability distribution over states
        probs = np.zeros(self.n_states)
        for i in range(self.n_states):
            probs[i] = np.exp(-0.5 * ((i - mean) / std) ** 2)
        
        # Normalize to create valid probability distribution
        probs /= probs.sum()
        
        # IID: same distribution from every state
        P = np.tile(probs, (self.n_states, 1))
        
        return P
    
    def _create_iid_transition(self) -> np.ndarray:
        """Create IID transition matrix (uniform distribution)."""
        prob = self.transition_params.get('prob', None)
        if prob is None:
            # Uniform distribution
            P = np.ones((self.n_states, self.n_states)) / self.n_states
        else:
            # Custom probability distribution
            P = np.tile(prob, (self.n_states, 1))
        return P
    
    def _create_iid_geometric_transition(self) -> np.ndarray:
        """
        Create IID Geometric transition matrix.
        Geometric distribution: P(X=k) = (1-p)^(k-1) * p for k = 0, 1, 2, ...
        """
        p = self.transition_params.get('p', 0.3)  # Success probability
        
        # Create geometric probability distribution over states
        probs = np.zeros(self.n_states)
        for i in range(self.n_states):
            probs[i] = ((1 - p) ** i) * p
        
        # Normalize to ensure sum = 1 (for finite support)
        probs /= probs.sum()
        
        # IID: same distribution from every state
        P = np.tile(probs, (self.n_states, 1))
        
        return P
    
    def _create_iid_poisson_transition(self) -> np.ndarray:
        """
        Create IID Poisson transition matrix.
        Poisson distribution: P(X=k) = (lambda^k * e^(-lambda)) / k!
        """
        lam = self.transition_params.get('lambda', self.n_states / 3.0)  # Rate parameter
        
        # Create Poisson probability distribution over states
        probs = np.zeros(self.n_states)
        for i in range(self.n_states):
            # Compute Poisson PMF
            if i == 0:
                probs[i] = np.exp(-lam)
            else:
                # Use log to avoid overflow: log(k!) = sum(log(j)) for j=1 to k
                log_factorial = np.sum(np.log(np.arange(1, i + 1)))
                probs[i] = np.exp(i * np.log(lam) - lam - log_factorial)
        
        # Normalize to ensure sum = 1 (for finite support)
        probs /= probs.sum()
        
        # IID: same distribution from every state
        P = np.tile(probs, (self.n_states, 1))
        
        return P
    
    def _create_iid_exponential_transition(self) -> np.ndarray:
        """
        Create IID Exponential transition matrix (discretized).
        Exponential distribution discretized over states.
        """
        rate = self.transition_params.get('rate', 1.0)  # Rate parameter (lambda)
        
        # Create discretized exponential distribution
        # For discrete states 0, 1, 2, ..., n_states-1
        # We use the CDF: F(k) = 1 - e^(-rate * k)
        # PMF: P(X=k) = F(k+1) - F(k)
        probs = np.zeros(self.n_states)
        for i in range(self.n_states):
            if i == self.n_states - 1:
                # Last state gets remaining probability
                probs[i] = np.exp(-rate * i)
            else:
                probs[i] = np.exp(-rate * i) - np.exp(-rate * (i + 1))
        
        # Normalize (should already be normalized, but for safety)
        probs /= probs.sum()
        
        # IID: same distribution from every state
        P = np.tile(probs, (self.n_states, 1))
        
        return P
    
    def _create_iid_binomial_transition(self) -> np.ndarray:
        """
        Create IID Binomial transition matrix.
        Binomial distribution: P(X=k) = C(n,k) * p^k * (1-p)^(n-k)
        """
        n_trials = self.transition_params.get('n', self.n_states - 1)  # Number of trials
        p = self.transition_params.get('p', 0.5)  # Success probability
        
        # Create binomial probability distribution over states
        probs = np.zeros(self.n_states)
        
        from scipy.special import comb
        
        for i in range(min(self.n_states, n_trials + 1)):
            # Binomial PMF
            probs[i] = comb(n_trials, i, exact=True) * (p ** i) * ((1 - p) ** (n_trials - i))
        
        # Normalize to ensure sum = 1
        probs /= probs.sum()
        
        # IID: same distribution from every state
        P = np.tile(probs, (self.n_states, 1))
        
        return P
    
    def _create_random_walk_transition(self) -> np.ndarray:
        """Create random walk transition matrix."""
        # extract up and down probs from params
        p_up = self.transition_params.get('p_up', 0.5)
        p_down = self.transition_params.get('p_down', 0.5)
        p_stay = max(1 - p_up - p_down,0)

        P = np.zeros((self.n_states, self.n_states))
        for i in range(self.n_states):
            if i > 0:
                P[i, i-1] = p_down
            if i < self.n_states - 1:
                P[i, i+1] = p_up
            P[i, i] = p_stay
            
            # Handle boundary conditions
            if i == 0:
                P[i, i] += p_down/2
                P[i, i+1] += p_down/2
            if i == self.n_states - 1:
                P[i, i] += p_up/2
                P[i, i-1] += p_up/2
        
        return P
    
    def _create_brownian_transition(self) -> np.ndarray:
        """Create discrete approximation of Brownian motion."""
        mu = self.transition_params.get('mu', 0.0)
        sigma = self.transition_params.get('sigma', 1.0)
        
        P = np.zeros((self.n_states, self.n_states))
        
        for i in range(self.n_states):
            # Gaussian distribution around current state
            for j in range(self.n_states):
                diff = j - i
                P[i, j] = np.exp(-0.5 * ((diff - mu) / sigma) ** 2)
            
            # Normalize
            P[i, :] /= P[i, :].sum()
        
        return P

    def _create_up_down_transition(self) -> np.ndarray:
            P = np.zeros((self.n_states, self.n_states))
            alpha = self.transition_params.get("alpha", 0.1)
            for i in range(self.n_states):
                for j in range(self.n_states):
                    if j == i:
                        P[i, j] = 1 - 2 * alpha
                    elif j == i + 1 or j == i - 1:
                        P[i, j] = alpha
                P[i, :] /= np.sum(P[i, :])
            return P

    def get_transition_probability(self, state_from: int, state_to: int) -> float:
        """
        Get transition probability between two states.
        
        Args:
            state_from: Current state index
            state_to: Next state index
            
        Returns:
            Probability of transition
        """
        return self.P[state_from, state_to]
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, int], Dict[str, Any]]:
        """
        Reset environment to initial state.
        
        Returns:
            observation: Initial observation
            info: Additional information
        """
        super().reset(seed=seed)
        
        # Reset time and stops
        self.current_time = 0
        self.stops_made = 0
        
        # Sample initial offer
        initial_dist = self.transition_params.get('initial_dist', None)
        if initial_dist is None:
            self.current_offer = np.random.choice(self.n_states, p=self.P[0])
        else:
            self.current_offer = np.random.choice(
                self.n_states, p=initial_dist
            )
        
        # Reset episode history
        self.episode_history = [{
            'time': self.current_time,
            'offer': self.current_offer,
            'stops_left': self.max_stops - self.stops_made,
            'action': None,
            'reward': 0.0
        }]
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(
        self, action: int
    ) -> Tuple[Dict[str, int], float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: 0 = continue, 1 = stop
            
        Returns:
            observation: Next observation
            reward: Reward for this step
            terminated: Whether episode has ended
            truncated: Whether episode was truncated
            info: Additional information
        """
        if action not in [0, 1]:
            raise ValueError(f"Invalid action: {action}")
        
        # Calculate reward for current action
        reward = self._compute_reward(action)
        
        # Record history
        self.episode_history.append({
            'time': self.current_time,
            'offer': self.current_offer,
            'stops_left': self.max_stops - self.stops_made,
            'action': action,
            'reward': reward
        })
        
        # Update state
        terminated = False
        truncated = False
        
        if action == 1:  # Stop
            self.stops_made += 1
            if self.stops_made >= self.max_stops:
                terminated = True
        
        # Advance time
        self.current_time += 1
        
        if self.current_time >= self.max_time:
            truncated = True
        
        # Sample next offer if not terminated
        if not (terminated or truncated):
            self.current_offer = self._sample_next_offer()
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    


    ### NOTES: A Much cleaner solution for paths would be to have a function that generates the next state given the current
    # instead of using the tpm elements directly - this can be more easily extended to continuous state spaces.
    def _generate_offline_paths(self, n_paths: int):
        """Generate offline paths for batch learning."""
        paths = []
        for _ in range(n_paths): 
            path = []
            current_offer = self.np_random.choice(self.n_states, p=self.P[0])
            for t in range(self.max_time):
                path.append(current_offer)
                current_offer = self.np_random.choice(
                    self.n_states, p=self.P[current_offer]
                )
            paths.append(path)
        self.offline_paths = paths
    
    def _sample_next_offer(self) -> int:
        """Sample next offer based on transition probabilities."""
        probs = self.P[self.current_offer, :]
        return self.np_random.choice(self.n_states, p=probs)
        
    
    def _get_observation(self) -> Dict[str, int]:
        """Get current observation."""
        return {
            'offer': self.current_offer,
            'time_left': self.max_time - self.current_time,
            'stops_left': self.max_stops - self.stops_made
        }
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional information."""
        return {
            'time': self.current_time,
            'stops_made': self.stops_made,
            'episode_history': self.episode_history
        }
    
    @abstractmethod
    def _compute_reward(self, action: int) -> float:
        """
        Compute reward for current action.
        Must be implemented by subclasses.
        
        Args:
            action: Action taken
            
        Returns:
            Reward value
        """
        pass
    
    def solve_optimal_policy(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute optimal value function and policy using dynamic programming.
        Assumes known transition probabilities.
        
        Returns:
            Tuple of (value_function, policy) where:
            - value_function: V[offer, time_left, stops_left]
            - policy: pi[offer, time_left, stops_left] = action (0=reject, 1=accept)
        """
        raise NotImplementedError(
            "Optimal policy solving not implemented for this environment. "
            "Override solve_optimal_policy() in subclass."
        )
    
    def render(self):
        """Render current state."""
        if len(self.episode_history) > 0:
            last_entry = self.episode_history[-1]
            print(f"Time: {last_entry['time']}/{self.max_time}, "
                  f"Offer: {last_entry['offer']}, "
                  f"Stops left: {last_entry['stops_left']}, "
                  f"Action: {last_entry.get('action', 'None')}, "
                  f"Reward: {last_entry['reward']:.2f}")
    
    def close(self):
        """Clean up resources."""
        pass
