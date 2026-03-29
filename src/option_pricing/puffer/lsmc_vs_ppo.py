import os
import sys
import numpy as np
import torch
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# PufferLib Imports
import pufferlib
import pufferlib.vector
import pufferlib.pytorch
import pufferlib.ocean
from pufferlib import pufferl

# ==========================================
# 1. RECURSIVE LSMC (Multiple Stopping)
# ==========================================
class RecursiveLSMC:
    def __init__(self, n_rights, max_steps, delta, n_processes, strike, r_rate, degree=2):
        self.L, self.T, self.delta, self.M = n_rights, max_steps, delta, n_processes
        self.K, self.r, self.degree = strike, r_rate, degree
        self.models = {}
        self.exercise_boundary = {}  # Store exercise boundaries for decision making

    def _get_features(self, prices):
        """Polynomial basis functions for multi-dimensional price inputs."""
        poly = PolynomialFeatures(degree=self.degree, include_bias=False)
        return poly.fit_transform(prices.reshape(-1, self.M) if prices.ndim == 1 else prices)

    def calculate_payoff_at_t(self, prices, t):
        """Pay-off logic matching the ocean environment."""
        avg_price = np.mean(prices) if prices.ndim == 1 else np.mean(prices, axis=1)
        payoff = np.maximum(self.K - avg_price, 0)
        discount = np.exp(-self.r * (t / self.T))
        return payoff * discount

    def train(self, env, n_train_paths):
        print(f"Generating {n_train_paths} paths for LSMC training...")
        all_paths = []
        for _ in range(n_train_paths):
            env.reset() 
            all_paths.append(env.paths.copy())
        
        paths = np.array(all_paths)  # (N, T+1, M)
        N, _, _ = paths.shape
        V = np.zeros((N, self.T + 1, self.L + 1))
        dt = 1.0 / self.T
        
        # Backward Induction Cascade
        for l in range(1, self.L + 1):
            # Terminal condition: exercise if in the money
            V[:, self.T, l] = self.calculate_payoff_at_t(paths[:, self.T], self.T) + V[:, self.T, l-1]
            
            for t in range(self.T - 1, -1, -1):
                t_delta = min(t + self.delta, self.T)
                payoff_t = self.calculate_payoff_at_t(paths[:, t], t)
                
                # Continuation value (discounted expected future value)
                continuation_target = V[:, t+1, l] * np.exp(-self.r * dt)
                
                # Stop value: immediate payoff + future value with one less right
                stop_val = payoff_t + V[:, t_delta, l-1] * np.exp(-self.r * (dt * self.delta))
                
                itm = payoff_t > 0
                if np.sum(itm) > (self.M * self.degree + 10):
                    X, y = self._get_features(paths[itm, t]), continuation_target[itm]
                    model = LinearRegression().fit(X, y)
                    self.models[(l, t)] = model
                    
                    # Predict continuation value for ITM paths
                    cont_pred = model.predict(X)
                    
                    # Exercise if stop value > predicted continuation
                    exercise = stop_val[itm] > cont_pred
                    V[itm, t, l] = np.where(exercise, stop_val[itm], continuation_target[itm])
                    
                    # Store exercise boundary info
                    if exercise.any():
                        exercised_prices = np.mean(paths[itm, t][exercise], axis=-1) if paths[itm, t][exercise].ndim > 1 else paths[itm, t][exercise]
                        self.exercise_boundary[(l, t)] = np.max(exercised_prices)
                
                V[~itm, t, l] = continuation_target[~itm]
        
        print(f"LSMC Training Complete. Models trained: {len(self.models)}")

    def run_lsmc_decision(self, test_path):
        """Run LSMC policy on a single test path."""
        rights, timer, total_payoff = self.L, 0, 0
        
        for t in range(self.T + 1):
            if rights == 0:
                break
            if timer > 0:
                timer -= 1
                continue
            
            payoff_t = self.calculate_payoff_at_t(test_path[t], t)
            
            # At maturity, always exercise if in the money
            if t == self.T:
                if payoff_t > 0:
                    total_payoff += payoff_t
                    rights -= 1
                continue
            
            # If in the money, check if we should exercise
            if payoff_t > 0:
                if (rights, t) in self.models:
                    X = self._get_features(test_path[t].reshape(1, -1))
                    continuation_value = self.models[(rights, t)].predict(X)[0]
                    
                    # Calculate stop value (payoff + discounted future with fewer rights)
                    dt = 1.0 / self.T
                    t_delta = min(t + self.delta, self.T)
                    
                    # For simplicity, estimate future value as 0 if no more rights after this
                    # In practice, you'd need the expected value with rights-1
                    if payoff_t > continuation_value:
                        total_payoff += payoff_t
                        rights -= 1
                        timer = self.delta
                else:
                    # No model for this (rights, t) - use simple heuristic
                    # Exercise if deep in the money (price < 0.9 * strike)
                    avg_price = np.mean(test_path[t])
                    if avg_price < 0.85 * self.K:
                        total_payoff += payoff_t
                        rights -= 1
                        timer = self.delta
        
        return total_payoff

# ==========================================
# 2. PPO TRAINER (Using Ocean Import)
# ==========================================
def ppo_trainer(env_name='puffer_option_pricing', env_params=None):
    if env_params is None: env_params = {}
    
    # High-level PufferL setup
    args = pufferl.load_config(env_name)
    for key in env_params: 
        args['env'][key] = env_params[key]
    
    args['train']['env'] = env_name
    args['package'] = 'ocean' 
    args['train']['entropy_coeff'] = 0.1

    vecenv = pufferl.load_env(env_name, args)
    policy = pufferl.load_policy(args, vecenv, env_name)
    trainer = pufferl.PuffeRL(args['train'], vecenv, policy)

    while trainer.epoch < trainer.total_epochs:
        trainer.train()
    
    torch.save(policy.state_dict(), "ppo_option_policy.pt")
    trainer.close()
    return policy

# ==========================================
# 3. BENCHMARKING (Corrected LSTM State)
# ==========================================
def benchmark(ppo_policy, lsmc_model, env_params, env_name='puffer_option_pricing'):
    test_seeds = range(1000, 1500)
    results = {"PPO": [], "LSMC": []}
    
    # Create a single instance for benchmarking
    env_creator = pufferlib.ocean.env_creator(env_name)
    env = env_creator(**env_params)
    
    ppo_policy.eval()
    device = next(ppo_policy.parameters()).device

    print(f"Benchmarking {len(test_seeds)} paths...")
    for seed in test_seeds:
        # PPO Path
        obs, _ = env.reset(seed=seed)
        done, ppo_reward = False, 0
        
        # LSTMCell expects 2D tensors: (batch_size, hidden_size)
        hidden_size = ppo_policy.lstm.hidden_size
        state = {
            'lstm_h': torch.zeros(1, hidden_size).to(device),  # 2D: (batch, hidden)
            'lstm_c': torch.zeros(1, hidden_size).to(device),  # 2D: (batch, hidden)
            'reward': torch.zeros(1).to(device),
            'done': torch.zeros(1).to(device),
            'env_id': slice(0, 1),
            'mask': np.array([True])
        }

        while not done:
            with torch.no_grad():
                obs_tensor = torch.as_tensor(obs).float().unsqueeze(0).to(device)
                logits, _ = ppo_policy.forward_eval(obs_tensor, state)
                action, _, _ = pufferlib.pytorch.sample_logits(logits)
            
            obs, reward, done, _, _ = env.step(action.cpu().numpy())
            
            # Handle numpy array outputs from env
            if hasattr(reward, '__len__'):
                reward = float(reward[0])
            if hasattr(done, '__len__'):
                done = bool(done[0])
                
            ppo_reward += reward
            
            state['reward'][0] = reward
            state['done'][0] = float(done)

        results["PPO"].append(ppo_reward)
        results["LSMC"].append(lsmc_model.run_lsmc_decision(env.paths))

    print(f"\nResults:\nPPO Mean: {np.mean(results['PPO']):.4f}\nLSMC Mean: {np.mean(results['LSMC']):.4f}")

# ==========================================
# 4. EXECUTION
# ==========================================
if __name__ == "__main__":
    env_name = 'puffer_option_pricing'
    env_params = {'n_processes': 200, 'n_rights': 15, 'max_steps': 40, 'delta': 3}
    
    # 1. Initialize Baseline Model
    lsmc_model = RecursiveLSMC(env_params['n_rights'], env_params['max_steps'], 
                               env_params['delta'], env_params['n_processes'], 
                               strike=100, r_rate=0.05)
    
    # 2. Get Env for LSMC Training
    env_creator = pufferlib.ocean.env_creator(env_name)
    train_env = env_creator(**env_params)
    lsmc_model.train(train_env, n_train_paths=5000)

    # 3. Train PPO
    ppo_policy = ppo_trainer(env_name=env_name, env_params=env_params)

    # 4. Run Benchmark
    benchmark(ppo_policy, lsmc_model, env_params, env_name=env_name)