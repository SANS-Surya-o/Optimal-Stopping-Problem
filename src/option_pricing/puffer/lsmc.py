# lsmc.py
import numpy as np
import torch
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

class RecursiveLSMC:
    def __init__(self, n_rights, max_steps, delta, n_processes, strike, r_rate, degree=2):
        self.L = n_rights
        self.T = max_steps
        self.delta = delta
        self.M = n_processes
        self.K = strike
        self.r = r_rate
        self.degree = degree
        self.models = {} 

    def _get_features(self, prices):
        """Polynomial basis functions for multi-dimensional price inputs."""
        # prices shape: (N_samples, M_assets)
        poly = PolynomialFeatures(degree=self.degree, include_bias=False)
        return poly.fit_transform(prices)

    def calculate_payoff_at_t(self, prices, t):
        """Internal payoff calculation, independent of the Env."""
        # Handle both (N, M) and (M,) shapes
        if prices.ndim == 1:
            avg_price = np.mean(prices)
        else:
            avg_price = np.mean(prices, axis=1)
        
        payoff = np.maximum(self.K - avg_price, 0)
        # Match env discount: e^(-r * t / T)
        discount = np.exp(-self.r * (t / self.T))
        return payoff * discount

    def train(self, env, n_train_paths):
        """Trains using synchronized env.reset() paths and backward induction."""
        print(f"Generating {n_train_paths} paths for LSMC training...")
        all_paths = []
        for _ in range(n_train_paths):
            env.reset() 
            all_paths.append(env.paths.copy())
        
        paths = np.array(all_paths) # Shape: (N, T+1, M)
        N = paths.shape[0]
        
        # V[path, time, rights_remaining]
        V = np.zeros((N, self.T + 1, self.L + 1))
        dt = 1.0 / self.T
        
        # Recursive Cascade
        for l in range(1, self.L + 1):
            # Terminal payoff at T
            V[:, self.T, l] = self.calculate_payoff_at_t(paths[:, self.T], self.T) + V[:, self.T, l-1]

            # Backward Induction
            for t in range(self.T - 1, -1, -1):
                t_delta = min(t + self.delta, self.T)
                payoff_t = self.calculate_payoff_at_t(paths[:, t], t)
                
                # Value if we stop now: Payoff + Discounted Future Rights
                future_rights_val = V[:, t_delta, l-1] * np.exp(-self.r * (dt * self.delta))
                stop_val = payoff_t + future_rights_val
                
                # Value if we continue: Discounted E[V(t+1, l)]
                continuation_target = V[:, t+1, l] * np.exp(-self.r * dt)
                
                # Regression on ITM paths
                itm = payoff_t > 0
                if np.sum(itm) > (self.M * self.degree + 10):
                    X = self._get_features(paths[itm, t])
                    y = continuation_target[itm]
                    
                    model = LinearRegression().fit(X, y)
                    self.models[(l, t)] = model
                    
                    pred_cont = model.predict(X)
                    V[itm, t, l] = np.where(stop_val[itm] > pred_cont, stop_val[itm], continuation_target[itm])
                
                V[~itm, t, l] = continuation_target[~itm]
        
        print(f"LSMC Training Complete. {len(self.models)} boundaries learned.")

    def run_lsmc_decision(self, test_path):
        """Evaluation logic for a single synchronized test path."""
        rights_remaining = self.L
        refractory_timer = 0
        total_payoff = 0
        
        for t in range(self.T + 1):
            if rights_remaining == 0:
                break
            if refractory_timer > 0:
                refractory_timer -= 1
                continue
            
            payoff_t = self.calculate_payoff_at_t(test_path[t], t)
            
            # Check regression boundary
            if payoff_t > 0 and (rights_remaining, t) in self.models:
                X = self._get_features(test_path[t].reshape(1, -1))
                expected_continuation = self.models[(rights_remaining, t)].predict(X)[0]
                
                if payoff_t > expected_continuation:
                    total_payoff += payoff_t
                    rights_remaining -= 1
                    refractory_timer = self.delta
        
        return total_payoff