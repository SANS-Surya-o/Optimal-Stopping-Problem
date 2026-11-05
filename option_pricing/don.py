# %%
import numpy as np
import matplotlib.pyplot as plt
import time
from dataclasses import dataclass
import torch
np.random.seed(42)

@dataclass
class Params:
    S0: float = 100.0
    K: float = 100.0
    T: float = 1.0
    r: float = 0.05
    sigma: float = 0.5
    N: int = 300    # number of exercise dates
    M: int = 1 # number of processes (1 = standard, >1 = Basket-style)
    L: int =  3  # exercise rights
    delta: int = 0    # refractory period between exercises (0 = next step allowed)
    
    @property
    def dt(self): return self.T / self.N
    
    @property
    def df(self): return np.exp(-self.r * self.dt)

params = Params()
print(f"American Put: S0={params.S0}, K={params.K}, T={params.T}, r={params.r}, σ={params.sigma}, N={params.N}, M={params.M}, L={params.L}, delta={params.delta}")
print(f"dt={params.dt:.4f}, discount_factor={params.df:.6f}")

# %% [markdown]
# ### Helpers

# %%
# ══════════════════════════════════════════════════════════════════════════════
# PATH GENERATION  — unified interface for all process types
# ══════════════════════════════════════════════════════════════════════════════
#
# generate_paths(params, n_paths, process='gbm', seed=None, randomize_S0=True)
#
# process options
# ───────────────
# 'gbm'      Geometric Brownian Motion  (continuous, log-normal increments)
#            The canonical Black-Scholes model.  Used by compute_exact_boundary_continuous.
#
# 'binomial' CRR Binomial Tree  (discrete price, discrete time)
#            At each step: S → S·u with prob p_u, S → S·d with prob p_d.
#            u = exp(σ√dt), d = 1/u, p_u = (exp(r·dt)-d)/(u-d).
#
# 'trinomial' Boyle Trinomial Tree (discrete price, discrete time)
#            At each step: S → S·u, S → S·m (flat), or S → S·d.
#            Solves the "sawtooth" boundary issue of the binomial model.
#
# 'iid_gaussian'
#            Each log-return is drawn i.i.d. N(μ·dt, σ²·dt).
#
# All variants return shape (n_paths, N+1, M) and support randomize_S0.
# M>1 is only meaningful for 'gbm' and 'iid_gaussian'.

import numpy as np

def _crr_params(params):
    """CRR up/down factors and risk-neutral probabilities for Binomial Tree."""
    dt = params.dt
    u = np.exp(params.sigma * np.sqrt(dt))
    d = 1.0 / u
    p_u = (np.exp(params.r * dt) - d) / (u - d)
    assert 0 < p_u < 1, f"p_u={p_u:.4f} outside (0,1) — reduce sigma or increase N"
    return u, d, p_u, 1.0 - p_u

def _trinomial_params(params):
    """Boyle (1986) parameters for Trinomial Tree."""
    dt = params.dt
    # lambda is typically sqrt(3) to ensure probabilities stay positive and stable
    lam = np.sqrt(3.0) 
    
    u = np.exp(lam * params.sigma * np.sqrt(dt))
    d = 1.0 / u
    m = 1.0
    
    nu = params.r - 0.5 * params.sigma**2
    
    p_u = 1.0 / (2.0 * lam**2) + (nu * np.sqrt(dt)) / (2.0 * lam * params.sigma)
    p_d = 1.0 / (2.0 * lam**2) - (nu * np.sqrt(dt)) / (2.0 * lam * params.sigma)
    p_m = 1.0 - p_u - p_d
    
    assert 0 < p_u < 1 and 0 < p_d < 1 and 0 < p_m < 1, "Trinomial probs outside (0,1). Check parameters."
    
    return u, m, d, p_u, p_m, p_d

def generate_paths(params, n_paths, process='gbm', seed=None, randomize_S0=False):
    """
    Generate Monte Carlo paths.

    Parameters
    ----------
    params       : Params dataclass
    n_paths      : int
    process      : 'gbm' | 'binomial' | 'trinomial' | 'iid_gaussian'
    seed         : int or None
    randomize_S0 : bool  — scatter S0 in [0.4K, 1.6K] for global policy learning

    Returns
    -------
    paths : np.ndarray, shape (n_paths, N+1, M)
    """
    rng = np.random.default_rng(seed)
    N, M, K = params.N, params.M, params.K

    # Starting prices
    if randomize_S0:
        S0 = rng.uniform(K * 0.4, K * 1.6, (n_paths, M))
    else:
        S0 = np.full((n_paths, M), params.S0)

    paths = np.zeros((n_paths, N + 1, M))
    paths[:, 0, :] = S0

    if process == 'gbm':
        dt = params.dt
        drift     = (params.r - 0.5 * params.sigma**2) * dt
        diffusion = params.sigma * np.sqrt(dt)
        Z = rng.standard_normal((n_paths, N, M))
        log_ret = np.cumsum(drift + diffusion * Z, axis=1)   # (n, N, M)
        paths[:, 1:, :] = S0[:, np.newaxis, :] * np.exp(log_ret)

    elif process == 'binomial':
        if M != 1:
            raise ValueError("binomial process only supports M=1")
        u, d, p_u, p_d = _crr_params(params)
        # Draw Bernoulli up/down steps
        ups = rng.random((n_paths, N)) < p_u      # True = up move
        for t in range(N):
            paths[:, t+1, 0] = np.where(ups[:, t],
                                         paths[:, t, 0] * u,
                                         paths[:, t, 0] * d)
                                         
    elif process == 'trinomial':
        if M != 1:
            raise ValueError("trinomial process only supports M=1")
        u, m, d, p_u, p_m, p_d = _trinomial_params(params)
        
        # Fast vectorized generation: draw step multipliers and cumulative product
        step_multipliers = rng.choice(
            a=[u, m, d], 
            p=[p_u, p_m, p_d], 
            size=(n_paths, N)
        )
        paths[:, 1:, 0] = S0[:, 0, np.newaxis] * np.cumprod(step_multipliers, axis=1)

    elif process == 'iid_gaussian':
        # Each step i.i.d.  — NOT cumulative, so no autocorrelation across steps
        dt = params.dt
        drift     = (params.r - 0.5 * params.sigma**2) * dt
        diffusion = params.sigma * np.sqrt(dt)
        Z = rng.standard_normal((n_paths, N, M))
        log_steps = drift + diffusion * Z
        paths[:, 1:, :] = S0[:, np.newaxis, :] * np.exp(np.cumsum(log_steps, axis=1))

    else:
        raise ValueError(f"Unknown process '{process}'. Choose gbm | binomial | trinomial | iid_gaussian")

    return paths


# ── Quick smoke-tests ─────────────────────────────────────────────────────────
_p = params   # uses the Params() defined in cell 0
train_paths = generate_paths(_p, 50,  process='gbm',       seed=42)
test_paths  = generate_paths(_p, 100, process='gbm',       seed=997)
_bin        = generate_paths(_p, 10,  process='binomial',  seed=0)
_tri        = generate_paths(_p, 10,  process='trinomial', seed=0)
_iid        = generate_paths(_p, 10,  process='iid_gaussian', seed=0)

print(f"GBM train: {train_paths.shape},  test: {test_paths.shape}")
print(f"Binomial sample:      {_bin.shape},  prices[0,0..5,0] = {_bin[0,:6,0].round(2)}")
print(f"Trinomial sample:     {_tri.shape},  prices[0,0..5,0] = {_tri[0,:6,0].round(2)}")
print(f"IID-Gaussian sample:  {_iid.shape},  prices[0,0..5,0] = {_iid[0,:6,0].round(2)}")

# %%
import numpy.polynomial.laguerre as lag

# ── Modular payoff ──
def payoff_put(prices, K):
    """Put payoff on mean price across processes.
    prices: (..., M) → returns (...)
    M=1: standard put. Basket options
    """
    avg = prices.mean(axis=-1)  # mean across M processes
    return np.maximum(K - avg, 0)


def discounted_payoff(paths_t, t, params):
    """Discounted payoff at time step t. paths_t shape: (n_paths, M) or (M,)"""
    return payoff_put(paths_t, params.K) * params.df**t



import itertools
import numpy.polynomial.laguerre as lag

def poly_basis(prices, K, degree=3, basis_type='laguerre'):
    """
    Generate basis features for regression.
    prices: shape (n_paths, M) -> Returns: (n_paths, n_features)
    """
    s = prices / K
    n_paths, M = s.shape
    features = [np.ones(n_paths)]  # Global intercept
    
    # 1. Individual Asset Terms
    for m in range(M):
        if basis_type == 'laguerre':
            L_vals = lag.lagvander(s[:, m], degree)
            features.append(L_vals[:, 1:])  # Skip L_0
        elif basis_type == 'monomial':
            for d in range(1, degree + 1):
                features.append(s[:, m]**d)
                
    # 2. Spatial Cross-Terms (Asset Covariance)
    if M > 1:
        # Generate pairwise products for all unique asset combinations
        for i, j in itertools.combinations(range(M), 2):
            features.append(s[:, i] * s[:, j])
            
    return np.column_stack(features)


# ── Pointwise V query ──
def query_V(models, prices_1d, l, t, params, degree=3):
    """Return V(l, t) for a SINGLE state (prices_1d has shape (M,)).
    Uses the already-trained regressor models[(l, t)].
    
    - l == 0 → 0
    - t > N  → 0  (past maturity)
    - t == N → terminal_value(l, prices)
    - t < N, model exists → max(exercise_val, cont_est)
    """
    if l == 0 or t > params.N:
        return 0.0
    p = payoff_put(prices_1d.reshape(1, -1), params.K).item()
    if t == params.N:
        # At maturity: exercise 1 right; if delta allows and l > 1,
        # remaining rights are wasted (can't exercise past N)
        return p
    # Look up continuation estimate from regressor
    key = (l, t)
    if key in models:
        x = poly_basis(prices_1d.reshape(1, -1), params.K, degree)[0]
        cont_est = float(x @ models[key])
    else:
        cont_est = 0.0  # no model → assume 0 continuation
    return max(p, cont_est)

def query_V_batch(models, prices, l, t, params, degree=3):
    """Computes the expected value of holding l rights at state S_t."""
    prices_2d = prices[:,t]
    n = len(prices_2d)
    
    # Value is 0 if no rights left or past maturity
    if l == 0 or t > params.N:  
        return np.zeros(n)
        
    p = payoff_put(prices_2d, params.K)
    
    # At expiration, value is purely the immediate payoff
    if t == params.N:
        return p

    key_cont = (l, t, 'cont')
    key_ex = (l, t, 'ex')
    
    # Return the max of expected exercise vs expected continuation
    if key_cont in models:
        X = poly_basis(prices_2d, params.K, degree)
        cont_est = X @ models[key_cont]
        ex_est = p + (X @ models[key_ex] if l > 1 else 0.0)
        return np.maximum(ex_est, cont_est)
        
    return np.zeros(n)

print(f"Payoff test (M={params.M}): put on S=95 → {payoff_put(np.array([[95.0]]), params.K).item():.2f}")
print(f"Basis shape for 5 samples, M={params.M}: {poly_basis(np.ones((5, params.M))*100, params.K).shape}")
print(f"query_V with empty models: {query_V({}, np.array([95.0]*params.M), 1, 10, params):.2f}")




# %%
def make_state(prices, t, rights_left, cooldown, params):
    """
    Normalized state vector: [prices/K, tau/N, rights/L, cooldown/delta]
    tau = N - t  (time REMAINING, not elapsed).
    The exercise boundary is a function of time remaining, so this encoding
    makes the polynomial approximation far more natural.
    """
    prices_norm = prices / params.K
    tau_norm    = (params.N - t) / params.N          # ← KEY CHANGE
    cd_norm     = cooldown / params.delta if params.delta > 0 else 0.0
    return np.concatenate([
        prices_norm,
        [tau_norm, rights_left / params.L, cd_norm]
    ]).astype(np.float32)


def fqi_state_features(state_vec, degree=3, basis_type='laguerre'):
    """
    Polynomial feature map for the augmented state vector.
    state_vec: shape (M + 3,)  →  [prices/K ..., tau/N, rights/L, cd/delta]
 
    Changes vs. original:
      - time component is now tau (time remaining), not t (elapsed).
        The feature meaning is unchanged; the encoding is just more natural.
      - cross-term uses tau instead of t_norm.
    """
    features = [1.0]   # intercept
 
    M         = len(state_vec) - 3
    prices    = state_vec[:M]          # already S/K from make_state
    tau_norm  = state_vec[-3]          # (N-t)/N  ∈ [0, 1]
    rights_norm = state_vec[-2]
    cd_norm   = state_vec[-1]
 
    # Intrinsic value (normalised)
    avg_price = np.mean(prices)
    intrinsic = max(1.0 - avg_price, 0.0)
    features.append(intrinsic)
 
    # 1. Per-asset Laguerre / monomial terms
    if basis_type == 'laguerre':
        for p in prices:
            L_vals = lag.lagval(p, np.eye(degree + 1))
            features.extend(L_vals[1:])       # skip L_0 = 1 (duplicate of intercept)
    elif basis_type == 'monomial':
        for p in prices:
            for d in range(1, degree + 1):
                features.append(p ** d)
 
    # 2. Scalar state dimensions: tau, rights, cooldown
    for d in range(1, degree + 1):
        features.append(tau_norm   ** d)
        features.append(rights_norm ** d)
        features.append(cd_norm    ** d)
 
    # 3. Cross-terms for basket options
    if M > 1:
        for i, j in itertools.combinations(range(M), 2):
            features.append(prices[i] * prices[j])
 
    # 4. Spatio-temporal & mechanic interactions
    features.append(avg_price * tau_norm)      # Mean Price × Time-to-maturity
    features.append(rights_norm * cd_norm)     # Rights × Cooldown
 
    return np.array(features, dtype=np.float64)

# %%
# ═══════════════════════════════════════════════════════════════════════════════
# UNIFIED BOUNDARY EXTRACTION
# Works identically for LSM/TD-LSM (regression), FQI/LSPI (Q-coeffs), DQN/CQL/PPO (neural)
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# UNIFIED BOUNDARY EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def make_decision_fn(algo_type, model, params, degree=3):
    """
    Returns a callable: f(price_1d: np.ndarray, t: int, rights: int) -> str ('E' or 'H')
    """
    if algo_type == 'lsm':
        def fn(price_1d, t, rights, cooldown=0):
            key_cont = (rights, t, 'cont')
            if key_cont not in model:
                return 'H'
                
            p = payoff_put(price_1d.reshape(1, -1), params.K).item()
            if p <= 0:
                return 'H'
                
            x = poly_basis(price_1d.reshape(1, -1), params.K, degree)[0]
            
            # 1. Expected Continuation
            cont_est = float(x @ model[key_cont])
            
            # 2. Expected Future Rights (safely evaluated at time t)
            key_ex = (rights, t, 'ex')
            expected_residual = float(x @ model[key_ex]) if rights > 1 else 0.0
            
            ex_val = p + expected_residual
            return 'E' if ex_val > cont_est else 'H'

    elif algo_type == 'fqi':
        def fn(price_1d, t, rights, cooldown=0):
            if rights not in model or rights <= 0:
                return 'H'
            state = make_state(price_1d, t, rights, cooldown, params)
            features = fqi_state_features(state, degree)
            # Lookup using the specific rights level
            q0 = features @ model[rights][0]
            q1 = features @ model[rights][1]
            return 'E' if q1 > q0 else 'H'

    elif algo_type == 'lspi':
        def fn(price_1d, t, rights, cooldown=0):
            state = make_state(price_1d, t, rights, cooldown, params)
            action = model.predict(state, rights)
            return 'E' if action == 1 else 'H'

    elif algo_type in ('dqn', 'cql_net'):
        q_net = model if algo_type == 'dqn' else model
        q_net.eval()
        
        # DYNAMICALLY GET THE MODEL'S DEVICE
        model_device = next(q_net.parameters()).device

        def fn(price_1d, t, rights, cooldown=0):
            # HARD CHECK
            if payoff_put(price_1d.reshape(1, -1), params.K).item() <= 0:
                return 'H'

            state = make_state(price_1d, t, rights, cooldown, params=params)
            with torch.no_grad():
                # SEND TENSOR TO MODEL'S DEVICE
                q = q_net(torch.tensor(state, dtype=torch.float32, device=model_device))
            return 'E' if q.argmax().item() == 1 else 'H'

    elif algo_type == 'ppo':
        model.eval()
        
        # DYNAMICALLY GET THE MODEL'S DEVICE
        model_device = next(model.parameters()).device

        def fn(price_1d, t, rights, cooldown=0):
            # HARD CHECK
            if payoff_put(price_1d.reshape(1, -1), params.K).item() <= 0:
                return 'H'

            state = make_state(price_1d, t, rights, cooldown, params=params)
            with torch.no_grad():
                # SEND TENSOR TO MODEL'S DEVICE
                logits, _ = model(torch.tensor(state, dtype=torch.float32, device=model_device))
            return 'E' if logits.argmax().item() == 1 else 'H'
        
    elif algo_type == 'lsm_net':
        def fn(price_1d, t, rights, cooldown=0):
            key_cont = (rights, t, 'cont')
            if key_cont not in model:
                return 'H'
                
            p = payoff_put(price_1d.reshape(1, -1), params.K).item()
            if p <= 0:
                return 'H'
                
            X_norm = price_1d / params.K
            if X_norm.ndim == 1: X_norm = X_norm.reshape(-1, 1)
            
            # FIX: Correctly extract the device from the specific PyTorch network
            model_device = next(model[key_cont].parameters()).device
            X_tensor = torch.tensor(X_norm, dtype=torch.float32, device=model_device)

            with torch.no_grad():
                cont_est = model[key_cont](X_tensor).item()
                
                key_ex = (rights, t, 'ex')
                expected_residual = model[key_ex](X_tensor).item() if rights > 1 else 0.0
            
            ex_val = p + expected_residual
            return 'E' if ex_val > cont_est else 'H'

    else:
        raise ValueError(f"Unknown algo_type: {algo_type}")

    return fn


def find_boundary_generic(decision_fn, t, rights, params, n_scan=300):
    """
    Robust boundary extraction using a Bottom-Up scan.
    Starts deep ITM (near 0) and walks UP to the strike K.
    Returns the highest price BEFORE the network switches to 'H'.
    This completely filters out noisy 'E' spikes near the money for Deep RL.
    """
    # 1. Reverse the scan to go from Bottom (0.01) UP to Top (0.99)
    prices = np.linspace(params.K * 0.01, params.K * 0.99, n_scan)
    
    last_exercise = np.nan
    
    for price in prices:
        # Stop scanning if we accidentally wander Out-Of-The-Money
        if payoff_put(np.array([[price]]), params.K).item() <= 0:
            break
            
        decision = decision_fn(np.array([price]), t, rights)
        
        if decision == 'E':
            # As long as it says Exercise, keep pushing the boundary higher
            last_exercise = price
            
    return last_exercise


# ═══════════════════════════════════════════════════════════════════════════════
# UNIFIED POLICY PRINTER
# Call after training any algorithm to visualise the learned decision surface
# ═══════════════════════════════════════════════════════════════════════════════

def print_policy_snapshot(decision_fn, params, algo_name="",
                          rights_levels=None, t_stride=None,
                          price_levels=None, n_prices=10):
    """
    Print an ASCII grid of the exercise decision surface.

    Rows = time steps (sampled at t_stride).
    Cols = price levels (n_prices uniformly from 0.4K to K).
    Cell = 'E' (exercise) or '·' (hold), per rights level.

    Deliberately prints for EACH rights level so the multi-right
    structure of the policy is visible.

    Parameters
    ----------
    decision_fn : callable(price_1d, t, rights) -> 'E'/'H'
    params      : Params
    algo_name   : str, used in header
    rights_levels : list of ints, default [1, ..., L]
    t_stride    : int, print every t_stride-th timestep; default N//20
    price_levels: 1-D array of prices to scan; if None, use n_prices uniform
    n_prices    : int, number of price columns
    """
    return
    ## Useless piece of shit

# %%
def make_state_batch(prices, t, rights, cooldowns, params):
    """
    Vectorized state builder — shape (B, M+3).
    Uses time-to-maturity consistent with the scalar make_state above.
    """
    B = len(prices)
    prices_norm = prices / params.K
    tau_norm    = np.full((B, 1), (params.N - t) / params.N)  
    r_arr       = (rights / params.L).reshape(B, 1)
    cd_norm     = np.where(params.delta > 0, cooldowns / params.delta, 0.0).reshape(B, 1)
    return np.hstack([prices_norm, tau_norm, r_arr, cd_norm]).astype(np.float32)




# %%
def fqi_state_features_batch(state_matrix, degree=3, basis_type='laguerre'):
    """
    state_matrix: (n, M+3)  →  returns (n, n_features)
    Vectorised equivalent of fqi_state_features; avoids the Python loop over rows.
    """
    n    = state_matrix.shape[0]
    M    = state_matrix.shape[1] - 3
    prices     = state_matrix[:, :M]          # (n, M)
    tau_norm   = state_matrix[:, -3]          # (n,)
    rights_norm = state_matrix[:, -2]         # (n,)
    cd_norm    = state_matrix[:, -1]          # (n,)
 
    cols = [np.ones(n)]                        # intercept
 
    avg_price = prices.mean(axis=1)
    intrinsic = np.maximum(1.0 - avg_price, 0.0)
    cols.append(intrinsic)
 
    if basis_type == 'laguerre':
        for m in range(M):
            p_col = prices[:, m]
            # lagvander gives the Vandermonde matrix of shape (n, degree+1)
            L_vander = lag.lagvander(p_col, degree)   # (n, degree+1)
            cols.append(L_vander[:, 1:])              # skip L_0
    elif basis_type == 'monomial':
        for m in range(M):
            for d in range(1, degree + 1):
                cols.append(prices[:, m] ** d)
 
    for d in range(1, degree + 1):
        cols.append(tau_norm    ** d)
        cols.append(rights_norm ** d)
        cols.append(cd_norm     ** d)
 
    if M > 1:
        for i, j in itertools.combinations(range(M), 2):
            cols.append(prices[:, i] * prices[:, j])
 
    cols.append(avg_price * tau_norm)
    cols.append(rights_norm * cd_norm)
 
    # Stack — handle both 1-D and 2-D columns uniformly
    arrays = []
    for c in cols:
        if np.ndim(c) == 1:
            arrays.append(c.reshape(n, 1))
        else:
            arrays.append(c)
    return np.hstack(arrays).astype(np.float64)

# %%
def collect_offline_transitions(paths, params, behavior_fn=None, epsilon=0.2):
    """
    Collect (s, a, r, s', done, l, l') transitions.
 
    Improvements over the original collect_transitions:
      1. Uses a *guided* behavior policy (e.g. from a pre-trained LSM model)
         so that transitions are concentrated near the true exercise boundary.
         Falls back to ε-greedy random if behavior_fn is None.
      2. epsilon-greedy exploration (default 20 %) ensures every action is
         still visited even when behavior_fn is provided.
      3. ALL time steps 1…N are visited for EVERY path regardless of early
         exercise, achieved by running a *separate* rights-tracking loop per
         rights level.  This avoids the undersampling of late-time / low-rights
         transitions that plagues the original per-path loop.
 
    Parameters
    ----------
    paths        : np.ndarray, shape (n_paths, N+1, M)
    params       : Params
    behavior_fn  : callable(price_1d, t, rights, cooldown) → 'E'/'H', or None
    epsilon      : float  exploration rate for ε-greedy
 
    Returns
    -------
    Same tuple structure as the original:
    (states, actions, rewards, next_states, dones, rights_arr, next_rights_arr)
    as concatenated np.ndarray objects.
    """
    N, K, L, delta = params.N, params.K, params.L, params.delta
    rng = np.random.default_rng(42)
 
    all_s, all_a, all_r, all_sp, all_d, all_l, all_lp = ([] for _ in range(7))
 
    for path in paths:
        # Run one episode PER starting-rights level so every (l, t) cell is
        # covered even when the path tends to exercise early.
        for l_start in range(1, L + 1):
            t, rights, cooldown = 1, l_start, 0
 
            while t <= N and rights > 0:
                state = make_state(path[t], t, rights, cooldown, params)
 
                # ── Behavior policy ───────────────────────────────────────────
                if cooldown > 0:
                    action = 0                           # forced hold in cooldown
                elif behavior_fn is not None and rng.random() > epsilon:
                    # Use guided policy
                    decision = behavior_fn(path[t], t, rights, cooldown)
                    action   = 1 if decision == 'E' else 0
                else:
                    # Pure random (or epsilon-greedy exploration)
                    action = int(rng.integers(0, 2))
 
                reward      = 0.0
                next_rights = rights
                next_cooldown = max(0, cooldown - 1)
 
                if action == 1:
                    reward       = payoff_put(path[t].reshape(1, -1), K).item()
                    next_rights -= 1
                    next_cooldown = delta
 
                next_t = t + 1
                done   = (next_t > N) or (next_rights <= 0)
 
                # Terminal auto-exercise for any remaining rights at maturity
                if next_t > N and next_rights > 0 and action == 0:
                    pay = payoff_put(path[N].reshape(1, -1), K).item()
                    if pay > 0:
                        reward += pay
 
                next_state = make_state(
                    path[min(next_t, N)], min(next_t, N),
                    max(next_rights, 0), next_cooldown, params
                )
 
                all_s.append(state);  all_a.append(action)
                all_r.append(reward); all_sp.append(next_state)
                all_d.append(done);   all_l.append(rights)
                all_lp.append(next_rights)
 
                t      += 1
                rights  = next_rights
                cooldown = next_cooldown
 
    return (
        np.array(all_s,  dtype=np.float32),
        np.array(all_a,  dtype=np.int32),
        np.array(all_r,  dtype=np.float64),
        np.array(all_sp, dtype=np.float32),
        np.array(all_d,  dtype=bool),
        np.array(all_l,  dtype=np.int32),
        np.array(all_lp, dtype=np.int32),
    )

# %% [markdown]
# ### Model Definitions

# %%
def train_lsm(paths, params, degree=3):

    """
    Standard LSM with multiple stopping rights.
    
    Bellman: V(l,t) = max( payoff(t) + df^(t_resume-t) · V(l-1, t_resume), df · V(l, t+1) )
    where t_resume = min(t + delta + 1, N).  (delta=0 → next step)
    
    V(l-1, t_resume) is queried via regressors already trained for layer l-1.
    Regression target = realized continuation = df · V(l, t+1).
    """
    N, df, K, L, delta = params.N, params.df, params.K, params.L, params.delta
    n_paths = len(paths)
    payoffs = np.array([payoff_put(paths[:, t], K) for t in range(N + 1)]).T
    
    models = {}
    
    for l in range(1, L + 1):
        V = payoffs[:, N].copy()
        
        for t in range(N - 1, 0, -1):
            realized_cont = df * V
            
            # Enforce refractory lockout past expiration
            if t + delta + 1 > N:
                realized_residual = np.zeros(n_paths)
            else:
                t_resume = t + delta + 1
                residual = query_V_batch(models, paths, l - 1, t_resume, params, degree)
                realized_residual = (df ** (t_resume - t)) * residual
            
            p_t = payoffs[:, t]
            itm = p_t > 0
            
            if itm.sum() < degree * params.M + 5:
                V = realized_cont
                continue
            
            X_itm = poly_basis(paths[itm, t], K, degree)
            
            # Regression 1: Expected Continuation
            coeffs_cont = np.linalg.lstsq(X_itm, realized_cont[itm], rcond=None)[0]
            models[(l, t, 'cont')] = coeffs_cont
            cont_est = X_itm @ coeffs_cont
            
            # Regression 2: Expected Residual (if we exercise)
            if l > 1:
                coeffs_ex = np.linalg.lstsq(X_itm, realized_residual[itm], rcond=None)[0]
                models[(l, t, 'ex')] = coeffs_ex
                expected_residual_est = X_itm @ coeffs_ex
            else:
                expected_residual_est = np.zeros(itm.sum())
                
            expected_ex_val = p_t[itm] + expected_residual_est
            
            # Decision is based purely on expected values (No look-ahead bias)
            exercise = expected_ex_val > cont_est
            
            # Update V tracking array using realized paths
            V_new = realized_cont.copy()
            realized_ex_val = p_t[itm] + realized_residual[itm]
            V_new[itm] = np.where(exercise, realized_ex_val, realized_cont[itm])
            V = V_new

    return models

# %%
def train_td_lsm(paths, params, degree=3):
    N, df, K, L, delta = params.N, params.df, params.K, params.L, params.delta
    n_paths = len(paths)
    payoffs = np.array([payoff_put(paths[:, t], K) for t in range(N + 1)]).T
    models = {}
    
    for l in range(1, L + 1):
        V_hat = payoffs[:, N].copy()
        
        for t in range(N - 1, 0, -1):
            td_target_cont = df * V_hat
            
            if t + delta + 1 > N:
                td_target_residual = np.zeros(n_paths)
            else:
                t_resume = t + delta + 1
                residual = query_V_batch(models, paths, l - 1, t_resume, params, degree)
                td_target_residual = (df ** (t_resume - t)) * residual
            
            p_t = payoffs[:, t]
            itm = p_t > 0
            
            if itm.sum() < degree * params.M + 5:
                V_hat = np.maximum(p_t + td_target_residual, td_target_cont)
                continue
            
            X_itm = poly_basis(paths[itm, t], K, degree)
            
            coeffs_cont = np.linalg.lstsq(X_itm, td_target_cont[itm], rcond=None)[0]
            models[(l, t, 'cont')] = coeffs_cont
            
            if l > 1:
                coeffs_ex = np.linalg.lstsq(X_itm, td_target_residual[itm], rcond=None)[0]
                models[(l, t, 'ex')] = coeffs_ex
            
            # TD Bootstrap: Update V_hat for ALL paths using the regressor
            X_all = poly_basis(paths[:, t], K, degree)
            cont_est_all = X_all @ coeffs_cont
            ex_est_all = p_t + (X_all @ coeffs_ex if l > 1 else 0.0)
            
            V_hat_new = td_target_cont.copy()
            V_hat_new[itm] = np.maximum(ex_est_all[itm], cont_est_all[itm])
            V_hat = V_hat_new
            
    return models

# %% [markdown]
# #### Transisiton colector for rl algos

# %%
def collect_transitions(paths, params):
    """
    Generate valid (s, a, r, s', done, l, l') transitions for off-policy algorithms.
    Uses a random behavior policy to explore the state space.
    """
    N, K, L, delta = params.N, params.K, params.L, params.delta
    transitions = []
    rng = np.random.default_rng(42)
    
    for path in paths:
        t, rights, cooldown = 1, L, 0
        
        while t <= N and rights > 0:
            state = make_state(path[t], t, rights, cooldown, params)
            
            # Action masking: if in cooldown, we are forced to hold (0)
            valid_actions = [0] if cooldown > 0 else [0, 1]
            action = rng.choice(valid_actions)
            
            reward = 0.0
            next_rights = rights
            next_cooldown = max(0, cooldown - 1)
            
            if action == 1:
                reward = payoff_put(path[t].reshape(1, -1), K).item()
                next_rights -= 1
                next_cooldown = delta
                
            next_t = t + 1
            done = (next_t > N) or (next_rights <= 0)
            
            # Terminal auto-exercise
            if next_t > N and next_rights > 0 and action == 0:
                pay = payoff_put(path[N].reshape(1, -1), K).item()
                if pay > 0:
                    reward += pay
                    
            next_state = make_state(path[min(next_t, N)], min(next_t, N), 
                                    max(next_rights, 0), next_cooldown, params)
                                    
            transitions.append((state, action, reward, next_state, done, rights, next_rights))
            
            t += 1
            rights = next_rights
            cooldown = next_cooldown
            
    return transitions

# %%
# ═══════════════════════════════════════════════════════════════════════════════
# DQN Implementation (PyTorch)
# ═══════════════════════════════════════════════════════════════════════════════
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self, state_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # 2 actions: hold or exercise
        )
    
    def forward(self, x):
        return self.net(x)


from torch.utils.data import TensorDataset, DataLoader

def train_dqn(paths, params, hidden_dim=64, lr=1e-3, n_epochs=15, batch_size=512, target_update=5):
    """Batched, GPU-optimized DQN using an offline replay dataset."""
    df = params.df
    state_dim = params.M + 3
    
    q_net = DQN(state_dim, hidden_dim).to(device)
    target_net = DQN(state_dim, hidden_dim).to(device)
    target_net.load_state_dict(q_net.state_dict())
    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    
    # 1. Collect Data instantly
    S, A, R, S_next, D, _, _ = collect_offline_transitions(paths, params)
    
    # 2. Push entire dataset to GPU memory (if it fits, which it should easily for this size)
    dataset = TensorDataset(
        torch.tensor(S, dtype=torch.float32, device=device),
        torch.tensor(A, dtype=torch.int64, device=device),
        torch.tensor(R, dtype=torch.float32, device=device),
        torch.tensor(S_next, dtype=torch.float32, device=device),
        torch.tensor(D, dtype=torch.float32, device=device)
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 3. Train at maximum GPU throughput
    q_net.train()
    for epoch in range(n_epochs):
        for states, actions, rewards, next_states, dones in dataloader:
            
            q_values = q_net(states).gather(1, actions.unsqueeze(1)).squeeze()
            
            with torch.no_grad():
                next_q = target_net(next_states).max(1)[0]
                targets = rewards + df * next_q * (1 - dones)
                
            loss = nn.functional.mse_loss(q_values, targets)
            
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q_net.parameters(), 1.0)
            optimizer.step()
            
        if (epoch + 1) % target_update == 0:
            target_net.load_state_dict(q_net.state_dict())
            
    return q_net

# %%
# ═══════════════════════════════════════════════════════════════════════════════
# PPO Implementation (PyTorch)
# ═══════════════════════════════════════════════════════════════════════════════

import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActorCritic(nn.Module):
    """Actor-Critic network for PPO."""
    def __init__(self, state_dim, hidden_dim=64):
        super().__init__()
        # Separate pathway for Policy (Actor)
        self.actor_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )
        # Separate pathway for Value (Critic)
        self.critic_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        # Returns logits and value
        return self.actor_net(x), self.critic_net(x).squeeze(-1)
    
    def get_action(self, state, deterministic=False):
        """Sample action and return action, log_prob, value."""
        logits, value = self.forward(state)
        probs = torch.softmax(logits, dim=-1)
        
        if deterministic:
            action = probs.argmax(dim=-1)
            log_prob = torch.log(probs.gather(-1, action.unsqueeze(-1)).squeeze(-1) + 1e-8)
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
        return action, log_prob, value

def train_ppo(paths, params, hidden_dim=64, lr=3e-4, n_epochs=5, clip_epsilon=0.2, n_updates=4):
    N, K, L, delta, df = params.N, params.K, params.L, params.delta, params.df
    n_paths = len(paths)
    state_dim = params.M + 3
    
    ac = ActorCritic(state_dim, hidden_dim).to(device)
    optimizer = optim.Adam(ac.parameters(), lr=lr)
    
    for epoch in range(n_epochs):
        rights = np.full(n_paths, L, dtype=int)
        cooldowns = np.zeros(n_paths, dtype=int)
        
        ep_states, ep_actions, ep_log_probs, ep_rewards, ep_values, ep_active = [], [], [], [], [], []
        
        # 1. Batched Rollout
        for t in range(1, N + 1):
            active = rights > 0
            state_batch = make_state_batch(paths[:, t], t, rights, cooldowns, params)
            state_tensor = torch.tensor(state_batch, device=device)
            
            with torch.no_grad():
                logits, values = ac(state_tensor)
                
                # LOGIT MASKING: Force probability of Exercise to 0 if blocked
                invalid_mask = torch.tensor((cooldowns > 0) | (~active), device=device)
                logits[invalid_mask, 1] = -1e9 
                
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)
            
            actions_np = actions.cpu().numpy()
            rewards = np.zeros(n_paths, dtype=np.float32)
            exercised = (actions_np == 1) & active
            
            if exercised.any():
                rewards[exercised] = payoff_put(paths[exercised, t], K)
                rights[exercised] -= 1
                cooldowns[exercised] = delta
            
            cooldowns[~exercised] = np.maximum(0, cooldowns[~exercised] - 1)
            
            # Terminal Evaluation
            if t == N:
                terminal_active = rights > 0
                if terminal_active.any():
                    pay = payoff_put(paths[terminal_active, N], K)
                    rewards[terminal_active] += np.where(pay > 0, pay, 0.0)

            ep_states.append(state_tensor)
            ep_actions.append(actions)
            ep_log_probs.append(log_probs)
            ep_rewards.append(torch.tensor(rewards, dtype=torch.float32, device=device))
            ep_values.append(values)
            ep_active.append(torch.tensor(active, dtype=torch.bool, device=device))

        # 2. Compute Returns Vector-wise Backwards
        returns = []
        R = torch.zeros(n_paths, device=device)
        for i in reversed(range(N)):
            R = ep_rewards[i] + df * R
            returns.insert(0, R)
            
        # 3. Flatten, filter only active steps, and train
        states = torch.cat(ep_states)[torch.cat(ep_active)]
        actions = torch.cat(ep_actions)[torch.cat(ep_active)]
        old_log_probs = torch.cat(ep_log_probs)[torch.cat(ep_active)]
        returns_t = torch.cat(returns)[torch.cat(ep_active)]
        values_t = torch.cat(ep_values)[torch.cat(ep_active)]
        
        if len(states) == 0: continue
            
        advantages = returns_t - values_t
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        ac.train()
        for _ in range(n_updates):
            logits, new_values = ac(states)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            new_log_probs = dist.log_prob(actions)
            
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            value_loss = nn.functional.mse_loss(new_values, returns_t)
            loss = policy_loss + 0.5 * value_loss
            
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(ac.parameters(), 0.5)
            optimizer.step()
            
    return ac

# %%
# ═══════════════════════════════════════════════════════════════════════════════
# Fitted Q-Iteration (FQI) - Batch Q-learning for continuous state spaces
# ═══════════════════════════════════════════════════════════════════════════════


def train_fqi(paths, params, degree=3, n_iterations=50, behavior_fn=None, epsilon=0.2):
    """
    Fitted Q-Iteration with polynomial basis.
 
    Key improvements over the original:
      • Guided transition collection (behavior_fn, ε-greedy).
      • Fully vectorised Bellman target computation — no inner Python loop.
      • n_iterations=50 (was 5); now affordable after vectorisation.
      • Ridge penalty l2_reg=1e-2 (was 1e-4) for numerical stability.
      • Features built via fqi_state_features_batch for speed.
      • State uses time-to-maturity (see make_state / fqi_state_features).
 
    Parameters
    ----------
    paths        : np.ndarray, shape (n_paths, N+1, M)
    params       : Params
    degree       : int    polynomial degree for basis
    n_iterations : int    FQI outer iterations  (50 recommended)
    behavior_fn  : callable or None   guided policy for data collection
    epsilon      : float  ε-greedy exploration rate
 
    Returns
    -------
    Q_coeffs : dict  {rights_level: {0: w_hold, 1: w_exercise}}
    """
    N, df, K, L, delta = params.N, params.df, params.K, params.L, params.delta
 
    # ── 1. Collect transitions ────────────────────────────────────────────────
    states, actions, rewards, next_states, dones, rights_arr, next_rights_arr = \
        collect_offline_transitions(paths, params, behavior_fn=behavior_fn, epsilon=epsilon)
 
    # ── 2. Build feature matrices once (expensive, do it upfront) ─────────────
    # print(f"  [FQI] Building feature matrices for {len(states)} transitions …", flush=True)
    X      = fqi_state_features_batch(states.astype(np.float64),      degree)
    X_next = fqi_state_features_batch(next_states.astype(np.float64), degree)
    n_features = X.shape[1]
 
    # ── 3. Initialise Q coefficients per rights level ─────────────────────────
    l2_reg   = 1e-2          # robust ridge penalty (was 1e-4)
    Q_coeffs = {l: {0: np.zeros(n_features), 1: np.zeros(n_features)}
                for l in range(1, L + 1)}
 
    # ── 4. Train layer by layer (l=1 first so higher layers can bootstrap) ────
    for l in range(1, L + 1):
        mask_l   = (rights_arr == l)
        X_l      = X[mask_l]
        X_next_l = X_next[mask_l]
        a_l      = actions[mask_l]
        r_l      = rewards[mask_l]
        done_l   = dones[mask_l]
        nr_l     = next_rights_arr[mask_l]
        ns_l     = next_states[mask_l]   # raw state vectors for cooldown check
 
        # print(f"  [FQI] l={l}  transitions={mask_l.sum()}  features={n_features}", flush=True)
 
        for it in range(n_iterations):
            # ── Vectorised target computation ─────────────────────────────────
            # For each transition, look up Q0 and Q1 at the next state using
            # the Q-function for the appropriate next-rights level.
            Q0_next = np.zeros(len(X_next_l))
            Q1_next = np.zeros(len(X_next_l))
 
            for nr in range(1, L + 1):
                mask_nr = (nr_l == nr) & (~done_l)
                if mask_nr.any():
                    Q0_next[mask_nr] = X_next_l[mask_nr] @ Q_coeffs[nr][0]
                    Q1_next[mask_nr] = X_next_l[mask_nr] @ Q_coeffs[nr][1]
 
            # Mask out exercise during cooldown (cooldown encoded in last column)
            cooldown_active = ns_l[:, -1] > 0
            Q1_next[cooldown_active] = -np.inf
 
            Q_next_max       = np.maximum(Q0_next, Q1_next)
            Q_next_max[done_l] = 0.0                         # terminal states
 
            targets = r_l + df * Q_next_max                  # Bellman target
 
            # ── Ridge regression per action ───────────────────────────────────
            for a in [0, 1]:
                mask_a = (a_l == a)
                if mask_a.sum() <= n_features:
                    continue
                Xa = X_l[mask_a]
                Ya = targets[mask_a]
                AtA = Xa.T @ Xa + l2_reg * np.eye(n_features)
                Atb = Xa.T @ Ya
                Q_coeffs[l][a] = np.linalg.solve(AtA, Atb)
 
        # print(f"  [FQI] l={l} done.", flush=True)
 
    return Q_coeffs

# %%
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

class ConservativeQLearning:
    def __init__(self, state_dim, hidden_dim=64, lr=1e-3, alpha=1.0, device=device):
        self.q_net = DQN(state_dim, hidden_dim).to(device)
        self.target_net = DQN(state_dim, hidden_dim).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.alpha = alpha  # CQL penalty weight
        self.device = device
        self.state_dim = state_dim
        
    def train(self, paths, params, n_epochs=15, batch_size=512, target_update=5):
        df = params.df
        
        # 1. Collect Data instantly using our NumPy vectorized function
        S, A, R, S_next, D, _, _ = collect_offline_transitions(paths, params)
        
        # 2. Push to GPU dataloader
        dataset = TensorDataset(
            torch.tensor(S, dtype=torch.float32, device=self.device),
            torch.tensor(A, dtype=torch.int64, device=self.device),
            torch.tensor(R, dtype=torch.float32, device=self.device),
            torch.tensor(S_next, dtype=torch.float32, device=self.device),
            torch.tensor(D, dtype=torch.float32, device=self.device)
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # 3. Train at maximum throughput
        self.q_net.train()
        for epoch in range(n_epochs):
            for states, actions, rewards, next_states, dones in dataloader:
                
                # Get Q-values for all actions
                q_values = self.q_net(states)
                
                # Get Q-values for the actions actually taken in the dataset
                q_a = q_values.gather(1, actions.unsqueeze(1)).squeeze()
                
                with torch.no_grad():
                    next_q = self.target_net(next_states).max(1)[0]
                    targets = rewards + df * next_q * (1 - dones)
                    
                # Standard Bellman Error (MSE)
                mse_loss = nn.functional.mse_loss(q_a, targets)
                
                # CQL Discrete Loss: minimize logsumexp(Q) - Q(s,a)
                # This pushes down the aggregate Q-values and pulls up the dataset action's Q-value
                cql_loss = (torch.logsumexp(q_values, dim=1) - q_a).mean()
                
                # Total Loss
                loss = mse_loss + self.alpha * cql_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
                self.optimizer.step()
                
            if (epoch + 1) % target_update == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())
                
        return self
        
    def predict(self, state):
        self.q_net.eval()
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device)
            return self.q_net(state_t).argmax().item()

# %%
class LeastSquarePolicyIteration:
    """
    LSPI via the LSTDQ operator.
 
    Key improvements over the original:
      • Guided transition collection (behavior_fn, ε-greedy).
      • l2_reg=1e-1  (was 1e-4) — the joint (2k)-dim parameter vector needs
        much stronger regularisation than per-action FQI regression.
      • np.linalg.lstsq instead of np.linalg.solve for numerical safety.
      • Vectorised next-action computation (no per-sample Python loop).
      • n_iterations=10  (was 5); LSTDQ converges faster than FQI per iteration.
      • State uses time-to-maturity (see make_state).
    """
 
    def __init__(self, degree=3):
        self.degree  = degree
        self.Q_coeffs = {}    # {rights_level: {0: w, 1: w}}
 
    # ── feature helpers ───────────────────────────────────────────────────────
 
    def get_features(self, state):
        """Single-sample features (used by predict and make_decision_fn)."""
        return fqi_state_features(state, self.degree)
 
    def _batch_features(self, state_matrix):
        return fqi_state_features_batch(state_matrix.astype(np.float64), self.degree)
 
    # ── training ──────────────────────────────────────────────────────────────
 
    def train(self, paths, params, n_iterations=10, behavior_fn=None, epsilon=0.2):
        """
        Parameters
        ----------
        paths        : np.ndarray  (n_paths, N+1, M)
        params       : Params
        n_iterations : int   LSTDQ outer iterations per rights level
        behavior_fn  : callable or None   guided policy
        epsilon      : float ε-greedy rate
 
        Returns
        -------
        self
        """
        N, df, K, L, delta = params.N, params.df, params.K, params.L, params.delta
 
        # ── 1. Transitions ────────────────────────────────────────────────────
        states, actions, rewards, next_states, dones, rights_arr, next_rights_arr = \
            collect_offline_transitions(paths, params, behavior_fn=behavior_fn, epsilon=epsilon)
 
        # ── 2. Feature matrices ───────────────────────────────────────────────
        # print(f"  [LSPI] Building feature matrices for {len(states)} transitions …", flush=True)
        X      = self._batch_features(states)
        X_next = self._batch_features(next_states)
        n_features = X.shape[1]
        l2_reg = 1e-1   # much larger than original 1e-4
 
        self.Q_coeffs = {l: {0: np.zeros(n_features), 1: np.zeros(n_features)}
                         for l in range(1, L + 1)}
 
        # ── 3. Layer-by-layer LSTDQ ───────────────────────────────────────────
        for l in range(1, L + 1):
            mask_l   = (rights_arr == l)
            X_l      = X[mask_l]
            X_next_l = X_next[mask_l]
            a_l      = actions[mask_l]
            r_l      = rewards[mask_l]
            done_l   = dones[mask_l]
            nr_l     = next_rights_arr[mask_l]
            ns_l     = next_states[mask_l]
 
            # print(f"  [LSPI] l={l}  transitions={mask_l.sum()}  features={n_features}", flush=True)
 
            # Joint parameter vector w = [w_hold | w_exercise],  dim = 2k
            k = n_features
            w = np.zeros(2 * k)
 
            # Pre-build Φ(s, a):  block-sparse joint feature matrix (n, 2k)
            Phi_sa = np.zeros((len(X_l), 2 * k))
            Phi_sa[a_l == 0, :k] = X_l[a_l == 0]
            Phi_sa[a_l == 1, k:] = X_l[a_l == 1]
 
            for it in range(n_iterations):
                # ── A. Greedy next action a'  (vectorised) ────────────────────
                # Compute Q(s', 0) and Q(s', 1) for each next state.
                Q0_sp = np.zeros(len(X_next_l))
                Q1_sp = np.zeros(len(X_next_l))
 
                for nr in range(1, L + 1):
                    mask_nr = (nr_l == nr) & (~done_l)
                    if not mask_nr.any():
                        continue
                    if nr == l:
                        Q0_sp[mask_nr] = X_next_l[mask_nr] @ w[:k]
                        Q1_sp[mask_nr] = X_next_l[mask_nr] @ w[k:]
                    else:
                        Q0_sp[mask_nr] = X_next_l[mask_nr] @ self.Q_coeffs[nr][0]
                        Q1_sp[mask_nr] = X_next_l[mask_nr] @ self.Q_coeffs[nr][1]
 
                # Force hold during cooldown
                cooldown_active = ns_l[:, -1] > 0
                Q1_sp[cooldown_active] = -np.inf
 
                # Greedy next action
                a_next = (Q1_sp > Q0_sp).astype(int)
                a_next[done_l] = 0
 
                # ── B. Build Φ(s', a') and fixed future values ────────────────
                Phi_spap        = np.zeros((len(X_next_l), 2 * k))
                fixed_future    = np.zeros(len(X_next_l))
 
                # Transitions that stay in the same rights level l
                same_l = (nr_l == l) & (~done_l)
                Phi_spap[same_l & (a_next == 0), :k] = X_next_l[same_l & (a_next == 0)]
                Phi_spap[same_l & (a_next == 1), k:] = X_next_l[same_l & (a_next == 1)]
 
                # Transitions that drop to a different rights level (fixed constant)
                for nr in range(1, L + 1):
                    if nr == l:
                        continue
                    mask_nr = (nr_l == nr) & (~done_l)
                    if not mask_nr.any():
                        continue
                    an = a_next[mask_nr]
                    q0 = X_next_l[mask_nr] @ self.Q_coeffs[nr][0]
                    q1 = X_next_l[mask_nr] @ self.Q_coeffs[nr][1]
                    fixed_future[mask_nr] = np.where(an == 1, q1, q0)
 
                b_target = r_l + df * fixed_future      # (n,)
 
                # ── C. LSTDQ solve:  A w = b ──────────────────────────────────
                A_mat = Phi_sa.T @ (Phi_sa - df * Phi_spap)   # (2k, 2k)
                b_vec = Phi_sa.T @ b_target                    # (2k,)
 
                # lstsq is safer than solve when A is near-singular
                w, _, _, _ = np.linalg.lstsq(
                    A_mat + l2_reg * np.eye(2 * k), b_vec, rcond=None
                )
 
            self.Q_coeffs[l][0] = w[:k]
            self.Q_coeffs[l][1] = w[k:]
            # print(f"  [LSPI] l={l} done.", flush=True)
 
        return self
 
    # ── inference ─────────────────────────────────────────────────────────────
 
    def predict(self, state, rights_left):
        if rights_left not in self.Q_coeffs or rights_left <= 0:
            return 0
        features = self.get_features(state)
        q0 = features @ self.Q_coeffs[rights_left][0]
        q1 = features @ self.Q_coeffs[rights_left][1]
        return 1 if q1 > q0 else 0
 

# %% [markdown]
# ### nn

# %%
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ValueNet(nn.Module):
    """A lightweight MLP to replace polynomial regression."""
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        return self.net(x).squeeze(-1)

def fit_network(net, X_np, Y_np, epochs=15, lr=1e-3):
    """Rapidly trains the network on the current cross-section of paths."""
    net.train()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    X_tensor = torch.tensor(X_np, dtype=torch.float32, device=device)
    Y_tensor = torch.tensor(Y_np, dtype=torch.float32, device=device)
    
    for _ in range(epochs):
        optimizer.zero_grad()
        predictions = net(X_tensor)
        loss = criterion(predictions, Y_tensor)
        loss.backward()
        optimizer.step()
        
    net.eval()
    return net

# %%
def query_V_batch_net(models, prices, l, t, params):
    """Evaluates the expected value using the PyTorch networks."""
    prices_2d = prices[:, t]
    n = len(prices_2d)
    
    if l == 0 or t > params.N:  
        return np.zeros(n)
        
    p = payoff_put(prices_2d, params.K)
    if t == params.N:
        return p

    key_cont = (l, t, 'cont')
    key_ex = (l, t, 'ex')
    
    if key_cont in models:
        # Standardize prices just like we did for polynomials
        X_norm = prices_2d / params.K 
        if X_norm.ndim == 1: X_norm = X_norm.reshape(-1, 1)
            
        X_tensor = torch.tensor(X_norm, dtype=torch.float32, device=device)
        
        with torch.no_grad():
            cont_est = models[key_cont](X_tensor).cpu().numpy()
            ex_residual = models[key_ex](X_tensor).cpu().numpy() if l > 1 else 0.0
            
        return np.maximum(p + ex_residual, cont_est)
        
    return np.zeros(n)

# %%
def train_lsm_net(paths, params, epochs=15):
    N, df, K, L, delta = params.N, params.df, params.K, params.L, params.delta
    n_paths = len(paths)
    payoffs = np.array([payoff_put(paths[:, t], K) for t in range(N + 1)]).T
    
    models = {}
    
    for l in range(1, L + 1):
        V = payoffs[:, N].copy()
        
        # Track the networks from t+1 for the warm-start
        prev_net_cont = None
        prev_net_ex = None
        
        for t in range(N - 1, 0, -1):
            realized_cont = df * V
            
            if t + delta + 1 > N:
                realized_residual = np.zeros(n_paths)
            else:
                t_resume = t + delta + 1
                residual = query_V_batch_net(models, paths, l - 1, t_resume, params)
                realized_residual = (df ** (t_resume - t)) * residual
            
            p_t = payoffs[:, t]
            itm = p_t > 0
            
            if itm.sum() < 32: # Need minimum batch size for Neural Net
                V = realized_cont
                continue
            
            # 1. Prepare normalized features
            X_itm = paths[itm, t] / K
            if X_itm.ndim == 1: X_itm = X_itm.reshape(-1, 1)
            
            # 2. Train Continuation Network (Warm Started)
            net_cont = ValueNet(params.M).to(device)
            if prev_net_cont is not None:
                net_cont.load_state_dict(prev_net_cont.state_dict())
                
            net_cont = fit_network(net_cont, X_itm, realized_cont[itm], epochs=epochs)
            models[(l, t, 'cont')] = net_cont
            prev_net_cont = net_cont # Save for t-1
            
            with torch.no_grad():
                cont_est = net_cont(torch.tensor(X_itm, dtype=torch.float32, device=device)).cpu().numpy()
            
            # 3. Train Residual Network
            if l > 1:
                net_ex = ValueNet(params.M).to(device)
                if prev_net_ex is not None:
                    net_ex.load_state_dict(prev_net_ex.state_dict())
                    
                net_ex = fit_network(net_ex, X_itm, realized_residual[itm], epochs=epochs)
                models[(l, t, 'ex')] = net_ex
                prev_net_ex = net_ex # Save for t-1
                
                with torch.no_grad():
                    expected_residual_est = net_ex(torch.tensor(X_itm, dtype=torch.float32, device=device)).cpu().numpy()
            else:
                expected_residual_est = np.zeros(itm.sum())
                
            expected_ex_val = p_t[itm] + expected_residual_est
            
            # 4. Decision Update
            exercise = expected_ex_val > cont_est
            V_new = realized_cont.copy()
            realized_ex_val = p_t[itm] + realized_residual[itm]
            V_new[itm] = np.where(exercise, realized_ex_val, realized_cont[itm])
            V = V_new

    return models

# %%
def train_tdlsm_net(paths, params, epochs=15):
    N, df, K, L, delta = params.N, params.df, params.K, params.L, params.delta
    n_paths = len(paths)
    payoffs = np.array([payoff_put(paths[:, t], K) for t in range(N + 1)]).T
    models = {}
    
    for l in range(1, L + 1):
        V_hat = payoffs[:, N].copy()
        
        prev_net_cont = None
        prev_net_ex = None
        
        for t in range(N - 1, 0, -1):
            td_target_cont = df * V_hat
            
            if t + delta + 1 > N:
                td_target_residual = np.zeros(n_paths)
            else:
                t_resume = t + delta + 1
                residual = query_V_batch_net(models, paths, l - 1, t_resume, params)
                td_target_residual = (df ** (t_resume - t)) * residual
            
            p_t = payoffs[:, t]
            itm = p_t > 0
            
            if itm.sum() < 32:
                V_hat = np.maximum(p_t + td_target_residual, td_target_cont)
                continue
            
            X_itm = paths[itm, t] / K
            if X_itm.ndim == 1: X_itm = X_itm.reshape(-1, 1)
            
            # 1. Train Continuation Network
            net_cont = ValueNet(params.M).to(device)
            if prev_net_cont is not None: net_cont.load_state_dict(prev_net_cont.state_dict())
            net_cont = fit_network(net_cont, X_itm, td_target_cont[itm], epochs=epochs)
            models[(l, t, 'cont')] = net_cont
            prev_net_cont = net_cont
            
            # 2. Train Residual Network
            if l > 1:
                net_ex = ValueNet(params.M).to(device)
                if prev_net_ex is not None: net_ex.load_state_dict(prev_net_ex.state_dict())
                net_ex = fit_network(net_ex, X_itm, td_target_residual[itm], epochs=epochs)
                models[(l, t, 'ex')] = net_ex
                prev_net_ex = net_ex
            
            # 3. TD Bootstrap: Predict across ALL paths (not just ITM)
            X_all = paths[:, t] / K
            if X_all.ndim == 1: X_all = X_all.reshape(-1, 1)
            X_all_tensor = torch.tensor(X_all, dtype=torch.float32, device=device)
            
            with torch.no_grad():
                cont_est_all = net_cont(X_all_tensor).cpu().numpy()
                ex_est_all = p_t + (models[(l, t, 'ex')](X_all_tensor).cpu().numpy() if l > 1 else 0.0)
            
            V_hat_new = td_target_cont.copy()
            V_hat_new[itm] = np.maximum(ex_est_all[itm], cont_est_all[itm])
            V_hat = V_hat_new
            
    return models

# %% [markdown]
# ### Evaluate 

# %%
def simulate_policy(decision_fn, paths, params):
    """
    Universal evaluation loop for all algorithms.
    Evaluates a policy on a set of paths and computes the mean discounted reward.
    """
    N, K, L, delta, df = params.N, params.K, params.L, params.delta, params.df
    rewards = np.zeros(len(paths))
    
    for i in range(len(paths)):
        path = paths[i]
        t, rights, cooldown, total_reward = 1, L, 0, 0.0
        
        while t <= N and rights > 0:
            # 1. Action Masking: Forced hold if in refractory period
            if cooldown > 0:
                decision = 'H'
            else:
                # 2. Query the agent's policy
                decision = decision_fn(path[t], t, rights, cooldown)
                
            # 3. Execute Decision
            if decision == 'E':
                pay = payoff_put(path[t].reshape(1, -1), K).item()
                total_reward += pay * (df ** t)
                rights -= 1
                cooldown = delta
            else:
                cooldown = max(0, cooldown - 1)
                
            t += 1
            
        # 4. Terminal Condition: Auto-exercise remaining rights at maturity
        if rights > 0:
            pay = payoff_put(path[N].reshape(1, -1), K).item()
            if pay > 0:
                total_reward += pay * (df ** N)
                
        rewards[i] = total_reward
        
    return rewards.mean(), rewards

# %%
# ═══════════════════════════════════════════════════════════════════════════════
# ▶▶  ALGORITHM SELECTOR  ◀◀
# ═══════════════════════════════════════════════════════════════════════════════
ALGORITHMS_TO_RUN = ['LSM-Net', 'TDLSM-Net', 'DQN', 'CQL']

# ─── Hyper-parameters ───────────────────────────────────────────────────────
param_configs = [
    # ── Config 1: The Vanilla Benchmark (American Put) ─────────────────────
    # Goal: Replicate original Longstaff-Schwartz results. Validates the codebase.
    # {'N': 50, 'M': 1, 'L': 1, 'delta': 0, 'sigma': 0.2, 'process': 'gbm', 'tag': 'American_Base'},

    # # ── Config 2: The Multi-Stop Core (Swing Option) ───────────────────────
    # # Goal: Observe how error compounds over multiple exercise rights.
    # {'N': 50, 'M': 1, 'L': 3, 'delta': 0, 'sigma': 0.2, 'process': 'gbm', 'tag': 'Swing_L3'},

    # ── Config 3: The High Volatility Stress Test ──────────────────────────
    # Goal: See which algorithm falls apart when paths cross the ITM/OTM boundary frequently.
    # {'N': 50, 'M': 1, 'L': 3, 'delta': 0, 'sigma': 0.6, 'process': 'gbm', 'tag': 'Swing_HighVol'},

    # ── Config 4: The Refractory Validation ────────────────────────────────
    # Goal: Prove the state-augmentation (cooldown) works. 
    # Delta=3 means exercising locks you out for ~6% of the option's life.
    # {'N': 50, 'M': 1, 'L': 3, 'delta': 3, 'sigma': 0.2, 'process': 'gbm', 'tag': 'Swing_Refractory'},

    # ── Config 5: The Dimensionality Curse (Basket Option) ─────────────────
    # Goal: Test the polynomial cross-terms. M=3 assets.
    # Expectation: FQI/LSPI matrix sizes grow, potentially causing instability.
    # {'N': 50, 'M': 3, 'L': 1, 'delta': 0, 'sigma': 0.2, 'process': 'gbm', 'tag': 'Basket_M3'},

    # ── Config 6: The Ultimate Boss (Multi-Asset Swing) ────────────────────
    # Goal: The hardest linear-basis problem. 3 assets, 3 rights, with cooldown.
    # {'N': 50, 'M': 3, 'L': 3, 'delta': 2, 'sigma': 0.2, 'process': 'gbm', 'tag': 'Basket_Swing_Full'},

    # {'N': 50, 'M': 1, 'L': 1, 'delta': 0, 'sigma': 0.5, 'process': 'gbm', 'tag': 'Basket_Swing_Full'},
    # extreme stress test
    # {'N': 252, 'M': 10, 'L': 10, 'delta': 0, 'sigma': 0.2, 'process': 'gbm', 'tag': 'Basket_Swing_Full'},




    ### deep
    # ── 1. The Bootstrapping Abyss (High-Frequency Swing) ────────────────────────
    # Goal: Break TD-LSM and FQI via error propagation over deep time horizons.
    # {'N': 252, 'M': 1, 'L': 5, 'delta': 10, 'sigma': 0.6, 'process': 'gbm', 'tag': 'Daily_Swing_HighVol'},

    # ── 2. The Dimensionality Curse (Deep RL's Playground) ───────────────────────
    # Goal: Choke the polynomial basis functions. Watch the neural nets dominate.
    {'N': 50, 'M': 5, 'L': 1, 'delta': 0, 'sigma': 0.3, 'process': 'gbm', 'tag': 'Basket_M5_Vanilla'},

    # ── 3. The Opportunity Cost Trap (Massive Cooldown) ──────────────────────────
    # Goal: Confuse the Q-networks. Exercising locks the agent out for half the option's life.
    # {'N': 100, 'M': 1, 'L': 2, 'delta': 45, 'sigma': 0.4, 'process': 'gbm', 'tag': 'Swing_MassiveCooldown'},

    # ── 4. The Instant Gratification Bait (Deep ITM + High Vol) ──────────────────
    # Goal: Tempt algorithms into premature exercise.
    # {'S0': 80.0, 'K': 100.0, 'N': 50, 'M': 1, 'L': 1, 'delta': 0, 'sigma': 0.8, 'process': 'gbm', 'tag': 'DeepITM_HighVol'},

    # ── 5. The Ultimate Boss (High-Dim Multi-Asset Swing) ────────────────────────
    # Goal: The hardest combination of spatial and temporal complexity.
    # {'N': 100, 'M': 5, 'L': 3, 'delta': 5, 'sigma': 0.3, 'process': 'gbm', 'tag': 'Basket_Swing_Full'}
]
    

TEST_PATHS = 100
train_sizes = [100, 500, 2000, 10000]
n_repeats   = 2

# ─── Style ──────────────────────────────────────────────────────────────────
import os
import time
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter

plt.style.use('seaborn-v0_8-whitegrid')
ALGO_COLORS = {
    'LSM':    '#2E86AB', 'TD-LSM': '#F18F01', 'FQI':    '#7B2D8E',
    'LSPI':   '#C73E1D', 'DQN':    '#1B998B', 'CQL':    '#44BBA4', 'PPO':    '#E94F37',
    'LSM-NET': '#2E86AB', 'TDLSM-NET':'#F18F01', 'DQN':'#C73E1D'
}
ALGO_MARKERS = {
    'LSM': 'o', 'TD-LSM': 's', 'FQI': '^',
    'LSPI': 'D', 'DQN': 'P', 'CQL': 'X', 'PPO': 'v',
}

# ══════════════════════════════════════════════════════════════════════════════
# EXACT OPTIMAL BOUNDARY  — continuous (GBM/IID) and discrete (Binomial)
# (Moved BEFORE the main loop so they can be called dynamically)
# ══════════════════════════════════════════════════════════════════════════════
def compute_exact_boundary_continuous(params, max_rights=None, n_grid=1500):
    if max_rights is None:
        max_rights = params.L
    log_S_min = np.log(params.K * 0.05)
    log_S_max = np.log(params.K * 8.0)
    log_S = np.linspace(log_S_min, log_S_max, n_grid)
    S     = np.exp(log_S)
    payoff = np.maximum(params.K - S, 0)
    mu_dt  = (params.r - 0.5 * params.sigma**2) * params.dt
    sig_dt = params.sigma * np.sqrt(params.dt)
    df     = params.df
    lS_row, lS_col = np.meshgrid(log_S, log_S, indexing='ij')
    P = st.norm.pdf(lS_col, loc=lS_row + mu_dt, scale=sig_dt)
    P /= P.sum(axis=1, keepdims=True)
    i0 = np.argmin(np.abs(S - params.S0))
    boundaries = np.zeros((max_rights + 1, params.N + 1))
    V_prev_l   = np.zeros((params.N + 1, n_grid))
    
    for l in range(1, max_rights + 1):
        V = np.zeros((params.N + 1, n_grid))
        V[params.N] = payoff.copy()
        boundaries[l, params.N] = params.K
        for t in range(params.N - 1, 0, -1):
            cont_val = df * (P @ V[t + 1])
            if l == 1:
                ex_val = payoff
            else:
                ex_val = payoff + df * (P @ V_prev_l[t + 1])
            V[t] = np.maximum(ex_val, cont_val)
            mask = (ex_val >= cont_val - 1e-8) & (payoff > 0)
            boundaries[l, t] = S[np.where(mask)[0][-1]] if mask.any() else 0.0
        V_prev_l = V.copy()
    V_root = V[0, i0]
    return boundaries, V_root

def compute_exact_boundary_binomial(params, max_rights=None):
    if max_rights is None:
        max_rights = params.L
    N, K, S0, df = params.N, params.K, params.S0, params.df
    u, d, p_u, p_d = _crr_params(params) # Assuming _crr_params is defined elsewhere
    
    S_node = np.zeros((N + 1, N + 1))
    for t in range(N + 1):
        for j in range(t + 1):
            S_node[t, j] = S0 * (u ** j) * (d ** (t - j))
    payoff_node = np.maximum(K - S_node, 0)
    V         = np.zeros((max_rights + 1, N + 1, N + 1))
    boundaries = np.zeros((max_rights + 1, N + 1))

    for l in range(1, max_rights + 1):
        for j in range(N + 1):
            V[l, N, j] = payoff_node[N, j]
        boundaries[l, N] = K
        for t in range(N - 1, -1, -1):
            bnd = 0.0
            for j in range(t + 1):
                cont = df * (p_u * V[l, t+1, j+1] + p_d * V[l, t+1, j])
                pay  = payoff_node[t, j]
                if l == 1:
                    ex = pay
                else:
                    ex = pay + df * (p_u * V[l-1, t+1, j+1] + p_d * V[l-1, t+1, j])
                V[l, t, j] = max(ex, cont)
                if pay > 0 and ex >= cont - 1e-12:
                    bnd = max(bnd, S_node[t, j])
            boundaries[l, t] = bnd
    V_root = V[max_rights, 0, 0]
    return boundaries, V_root


def compute_exact_boundary_trinomial(params, max_rights=None):
    if max_rights is None:
        max_rights = params.L
    N, K, S0, df = params.N, params.K, params.S0, params.df
    u, m, d, p_u, p_m, p_d = _trinomial_params(params) 
    
    # Precompute all node prices. 
    # A trinomial tree at time t has 2t + 1 nodes.
    S_node = np.zeros((N + 1, 2 * N + 1))
    for t in range(N + 1):
        for j in range(2 * t + 1):
            # j - t represents the net number of 'up' moves.
            # If j = t, net moves = 0, so price is S0.
            S_node[t, j] = S0 * (u ** (j - t))
            
    payoff_node = np.maximum(K - S_node, 0)
    
    # V[l, t, j] — value with l rights at node (t, j)
    V = np.zeros((max_rights + 1, N + 1, 2 * N + 1))
    boundaries = np.zeros((max_rights + 1, N + 1))

    for l in range(1, max_rights + 1):
        # Terminal condition at maturity
        for j in range(2 * N + 1):
            V[l, N, j] = payoff_node[N, j]
        boundaries[l, N] = K

        for t in range(N - 1, -1, -1):
            diffs = np.zeros(2 * t + 1)
            
            for j in range(2 * t + 1):
                # Trinomial Transitions:
                # Up goes to index j+2, Mid to j+1, Down to j
                cont = df * (p_u * V[l, t+1, j+2] + 
                             p_m * V[l, t+1, j+1] + 
                             p_d * V[l, t+1, j])
                
                pay = payoff_node[t, j]
                
                if l == 1:
                    ex = pay
                else:
                    # Must also use Trinomial transitions for the multi-right residual!
                    ex_cont = df * (p_u * V[l-1, t+1, j+2] + 
                                    p_m * V[l-1, t+1, j+1] + 
                                    p_d * V[l-1, t+1, j])
                    ex = pay + ex_cont
                    
                V[l, t, j] = max(ex, cont)
                diffs[j] = ex - cont  # Positive means Exercise, Negative means Continue
                
            # Find the exact boundary via linear interpolation
            bnd = 0.0
            for j in range(2 * t):
                # We exercise at low prices and hold at high prices.
                # Since lower index j means a lower price, we look for the 
                # crossover where diffs drops from Positive to Negative.
                if diffs[j] >= 0 and diffs[j+1] < 0 and payoff_node[t, j] > 0:
                    frac = diffs[j] / (diffs[j] - diffs[j+1])
                    bnd = S_node[t, j] + frac * (S_node[t, j+1] - S_node[t, j])
                    break
            
            # Edge case: If all nodes dictate exercise
            if bnd == 0.0 and diffs[-1] >= 0 and payoff_node[t, -1] > 0:
                bnd = S_node[t, -1]
                
            boundaries[l, t] = bnd

    # The root of the tree is at t=0, net moves=0, which is index j=0
    V_root = V[max_rights, 0, 0] 
    return boundaries, V_root

def compute_exact_boundary(params, process, max_rights=None, **kwargs):
    if process == 'binomial':
        return compute_exact_boundary_binomial(params, max_rights)
    elif process == 'trinomial':
        return compute_exact_boundary_trinomial(params, max_rights)
    else:
        return compute_exact_boundary_continuous(params, max_rights, **kwargs)


# ══════════════════════════════════════════════════════════════════════════════
# VISUALISER  — exercise boundaries for all trained algorithms
# ══════════════════════════════════════════════════════════════════════════════
_ALGO_TYPE = {
    'LSM':'lsm', 'TD-LSM':'lsm', 'FQI':'fqi', 'LSPI':'lspi',
    'DQN':'dqn', 'CQL':'cql_net', 'PPO':'ppo', 'LSM-Net':'lsm_net', 'TDLSM-Net':'lsm_net'
}

def _unwrap_model(algo, model):
    if algo == 'CQL': return model.q_net
    return model

from PIL import Image  # Required for GIF creation

def plot_boundaries_all(p, models_map, rights_left=1, process='gbm',
                        degree=3, n_scan=300, n_grid=1200,
                        smooth_size=5, save_path=None, title_suffix=""):
    # Skip plotting if the dimensionality is too high
    if p.M > 1:
        print("Skipping plot: dimensionality > 1")
        return

    # True optimal boundary
    exact_bnd, V_root = compute_exact_boundary(p, process, max_rights=rights_left, n_grid=n_grid)
    true_bnd = exact_bnd[rights_left, 1:]
    
    # Learned boundaries
    learned = {}
    for algo, model in models_map.items():
        algo_type   = _ALGO_TYPE[algo]
        clean_model = _unwrap_model(algo, model)
        # Assuming device is globally defined as torch.device(...)
        decision_fn = make_decision_fn(algo_type, clean_model, p, degree) 
        bnd = []
        times = np.arange(1, p.N + 1)
        for t in times:
            if t == p.N:
                bnd.append(p.K)
            else:
                bnd.append(find_boundary_generic(decision_fn, t, rights_left, p, n_scan))
        bnd = np.array(bnd, dtype=float)
        valid = ~np.isnan(bnd)
        if valid.sum() > smooth_size and smooth_size > 1:
            bnd[valid] = median_filter(bnd[valid], size=smooth_size)
        learned[algo] = bnd

    # Plot
    fig, ax = plt.subplots(figsize=(12, 5), dpi=110)
    fig.patch.set_facecolor('white'); ax.set_facecolor('white')
    valid_true = true_bnd > 0
    times = np.arange(1, p.N + 1)
    ax.plot(times[valid_true], true_bnd[valid_true], color='black', linewidth=2.5, linestyle='-', label=f'Exact DP ({process})', zorder=5)
    ax.fill_between(times[valid_true], 0, true_bnd[valid_true], alpha=0.06, color='black', label='Exercise region (DP)')
    STYLES = {'LSM':'--','TD-LSM':'--','FQI':':','LSPI':':','DQN':'-.','CQL':'-.','PPO':'-.'}
    for algo, bnd in learned.items():
        valid = ~np.isnan(bnd) & (bnd > 0)
        ax.plot(times[valid], bnd[valid], color=ALGO_COLORS.get(algo, 'gray'),
                linestyle=STYLES.get(algo, '--'), linewidth=1.8, label=algo, zorder=4)

    ax.axhline(p.K, color='gray', linewidth=1, linestyle=':', alpha=0.6, label=f'Strike K={p.K}')
    ax.set_xlabel('Time step t', fontsize=13)
    ax.set_ylabel('Asset price S*(t)', fontsize=13)
    
    full_title = f'Exercise boundary comparison  ({process})\nN={p.N}, L={p.L}, σ={p.sigma}, rights_left={rights_left}'
    if title_suffix:
        full_title += f"  {title_suffix}"
        
    ax.set_title(full_title, fontsize=14, fontweight='bold', pad=12)
    ax.legend(fontsize=10, frameon=True, fancybox=False, edgecolor='lightgray', framealpha=0.95, ncol=2)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.set_xlim(1, p.N); ax.set_ylim(0, p.K * 1.05)
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=110, bbox_inches='tight')
        plt.close(fig)  # Crucial to prevent memory leaks in loops
    else:
        plt.show()

# ─── Per-algorithm train/evaluate dispatch ───────────────────────────────────
# [Keep train_algo and eval_algo functions exactly as you provided them...]
# ─── Per-algorithm train/evaluate dispatch ───────────────────────────────────
def train_algo(algo, paths, params, lsm_model=None):
    """Train algo and return its model object."""
    if algo == 'LSM':
        return train_lsm(paths, params)
    elif algo == 'TD-LSM':
        return train_td_lsm(paths, params)
    elif algo == 'FQI':
        # return train_fqi(paths, params, n_iterations=5)
        behavior_fn = None
        if lsm_model is not None:
            # Build a decision_fn from the LSM model — same interface as always
            behavior_fn = make_decision_fn('lsm', lsm_model, params)
        return train_fqi(paths, params, n_iterations=500,
                         behavior_fn=behavior_fn, epsilon=0.2)
    elif algo == 'LSPI':
        behavior_fn = None
        if lsm_model is not None:
            behavior_fn = make_decision_fn('lsm', lsm_model, params)
        return LeastSquarePolicyIteration().train(paths, params, n_iterations=100,
                                                   behavior_fn=behavior_fn, epsilon=0.2)
    elif algo == 'DQN':
        return train_dqn(paths, params, n_epochs=10)
    elif algo == 'CQL':
        state_dim = params.M + 3
        return ConservativeQLearning(state_dim).train(paths, params, n_epochs=10)

    elif algo == 'PPO':
        return train_ppo(paths, params, n_epochs=10)
    elif algo == 'LSM-Net':
        return train_lsm_net(paths, params, epochs=10)
    elif algo == 'TDLSM-Net':
        return train_tdlsm_net(paths, params, epochs=10)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

def eval_algo(algo, model, paths, params, degree=3, device='cpu'):
    """Universal evaluator relying on the unified decision_fn."""
    # Map 'TD-LSM' -> 'lsm', 'CQL' -> 'cql_net', etc.
    algo_type = _ALGO_TYPE.get(algo, algo.lower())
    
    # Extract the base PyTorch module if it's wrapped in a class (like CQL)
    clean_model = _unwrap_model(algo, model)
    
    # Build the standardized decision function
    decision_fn = make_decision_fn(algo_type, clean_model, params, degree)
    
    # Run the unified simulation
    return simulate_policy(decision_fn, paths, params)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN COMPARISON LOOP 
# ══════════════════════════════════════════════════════════════════════════════
all_results = {}   # keyed by config_idx

for config_idx, config in enumerate(param_configs):
    process_type = config.get('process', 'gbm')
    
    test_params = Params(
        S0=config.get('S0',params.S0), K=config.get('K',params.K), T=config.get('T', params.T), r=config.get('r', params.r),
        sigma=config.get('sigma', 0.5),
        N=config['N'], M=config.get('M',1), L=config['L'],
        delta=config.get('delta', 0)
    )

    print(f"\n{'='*80}")
    print(f"Config {config_idx+1}/{len(param_configs)}: "
          f"N={test_params.N}, L={test_params.L}, M={test_params.M}, σ={test_params.sigma}, δ={test_params.delta}, process={process_type}")
    print(f"Running: {ALGORITHMS_TO_RUN}")
    print('='*80)

    folder_name = (f"N{test_params.N}_L{test_params.L}_M={test_params.M}" +
                  f"_sigma{test_params.sigma}_delta{test_params.delta}_{process_type}")
    os.makedirs(folder_name, exist_ok=True)

    test_paths_config = generate_paths(test_params, TEST_PATHS, seed=997)
    results = {algo: {'means': [], 'stds': [], 'times': []} for algo in ALGORITHMS_TO_RUN}
    
    # Dictionary to hold the frame paths for our GIF
    frames_dict = {l: [] for l in range(1, test_params.L + 1)}

    for n in train_sizes:
        algo_vals  = {a: [] for a in ALGORITHMS_TO_RUN}
        algo_times = {a: [] for a in ALGORITHMS_TO_RUN}
        trained_models = {}

        # lsm_model = train_algo('LSM', paths_n, test_params)
        for rep in range(n_repeats):
            paths_n = generate_paths(test_params, n, seed=42 + rep)
            for algo in ALGORITHMS_TO_RUN:
                t0 = time.time()
                model = train_algo(algo, paths_n, test_params)
                algo_times[algo].append(time.time() - t0)
                val, _ = eval_algo(algo, model, test_paths_config, test_params)
                algo_vals[algo].append(val)
                trained_models[algo] = model

        for algo in ALGORITHMS_TO_RUN:
            results[algo]['means'].append(np.mean(algo_vals[algo]))
            results[algo]['stds'].append(np.std(algo_vals[algo]))
            results[algo]['times'].append(np.mean(algo_times[algo]))

        line = f"n={n:>6d}  " + "  ".join(
            f"{a}={np.mean(algo_vals[a]):.4f}" for a in ALGORITHMS_TO_RUN
        )
        total_t = sum(np.mean(algo_times[a]) for a in ALGORITHMS_TO_RUN)
        print(f"{line}  [total: {total_t:.1f}s]")
        
        # --- NEW: Save boundary frame for the GIF at this train_size ---
        for l_rights in range(1, test_params.L + 1):
            frame_path = os.path.join(folder_name, f'tmp_frame_L{l_rights}_n{n}.png')
            plot_boundaries_all(
                p=test_params, 
                models_map=trained_models, 
                rights_left=l_rights, 
                process=process_type,
                save_path=frame_path,
                title_suffix=f"| Training Paths: {n}"
            )
            frames_dict[l_rights].append(frame_path)

    all_results[config_idx] = {'params': test_params, 'results': results,
                                'models': trained_models,
                                'test_paths': test_paths_config}

    # ── Plot 1: Payoff Comparison with Confidence Intervals ─────────────────────
    fig, ax = plt.subplots(figsize=(10, 5), dpi=110)
    fig.patch.set_facecolor('white'); ax.set_facecolor('white')
    
    for algo in ALGORITHMS_TO_RUN:
        means = np.array(results[algo]['means'])
        stds = np.array(results[algo]['stds'])
        
        # Plot the main solid line
        ax.plot(train_sizes, means,
                marker=ALGO_MARKERS.get(algo, 'o'),
                color=ALGO_COLORS.get(algo, 'gray'),
                label=algo, linewidth=2.5, markersize=8,
                markeredgecolor='white', markeredgewidth=1.5)
                
        # Draw the shaded confidence interval
        ax.fill_between(train_sizes, 
                        means - stds, 
                        means + stds, 
                        color=ALGO_COLORS.get(algo, 'gray'), 
                        alpha=0.15) # alpha makes it semi-transparent

    ax.set_xlabel('Training paths', fontsize=13)
    ax.set_ylabel('Mean discounted payoff (test)', fontsize=13)
    ax.set_title(f"Payoff comparison — N={test_params.N}, L={test_params.L}, M={test_params.M}, σ={test_params.sigma}",
                 fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.legend(fontsize=11, frameon=True, fancybox=False, edgecolor='lightgray')
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(folder_name, 'payoff_comparison.png'), dpi=110, bbox_inches='tight')
    plt.show()

    # ── Plot 2: Training Time ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5), dpi=110)
    fig.patch.set_facecolor('white'); ax.set_facecolor('white')
    for algo in ALGORITHMS_TO_RUN:
        ax.plot(train_sizes, results[algo]['times'],
                marker=ALGO_MARKERS.get(algo, 's'),
                color=ALGO_COLORS.get(algo, 'gray'),
                label=algo, linewidth=2.5, markersize=8,
                markeredgecolor='white', markeredgewidth=1.5)
    ax.set_xlabel('Training paths', fontsize=13)
    ax.set_ylabel('Training time (seconds)', fontsize=13)
    ax.set_title(f"Training time — N={test_params.N}, L={test_params.L}, M={test_params.M}, σ={test_params.sigma}",
                 fontsize=14, fontweight='bold')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.legend(fontsize=11, frameon=True, fancybox=False, edgecolor='lightgray')
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(folder_name, 'training_time.png'), dpi=110, bbox_inches='tight')
    plt.show()

    # ── Plot 3: Generate Boundary GIF & Final Static Image ────────────────────
    if test_params.M > 1:
        print("Skipping boundary plots/GIF: dimensionality > 1")
        continue
    for l_rights, frame_paths in frames_dict.items():
        if not frame_paths:
            continue
            
        gif_path = os.path.join(folder_name, f'boundary_evolution_L{l_rights}_{process_type}.gif')
        print(f"Saving GIF to {gif_path}...")
        
        # Load images and save as looping GIF (duration is in ms, 800ms = 0.8 sec per frame)
        imgs = [Image.open(fp) for fp in frame_paths]
        imgs[0].save(gif_path, save_all=True, append_images=imgs[1:], duration=800, loop=0)
        
        # Save the very last frame as the standalone static boundary image
        last_frame_path = os.path.join(folder_name, f'boundary_L{l_rights}_{process_type}.png')
        imgs[-1].save(last_frame_path)
        
        # Clean up the temporary frame files to keep the directory clean
        for fp in frame_paths:
            os.remove(fp)

print("\n" + "="*80)
print("All configs complete! Plots and GIFs stored in respective parameter folders.")
print("="*80)

# %% [markdown]
# 1. For small complexity problems, yes fqi, lspi can beat lsm. Need to explore exactly waht modifications lie et all papers used for beating lsm. 
# 2. For large complexity, longer horizons, hands down tdlsm > lsm > others. 

# %% [markdown]
# To do:
# 1. Fix the delta errors everywhere. 
# 2. Then test for different deltas.
# 3. Test with real stock data. 
# 4. randomise_S0 = True for training, but false for test dataset. 
# 5. Checking if the paper already exists that does the same thing. 

# %% [markdown]
# Experiment Design
# 1. Should I have DQN and CQL? : For multiple stopping DQN and CQL seem to be performing well. 
# 2. For L=1, various N and sigma. For the pure american put option, see if results from Li et all is being replicated. 
# 3. Test various settings, show that TD-LSM performs better than LSM. 
# 4. Tets with real stock data
# 
# Non deep RL vs Deep RL:
# 1. LSM, Q-LEARNING, TD-LSM, FQI, LSPI.
# 2. CQL, LSM Net, DQN, TD-LSM Net.  

# %% [markdown]
# Contributions of the paper: 
# 1. TD-LSM.
# 2. Proofs.
# 3. Compare all existing RL algorithms.


