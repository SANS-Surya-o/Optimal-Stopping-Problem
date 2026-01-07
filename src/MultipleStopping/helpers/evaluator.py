"""
Policy Evaluator: Compare performance of different policies.

Evaluates and compares policies on the environment with various metrics and visualizations.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple, Optional, List
from tqdm import tqdm


class PolicyEvaluator:
    """
    Evaluates and compares multiple policies on an environment.
    
    Supports:
    - Running episodes with different policies
    - Computing performance metrics (mean reward, std, success rate, etc.)
    - Visualizing comparisons (reward distributions, action patterns, etc.)
    """
    
    def __init__(self, env):
        """
        Initialize evaluator.
        
        Args:
            env: Environment to evaluate on
        """
        self.env = env
        self.results = {}
    
    def evaluate_policy(
        self,
        policy: np.ndarray,
        n_episodes: int = 1000,
        policy_name: str = "Policy",
        verbose: bool = True,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single policy on the environment.
        
        Args:
            policy: Policy array of shape (n_states, max_time+1, max_stops+1)
                    where policy[s, t, stops] = action (0 or 1)
            n_episodes: Number of episodes to evaluate
            policy_name: Name for this policy
            verbose: Whether to show progress bar
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary with evaluation metrics and episode data
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Storage for episode data
        episode_rewards = []
        episode_lengths = []
        buy_prices = []
        sell_prices = []
        buy_times = []
        sell_times = []
        holding_times = []
        profits = []
        
        iterator = tqdm(range(n_episodes), desc=f"Evaluating {policy_name}") if verbose else range(n_episodes)
        
        for ep in iterator:
            obs, info = self.env.reset()
            total_reward = 0.0
            steps = 0
            done = False
            
            while not done:
                s = obs['offer']
                t = obs['time_left']
                stops = obs['stops_left']
                
                # Get action from policy
                action = policy[s, t, stops]
                
                obs, reward, terminated, truncated, info = self.env.step(action)
                total_reward += reward
                steps += 1
                done = terminated or truncated
            
            # Record episode data
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            
            # Handle backward compatibility for max_stops=2
            # For other max_stops values, these fields may not exist
            if 'buy_price' in info and info['buy_price'] is not None:
                buy_prices.append(info['buy_price'])
                buy_times.append(info['buy_time'])
            
            if 'sell_price' in info and info['sell_price'] is not None:
                sell_prices.append(info['sell_price'])
                sell_times.append(info['sell_time'])
            
            if ('buy_time' in info and 'sell_time' in info and 
                info['buy_time'] is not None and info['sell_time'] is not None):
                holding_time = info['sell_time'] - info['buy_time']
                holding_times.append(holding_time)
                profit = info['sell_price'] - info['buy_price']
                profits.append(profit)
        
        # Compute metrics
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        
        # Sharpe ratio (risk-adjusted return)
        sharpe_ratio = mean_reward / std_reward if std_reward > 0 else np.nan
        
        results = {
            'policy_name': policy_name,
            'n_episodes': n_episodes,
            'episode_rewards': np.array(episode_rewards),
            'episode_lengths': np.array(episode_lengths),
            'buy_prices': np.array(buy_prices) if buy_prices else np.array([]),
            'sell_prices': np.array(sell_prices) if sell_prices else np.array([]),
            'buy_times': np.array(buy_times) if buy_times else np.array([]),
            'sell_times': np.array(sell_times) if sell_times else np.array([]),
            'holding_times': np.array(holding_times) if holding_times else np.array([]),
            'profits': np.array(profits) if profits else np.array([]),
            
            # Summary statistics
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'median_reward': np.median(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'sharpe_ratio': sharpe_ratio,
            'mean_episode_length': np.mean(episode_lengths),
            'std_episode_length': np.std(episode_lengths),
            
            # Buy-sell specific metrics (optional, may not apply to all envs)
            'completion_rate': len(profits) / n_episodes,  # Fraction that completed buy+sell
            'mean_buy_price': np.mean(buy_prices) if len(buy_prices) > 0 else np.nan,
            'mean_sell_price': np.mean(sell_prices) if len(sell_prices) > 0 else np.nan,
            'mean_holding_time': np.mean(holding_times) if len(holding_times) > 0 else np.nan,
            'mean_profit': np.mean(profits) if len(profits) > 0 else np.nan,
        }
        
        # Store results
        self.results[policy_name] = results
        
        if verbose:
            self._print_summary(results)
        
        return results
    
    def compare_policies(
        self,
        policies: List[np.ndarray],
        policy_names: List[str],
        n_episodes: int = 1000,
        seed: Optional[int] = 42,
        verbose: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Compare multiple policies on the environment.
        
        Args:
            policies: List of policy arrays
            policy_names: List of names for each policy
            n_episodes: Number of episodes to evaluate each
            seed: Random seed for reproducibility
            verbose: Whether to show progress
            
        Returns:
            List of results dictionaries
        """
        if len(policies) != len(policy_names):
            raise ValueError(f"Number of policies ({len(policies)}) must match number of names ({len(policy_names)})")
        
        if len(policies) < 2:
            raise ValueError("Need at least 2 policies to compare")
        
        # Evaluate all policies with same seed for fair comparison
        all_results = []
        for policy, name in zip(policies, policy_names):
            results = self.evaluate_policy(policy, n_episodes, name, verbose, seed)
            all_results.append(results)
        
        if verbose:
            self._print_comparison_multi(all_results)
        
        return all_results
    
    def _print_summary(self, results: Dict[str, Any]):
        """Print summary statistics for a policy."""
        print(f"\n{'='*60}")
        print(f"Evaluation Summary: {results['policy_name']}")
        print(f"{'='*60}")
        print(f"Episodes: {results['n_episodes']}")
        print(f"\nReward Statistics:")
        print(f"  Mean:        {results['mean_reward']:>8.4f}")
        print(f"  Std:         {results['std_reward']:>8.4f}")
        print(f"  Sharpe:      {results['sharpe_ratio']:>8.4f}")
        print(f"  Median:      {results['median_reward']:>8.4f}")
        print(f"  Min:         {results['min_reward']:>8.4f}")
        print(f"  Max:         {results['max_reward']:>8.4f}")
        print(f"\nEpisode Length:")
        print(f"  Mean:        {results['mean_episode_length']:>8.2f}")
        print(f"  Std:         {results['std_episode_length']:>8.2f}")
        
        # Only print buy-sell metrics if they exist (not NaN)
        if not np.isnan(results['completion_rate']) and results['completion_rate'] > 0:
            print(f"\nBuy-Sell Specific Metrics:")
            print(f"  Completion rate:   {results['completion_rate']:>6.2%}")
            print(f"  Mean buy price:    {results['mean_buy_price']:>8.4f}")
            print(f"  Mean sell price:   {results['mean_sell_price']:>8.4f}")
            print(f"  Mean profit:       {results['mean_profit']:>8.4f}")
            print(f"  Mean holding time: {results['mean_holding_time']:>8.2f}")
        print(f"{'='*60}\n")
    
    def _print_comparison(self, results1: Dict[str, Any], results2: Dict[str, Any]):
        """Print comparison between two policies (legacy, kept for compatibility)."""
        self._print_comparison_multi([results1, results2])
    
    def _print_comparison_multi(self, all_results: List[Dict[str, Any]]):
        """Print comparison between multiple policies."""
        if len(all_results) < 2:
            print("Need at least 2 policies to compare")
            return
        
        print(f"\n{'='*80}")
        print(f"Policy Comparison: {' vs '.join([r['policy_name'] for r in all_results])}")
        print(f"{'='*80}")
        
        metrics = [
            ('Mean Reward', 'mean_reward', True),
            ('Std Reward', 'std_reward', False),
            ('Sharpe Ratio', 'sharpe_ratio', True),
            ('Median Reward', 'median_reward', True),
            ('Mean Episode Length', 'mean_episode_length', False),
        ]
        
        # Add buy-sell metrics only if they're meaningful for at least one policy
        if any(not np.isnan(r['completion_rate']) and r['completion_rate'] > 0 for r in all_results):
            metrics.extend([
                ('Completion Rate', 'completion_rate', True),
                ('Mean Profit', 'mean_profit', True),
                ('Mean Holding Time', 'mean_holding_time', False),
            ])
        
        # Print header
        header = f"{'Metric':<25}"
        for r in all_results:
            header += f" {r['policy_name'][:12]:>12}"
        header += f"   {'Best':>12}"
        print(f"\n{header}")
        print(f"{'-'*80}")
        
        # Print each metric
        for metric_name, metric_key, higher_better in metrics:
            line = f"{metric_name:<25}"
            values = [r[metric_key] for r in all_results]
            
            # Add values
            for val in values:
                if np.isnan(val):
                    line += f" {'N/A':>12}"
                elif metric_key == 'completion_rate':
                    line += f" {val:>11.2%}"
                else:
                    line += f" {val:>12.4f}"
            
            # Find best
            valid_vals = [(i, v) for i, v in enumerate(values) if not np.isnan(v)]
            if valid_vals:
                if higher_better:
                    best_idx = max(valid_vals, key=lambda x: x[1])[0]
                else:
                    best_idx = min(valid_vals, key=lambda x: x[1])[0]
                best_name = all_results[best_idx]['policy_name'][:12]
            else:
                best_name = "N/A"
            
            line += f"   {best_name:>12}"
            print(line)
        
        print(f"{'='*80}\n")
    
    def plot_comparison(
        self,
        policy_names: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (14, 8)
    ):
        """
        Create comprehensive comparison plots between multiple policies.
        
        Args:
            policy_names: List of policy names to compare (uses all if None)
            figsize: Figure size
        """
        if len(self.results) < 2:
            print("Need at least 2 evaluated policies to compare")
            return
        
        # Get results
        if policy_names is None:
            policy_names = list(self.results.keys())
        
        all_results = [self.results[name] for name in policy_names]
        n_policies = len(all_results)
        
        # Color palette
        colors = plt.cm.tab10(np.linspace(0, 1, n_policies))
        
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.35)
        
        # 1. Reward distributions (histogram)
        ax1 = fig.add_subplot(gs[0, :2])
        for idx, (results, color) in enumerate(zip(all_results, colors)):
            ax1.hist(results['episode_rewards'], bins=50, alpha=0.5, 
                    label=results['policy_name'], color=color, density=True)
            ax1.axvline(results['mean_reward'], color=color, linestyle='--', 
                       linewidth=2, alpha=0.8)
        ax1.set_xlabel('Episode Reward', fontsize=11)
        ax1.set_ylabel('Density', fontsize=11)
        ax1.set_title('Reward Distribution', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(alpha=0.3)
        
        # 2. Cumulative reward distribution (CDF)
        ax2 = fig.add_subplot(gs[0, 2])
        for idx, (results, color) in enumerate(zip(all_results, colors)):
            sorted_rewards = np.sort(results['episode_rewards'])
            cdf = np.arange(1, len(sorted_rewards) + 1) / len(sorted_rewards)
            ax2.plot(sorted_rewards, cdf, label=results['policy_name'], 
                    color=color, linewidth=2)
        ax2.set_xlabel('Episode Reward', fontsize=11)
        ax2.set_ylabel('CDF', fontsize=11)
        ax2.set_title('Cumulative Distribution', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(alpha=0.3)
        
        # 3. Episode length distribution
        ax3 = fig.add_subplot(gs[1, 0])
        for idx, (results, color) in enumerate(zip(all_results, colors)):
            ax3.hist(results['episode_lengths'], bins=30, alpha=0.5, 
                    label=results['policy_name'], color=color, density=True)
            ax3.axvline(results['mean_episode_length'], color=color, 
                       linestyle='--', linewidth=2, alpha=0.8)
        ax3.set_xlabel('Episode Length', fontsize=11)
        ax3.set_ylabel('Density', fontsize=11)
        ax3.set_title('Episode Length Distribution', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(alpha=0.3)
        
        # 4. Box plot comparison
        ax4 = fig.add_subplot(gs[1, 1])
        box_data = [r['episode_rewards'] for r in all_results]
        labels = [r['policy_name'] for r in all_results]
        bp = ax4.boxplot(box_data, labels=labels, patch_artist=True, widths=0.6)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax4.set_ylabel('Episode Reward', fontsize=11)
        ax4.set_title('Reward Distribution (Box Plot)', fontsize=12, fontweight='bold')
        ax4.grid(alpha=0.3, axis='y')
        ax4.tick_params(axis='x', rotation=15)
        
        # 5. Key metrics comparison bar chart
        ax5 = fig.add_subplot(gs[1, 2])
        metrics_names = ['Mean\nReward', 'Sharpe\nRatio', 'Mean\nLength']
        
        x = np.arange(len(metrics_names))
        width = 0.8 / n_policies
        
        for idx, (results, color) in enumerate(zip(all_results, colors)):
            vals = [results['mean_reward'], results['sharpe_ratio'], 
                   results['mean_episode_length']]
            # Normalize for visualization
            max_vals = [max(abs(r['mean_reward']) for r in all_results),
                       max(abs(r['sharpe_ratio']) for r in all_results if not np.isnan(r['sharpe_ratio'])),
                       max(abs(r['mean_episode_length']) for r in all_results)]
            vals_norm = [v / m if m != 0 and not np.isnan(v) else 0 
                        for v, m in zip(vals, max_vals)]
            
            offset = (idx - n_policies/2 + 0.5) * width
            ax5.bar(x + offset, vals_norm, width, label=results['policy_name'], 
                   color=color, alpha=0.7)
        
        ax5.set_ylabel('Normalized Value', fontsize=11)
        ax5.set_title('Key Metrics Comparison', fontsize=12, fontweight='bold')
        ax5.set_xticks(x)
        ax5.set_xticklabels(metrics_names, fontsize=9)
        ax5.legend(fontsize=9)
        ax5.grid(alpha=0.3, axis='y')
        ax5.axhline(0, color='black', linewidth=0.8)
        
        title = f'Policy Comparison: {" vs ".join(policy_names)}'
        if len(title) > 80:
            title = f'Policy Comparison ({n_policies} policies)'
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
        
        plt.show()
    
    def plot_reward_convergence(
        self,
        window: int = 100,
        figsize: Tuple[int, int] = (12, 5)
    ):
        """
        Plot rolling average of rewards for all evaluated policies.
        
        Args:
            window: Rolling average window size
            figsize: Figure size
        """
        if len(self.results) == 0:
            print("⚠ No evaluated policies to plot")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.results)))
        
        for idx, (name, results) in enumerate(self.results.items()):
            rewards = results['episode_rewards']
            
            # Rolling average
            rolling_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax1.plot(rolling_avg, label=name, color=colors[idx], linewidth=2, alpha=0.8)
            
            # Box plot
            ax2.boxplot([rewards], positions=[idx], labels=[name], widths=0.6)
        
        ax1.set_xlabel('Episode', fontsize=11)
        ax1.set_ylabel('Reward (rolling avg)', fontsize=11)
        ax1.set_title(f'Reward Convergence (window={window})', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        ax2.set_ylabel('Episode Reward', fontsize=11)
        ax2.set_title('Reward Distribution (Box Plot)', fontsize=12, fontweight='bold')
        ax2.grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()
