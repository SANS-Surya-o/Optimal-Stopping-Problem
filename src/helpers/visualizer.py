from matplotlib import pyplot as plt
import numpy as np
from typing import List, Optional, Tuple
import io
import os


class visualizer:
    def __init__(self, env, agent, optimal_policy=None, policy=None, policy_provided=False):
        self.env = env
        self.agent = agent
        self.optimal_policy = env.solve_optimal_policy()[1] if optimal_policy is None else optimal_policy
        self.policy = policy
        self.policy_provided = policy_provided

    def plot_visit_counts(self):
        # visit counts shape: (n_states, max_time+1, max_stops+1, n_actions)
        # somehow visualise offer, time_left and stops left dimensions along with actions - make 2 plots for when stops_left=1 and stops_left=2
        visit_counts = self.agent.visit_counts
        max_time = visit_counts.shape[1]
        time_ticks = np.arange(0, max_time, max(1, max_time//10))  # Positions (0-indexed in plot)
        time_labels = np.arange(0, max_time, max(1, max_time//10))  # Labels (actual time_left values)
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        ims = []
        # stops_left=1, action=0 (reject)
        ims.append(axs[0, 0].imshow(visit_counts[:, :, 1, 0], cmap='viridis', aspect='auto', origin='lower'))
        axs[0, 0].set_title('Visit Counts - Stops Left=1, Reject', fontsize=11, fontweight='bold')
        axs[0, 0].set_xlabel('Time Left', fontsize=10)
        axs[0, 0].set_ylabel('Offer Value Index',   fontsize=10)
        axs[0, 0].set_xticks(time_ticks)
        axs[0, 0].set_xticklabels(time_labels)
        # stops_left=1, action=1 (accept)
        ims.append(axs[0, 1].imshow(visit_counts[:, :, 1, 1], cmap='viridis', aspect='auto', origin='lower'))
        axs[0, 1].set_title('Visit Counts - Stops Left=1, Accept', fontsize=11, fontweight='bold')
        axs[0, 1].set_xlabel('Time Left', fontsize=10)
        axs[0, 1].set_ylabel('Offer Value Index', fontsize=10)
        axs[0, 1].set_xticks(time_ticks)
        axs[0, 1].set_xticklabels(time_labels)
        # stops_left=2, action=0 (reject)
        ims.append(axs[1, 0].imshow(visit_counts[:, :, 2, 0], cmap='viridis', aspect='auto', origin='lower'))
        axs[1, 0].set_title('Visit Counts - Stops Left=2, Reject', fontsize=11, fontweight='bold')
        axs[1, 0].set_xlabel('Time Left', fontsize=10)
        axs[1, 0].set_ylabel('Offer Value Index', fontsize=10)
        axs[1, 0].set_xticks(time_ticks)
        axs[1, 0].set_xticklabels(time_labels)
        # stops_left=2, action=1 (accept)
        ims.append(axs[1, 1].imshow(visit_counts[:, :, 2, 1], cmap='viridis', aspect='auto', origin='lower'))
        axs[1, 1].set_title('Visit Counts - Stops Left=2, Accept', fontsize=11, fontweight='bold')
        axs[1, 1].set_xlabel('Time Left', fontsize=10)
        axs[1, 1].set_ylabel('Offer Value Index', fontsize=10)
        axs[1, 1].set_xticks(time_ticks)
        axs[1, 1].set_xticklabels(time_labels)              
        plt.tight_layout(rect=[0, 0, 0.92, 1])
        cax = fig.add_axes([0.94, 0.15, 0.02, 0.7])  # [left, bottom, width, height] in figure coords
        cb = fig.colorbar(ims[0], cax=cax, orientation='vertical')
        cb.set_label('Visit Count', fontsize=10)
        plt.show() 
        

    def plot_episode_rewards(self, rolling_window=100):
        rewards = self.agent.episode_rewards
        # rolling average over 100 episodes
        rewards = np.convolve(rewards, np.ones(rolling_window)/rolling_window, mode='valid')
        plt.plot(rewards)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Episode Rewards over Time')
        plt.show()

    def plot_optimal(self):
        """
        Visualize the optimal policy and value function from DP.
        Shows buy/sell policies and V(s,t,stops) heatmaps.
        """
        V, Q, policy = self.env.solve_optimal_policy(verbose=False)
        
        # Exclude time_left=0
        policy_view = policy[:, 1:, :]
        V_view = V[:, 1:, :]
        
        max_time = policy_view.shape[1]
        time_ticks = np.arange(0, max_time, max(1, max_time//10))
        time_labels = np.arange(1, max_time+1, max(1, max_time//10))
        
        # --- Plot 1: Optimal Policies ---
        fig, axs = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Optimal Policy (from DP)', fontsize=14, fontweight='bold')
        
        # Buy policy (stops_left=2)
        im1 = axs[0].imshow(policy_view[:, :, 2], cmap='RdYlGn', vmin=0, vmax=1, 
                           aspect='auto', origin='lower')
        axs[0].set_xlabel('Time Left', fontsize=11)
        axs[0].set_ylabel('Offer Value Index', fontsize=11)
        axs[0].set_title(f'Buy Policy (stops_left=2)\nAccept Rate: {policy_view[:,:,2].mean():.1%}', 
                        fontsize=11, fontweight='bold')
        axs[0].set_xticks(time_ticks)
        axs[0].set_xticklabels(time_labels)
        cbar1 = plt.colorbar(im1, ax=axs[0], orientation='vertical', pad=0.02)
        cbar1.set_ticks([0, 1])
        cbar1.set_ticklabels(['Reject', 'Accept'])
        
        # Sell policy (stops_left=1)
        im2 = axs[1].imshow(policy_view[:, :, 1], cmap='RdYlGn', vmin=0, vmax=1, 
                           aspect='auto', origin='lower')
        axs[1].set_xlabel('Time Left', fontsize=11)
        axs[1].set_ylabel('Offer Value Index', fontsize=11)
        axs[1].set_title(f'Sell Policy (stops_left=1)\nAccept Rate: {policy_view[:,:,1].mean():.1%}', 
                        fontsize=11, fontweight='bold')
        axs[1].set_xticks(time_ticks)
        axs[1].set_xticklabels(time_labels)
        cbar2 = plt.colorbar(im2, ax=axs[1], orientation='vertical', pad=0.02)
        cbar2.set_ticks([0, 1])
        cbar2.set_ticklabels(['Reject', 'Accept'])
        
        plt.tight_layout()
        plt.show()
        
        # --- Plot 2: Value Functions ---
        fig, axs = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Optimal Value Function V(s, t, stops)', fontsize=14, fontweight='bold')
        
        # V for stops_left=2 (before buying)
        im1 = axs[0].imshow(V_view[:, :, 2], cmap='viridis', aspect='auto', origin='lower')
        axs[0].set_xlabel('Time Left', fontsize=11)
        axs[0].set_ylabel('Offer Value Index', fontsize=11)
        axs[0].set_title('V(s, t, stops=2) - Before Buying', fontsize=11, fontweight='bold')
        axs[0].set_xticks(time_ticks)
        axs[0].set_xticklabels(time_labels)
        plt.colorbar(im1, ax=axs[0], label='Value')
        
        # V for stops_left=1 (holding, before selling)
        im2 = axs[1].imshow(V_view[:, :, 1], cmap='viridis', aspect='auto', origin='lower')
        axs[1].set_xlabel('Time Left', fontsize=11)
        axs[1].set_ylabel('Offer Value Index', fontsize=11)
        axs[1].set_title('V(s, t, stops=1) - Holding (Before Selling)', fontsize=11, fontweight='bold')
        axs[1].set_xticks(time_ticks)
        axs[1].set_xticklabels(time_labels)
        plt.colorbar(im2, ax=axs[1], label='Value')
        
        plt.tight_layout()
        plt.show()
        
        # Print summary
        print(f"\nOptimal Value Function Summary:")
        print(f"  V(s, max_time, 2) range: [{V[:, -1, 2].min():.2f}, {V[:, -1, 2].max():.2f}]")
        print(f"  V(s, max_time, 1) range: [{V[:, -1, 1].min():.2f}, {V[:, -1, 1].max():.2f}]")
        print(f"  Expected value at start (avg over states): {V[:, -1, 2].mean():.4f}")

    def plot_value_function(self, V: Optional[np.ndarray] = None):
        """
        Plot the value function V(s, t, stops) as heatmaps.
        
        Args:
            V: Value function array of shape (n_states, max_time+1, max_stops+1).
               If None, computes optimal V from DP.
        """
        if V is None:
            V, _, _ = self.env.solve_optimal_policy(verbose=False)
        
        # Exclude time_left=0
        V_view = V[:, 1:, :]
        
        max_time = V_view.shape[1]
        time_ticks = np.arange(0, max_time, max(1, max_time//10))
        time_labels = np.arange(1, max_time+1, max(1, max_time//10))
        
        # Common color scale
        vmin = V_view[:, :, 1:].min()
        vmax = V_view[:, :, 1:].max()
        
        fig, axs = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Value Function V(s, t, stops)', fontsize=14, fontweight='bold')
        
        # V for stops_left=2
        im1 = axs[0].imshow(V_view[:, :, 2], cmap='viridis', aspect='auto', origin='lower',
                           vmin=vmin, vmax=vmax)
        axs[0].set_xlabel('Time Left', fontsize=11)
        axs[0].set_ylabel('Offer Value Index', fontsize=11)
        axs[0].set_title('V(s, t, stops=2)\nBefore Buying', fontsize=11, fontweight='bold')
        axs[0].set_xticks(time_ticks)
        axs[0].set_xticklabels(time_labels)
        plt.colorbar(im1, ax=axs[0], label='Value')
        
        # V for stops_left=1
        im2 = axs[1].imshow(V_view[:, :, 1], cmap='viridis', aspect='auto', origin='lower',
                           vmin=vmin, vmax=vmax)
        axs[1].set_xlabel('Time Left', fontsize=11)
        axs[1].set_ylabel('Offer Value Index', fontsize=11)
        axs[1].set_title('V(s, t, stops=1)\nHolding (Before Selling)', fontsize=11, fontweight='bold')
        axs[1].set_xticks(time_ticks)
        axs[1].set_xticklabels(time_labels)
        plt.colorbar(im2, ax=axs[1], label='Value')
        
        plt.tight_layout()
        plt.show()

    def plot_Q_values(self, Q: Optional[np.ndarray] = None):
        """
        Plot Q-values Q(s, t, stops, action) as heatmaps.
        
        Args:
            Q: Q-value array of shape (n_states, max_time+1, max_stops+1, 2).
               If None, computes optimal Q from DP.
        """
        if Q is None:
            _, Q, _ = self.env.solve_optimal_policy(verbose=False)
        
        # Exclude time_left=0
        Q_view = Q[:, 1:, :, :]
        
        max_time = Q_view.shape[1]
        time_ticks = np.arange(0, max_time, max(1, max_time//10))
        time_labels = np.arange(1, max_time+1, max(1, max_time//10))
        
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Optimal Q(s, t, stops, action)', fontsize=14, fontweight='bold')
        
        # Row 0: Buy phase (stops_left=2)
        im1 = axs[0, 0].imshow(Q_view[:, :, 2, 0], cmap='viridis', aspect='auto', origin='lower')
        axs[0, 0].set_title('Q(s, t, stops=2, continue)', fontsize=11, fontweight='bold')
        axs[0, 0].set_xlabel('Time Left')
        axs[0, 0].set_ylabel('State')
        axs[0, 0].set_xticks(time_ticks)
        axs[0, 0].set_xticklabels(time_labels)
        plt.colorbar(im1, ax=axs[0, 0])
        
        im2 = axs[0, 1].imshow(Q_view[:, :, 2, 1], cmap='viridis', aspect='auto', origin='lower')
        axs[0, 1].set_title('Q(s, t, stops=2, buy)', fontsize=11, fontweight='bold')
        axs[0, 1].set_xlabel('Time Left')
        axs[0, 1].set_ylabel('State')
        axs[0, 1].set_xticks(time_ticks)
        axs[0, 1].set_xticklabels(time_labels)
        plt.colorbar(im2, ax=axs[0, 1])
        
        # Row 1: Sell phase (stops_left=1)
        im3 = axs[1, 0].imshow(Q_view[:, :, 1, 0], cmap='viridis', aspect='auto', origin='lower')
        axs[1, 0].set_title('Q(s, t, stops=1, continue)', fontsize=11, fontweight='bold')
        axs[1, 0].set_xlabel('Time Left')
        axs[1, 0].set_ylabel('State')
        axs[1, 0].set_xticks(time_ticks)
        axs[1, 0].set_xticklabels(time_labels)
        plt.colorbar(im3, ax=axs[1, 0])
        
        im4 = axs[1, 1].imshow(Q_view[:, :, 1, 1], cmap='viridis', aspect='auto', origin='lower')
        axs[1, 1].set_title('Q(s, t, stops=1, sell)', fontsize=11, fontweight='bold')
        axs[1, 1].set_xlabel('Time Left')
        axs[1, 1].set_ylabel('State')
        axs[1, 1].set_xticks(time_ticks)
        axs[1, 1].set_xticklabels(time_labels)
        plt.colorbar(im4, ax=axs[1, 1])
        
        plt.tight_layout()
        plt.show()


    def visualise_Q_table(self):
        Q_table = self.agent.Q
        # Exclude time_left=0 by slicing from index 1
        Q_table_view = Q_table[:, 1:, :, :]  # Start from time_left=1
        
        # ensure a common color scale across all subplots
        vmin = np.nanmin(Q_table_view)
        vmax = np.nanmax(Q_table_view)
        
        # Get the actual time_left range (1 to max_time)
        max_time = Q_table_view.shape[1]
        time_ticks = np.arange(0, max_time, max(1, max_time//10))  # Positions (0-indexed in plot)
        time_labels = np.arange(1, max_time+1, max(1, max_time//10))  # Labels (actual time_left values)

        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        ims = []
        
        # Yet to sell (stops_left=1), reject
        ims.append(axs[0, 0].imshow(Q_table_view[:, :, 1, 0], cmap='viridis', vmin=vmin, vmax=vmax, aspect='auto', origin='lower'))
        axs[0, 0].set_title('Q-values - Yet to sell, reject', fontsize=11, fontweight='bold')
        axs[0, 0].set_xlabel('Time Left', fontsize=10)
        axs[0, 0].set_ylabel('Offer Value Index', fontsize=10)
        axs[0, 0].set_xticks(time_ticks)
        axs[0, 0].set_xticklabels(time_labels)

        # Yet to sell (stops_left=1), accept
        ims.append(axs[0, 1].imshow(Q_table_view[:, :, 1, 1], cmap='viridis', vmin=vmin, vmax=vmax, aspect='auto', origin='lower'))
        axs[0, 1].set_title('Q-values - Yet to sell, accept', fontsize=11, fontweight='bold')
        axs[0, 1].set_xlabel('Time Left', fontsize=10)
        axs[0, 1].set_ylabel('Offer Value Index', fontsize=10)
        axs[0, 1].set_xticks(time_ticks)
        axs[0, 1].set_xticklabels(time_labels)

        # Yet to buy (stops_left=2), reject
        ims.append(axs[1, 0].imshow(Q_table_view[:, :, 2, 0], cmap='viridis', vmin=vmin, vmax=vmax, aspect='auto', origin='lower'))
        axs[1, 0].set_title('Q-values - Yet to buy, reject', fontsize=11, fontweight='bold')
        axs[1, 0].set_xlabel('Time Left', fontsize=10)
        axs[1, 0].set_ylabel('Offer Value Index', fontsize=10)
        axs[1, 0].set_xticks(time_ticks)
        axs[1, 0].set_xticklabels(time_labels)

        # Yet to buy (stops_left=2), accept
        ims.append(axs[1, 1].imshow(Q_table_view[:, :, 2, 1], cmap='viridis', vmin=vmin, vmax=vmax, aspect='auto', origin='lower'))
        axs[1, 1].set_title('Q-values - Yet to buy, accept', fontsize=11, fontweight='bold')
        axs[1, 1].set_xlabel('Time Left', fontsize=10)
        axs[1, 1].set_ylabel('Offer Value Index', fontsize=10)
        axs[1, 1].set_xticks(time_ticks)
        axs[1, 1].set_xticklabels(time_labels)

        # reserve space on the right for the colorbar and use a dedicated axis so the colorbar aligns nicely
        plt.tight_layout(rect=[0, 0, 0.92, 1])
        cax = fig.add_axes([0.94, 0.15, 0.02, 0.7])  # [left, bottom, width, height] in figure coords
        cb = fig.colorbar(ims[0], cax=cax, orientation='vertical')
        cb.set_label('Q-value', fontsize=10)

        plt.show()

    def visualise_policy(self):
        """
        Visualize the learned policy for buy and sell decisions.
        Shows accept (1) vs reject (0) decisions based on argmax of Q-values.
        """
        # Get policy from Q-table (argmax over actions)
        if self.policy_provided:
            policy = self.policy
        else:
            policy = self.agent.get_policy()  # shape: (n_states, max_time+1, max_stops+1)

        # Create figure with two subplots (one for buy, one for sell)
        fig, axs = plt.subplots(1, 2, figsize=(14, 5))
        
        # Policy for buying (stops_left = 2), exclude time_left=0
        policy_buy = policy[:, 1:, 2]  # (n_states, max_time)
        max_time = policy_buy.shape[1]
        time_ticks = np.arange(0, max_time, max(1, max_time//10))
        time_labels = np.arange(1, max_time+1, max(1, max_time//10))
        
        im1 = axs[0].imshow(policy_buy, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto', origin='lower')
        axs[0].set_xlabel('Time Left', fontsize=12)
        axs[0].set_ylabel('Offer Value Index', fontsize=12)
        axs[0].set_title('Buy Policy (First Stop)\nGreen=Accept, Red=Reject', fontsize=13, fontweight='bold')
        axs[0].grid(False)
        axs[0].set_xticks(time_ticks)
        axs[0].set_xticklabels(time_labels)
        
        # Add colorbar for buy policy
        cbar1 = plt.colorbar(im1, ax=axs[0], orientation='vertical', pad=0.02)
        cbar1.set_ticks([0, 1])
        cbar1.set_ticklabels(['Reject (0)', 'Accept (1)'])
        
        # Policy for selling (stops_left = 1), exclude time_left=0
        policy_sell = policy[:, 1:, 1]  # (n_states, max_time)
        im2 = axs[1].imshow(policy_sell, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto', origin='lower')
        axs[1].set_xlabel('Time Left', fontsize=12)
        axs[1].set_ylabel('Offer Value Index', fontsize=12)
        axs[1].set_title('Sell Policy (Second Stop)\nGreen=Accept, Red=Reject', fontsize=13, fontweight='bold')
        axs[1].grid(False)
        axs[1].set_xticks(time_ticks)
        axs[1].set_xticklabels(time_labels)
        
        # Add colorbar for sell policy
        cbar2 = plt.colorbar(im2, ax=axs[1], orientation='vertical', pad=0.02)
        cbar2.set_ticks([0, 1])
        cbar2.set_ticklabels(['Reject (0)', 'Accept (1)'])
        
        plt.tight_layout()
        plt.show()
        
        # Print some statistics
        print("\nQ-Learning Policy Statistics:")
        print(f"Buy Policy - Accept rate: {policy_buy.mean():.2%}")
        print(f"Sell Policy - Accept rate: {policy_sell.mean():.2%}")

    def compare_with_optimal(self):
        """
        Compare Q-learning policy with optimal DP policy.
        """
        if self.optimal_policy is None:
            print("⚠ Optimal policy not provided.")
            print("  Compute it first: V, opt_policy = env.solve_optimal_policy()")
            print("  Then: viz = visualizer(env, agent, optimal_policy=opt_policy)")
            return
        
        # Get policies
        if self.policy_provided:
            q_policy = self.policy  # Q-learning policy provided
        else:
            q_policy = self.agent.get_policy()  # Q-learning policy
        opt_policy = self.optimal_policy  # Optimal policy
        
        # Exclude time_left=0
        q_policy_view = q_policy[:, 1:, :]
        opt_policy_view = opt_policy[:, 1:, :]
        
        max_time = q_policy_view.shape[1]
        time_ticks = np.arange(0, max_time, max(1, max_time//10))
        time_labels = np.arange(1, max_time+1, max(1, max_time//10))
        
        # Create comparison figure
        fig, axs = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Policy Comparison: Q-Learning vs Optimal DP', fontsize=16, fontweight='bold')
        
        # Row 1: Buy policies (stops_left=2)
        # Q-learning buy policy
        im1 = axs[0, 0].imshow(q_policy_view[:, :, 2], cmap='RdYlGn', vmin=0, vmax=1, aspect='auto', origin='lower')
        axs[0, 0].set_title('Q-Learning: Buy Policy', fontsize=11, fontweight='bold')
        axs[0, 0].set_xlabel('Time Left', fontsize=10)
        axs[0, 0].set_ylabel('Offer Index', fontsize=10)
        axs[0, 0].set_xticks(time_ticks)
        axs[0, 0].set_xticklabels(time_labels)
        plt.colorbar(im1, ax=axs[0, 0], ticks=[0, 1], label='Action')
        
        # Optimal buy policy
        im2 = axs[0, 1].imshow(opt_policy_view[:, :, 2], cmap='RdYlGn', vmin=0, vmax=1, aspect='auto', origin='lower')
        axs[0, 1].set_title('Optimal: Buy Policy', fontsize=11, fontweight='bold')
        axs[0, 1].set_xlabel('Time Left', fontsize=10)
        axs[0, 1].set_ylabel('Offer Index', fontsize=10)
        axs[0, 1].set_xticks(time_ticks)
        axs[0, 1].set_xticklabels(time_labels)
        plt.colorbar(im2, ax=axs[0, 1], ticks=[0, 1], label='Action')
        
        # Difference (buy)
        diff_buy = (q_policy_view[:, :, 2] != opt_policy_view[:, :, 2]).astype(float)
        im3 = axs[0, 2].imshow(diff_buy, cmap='Reds', vmin=0, vmax=1, aspect='auto', origin='lower')
        axs[0, 2].set_title(f'Difference (Buy)\nMismatch: {diff_buy.mean():.1%}', fontsize=11, fontweight='bold')
        axs[0, 2].set_xlabel('Time Left', fontsize=10)
        axs[0, 2].set_ylabel('Offer Index', fontsize=10)
        axs[0, 2].set_xticks(time_ticks)
        axs[0, 2].set_xticklabels(time_labels)
        plt.colorbar(im3, ax=axs[0, 2], ticks=[0, 1], label='Different')
        
        # Row 2: Sell policies (stops_left=1)
        # Q-learning sell policy
        im4 = axs[1, 0].imshow(q_policy_view[:, :, 1], cmap='RdYlGn', vmin=0, vmax=1, aspect='auto', origin='lower')
        axs[1, 0].set_title('Q-Learning: Sell Policy', fontsize=11, fontweight='bold')
        axs[1, 0].set_xlabel('Time Left', fontsize=10)
        axs[1, 0].set_ylabel('Offer Index', fontsize=10)
        axs[1, 0].set_xticks(time_ticks)
        axs[1, 0].set_xticklabels(time_labels)
        plt.colorbar(im4, ax=axs[1, 0], ticks=[0, 1], label='Action')
        
        # Optimal sell policy
        im5 = axs[1, 1].imshow(opt_policy_view[:, :, 1], cmap='RdYlGn', vmin=0, vmax=1, aspect='auto', origin='lower')
        axs[1, 1].set_title('Optimal: Sell Policy', fontsize=11, fontweight='bold')
        axs[1, 1].set_xlabel('Time Left', fontsize=10)
        axs[1, 1].set_ylabel('Offer Index', fontsize=10)
        axs[1, 1].set_xticks(time_ticks)
        axs[1, 1].set_xticklabels(time_labels)
        plt.colorbar(im5, ax=axs[1, 1], ticks=[0, 1], label='Action')
        
        # Difference (sell)
        diff_sell = (q_policy_view[:, :, 1] != opt_policy_view[:, :, 1]).astype(float)
        im6 = axs[1, 2].imshow(diff_sell, cmap='Reds', vmin=0, vmax=1, aspect='auto', origin='lower')
        axs[1, 2].set_title(f'Difference (Sell)\nMismatch: {diff_sell.mean():.1%}', fontsize=11, fontweight='bold')
        axs[1, 2].set_xlabel('Time Left', fontsize=10)
        axs[1, 2].set_ylabel('Offer Index', fontsize=10)
        axs[1, 2].set_xticks(time_ticks)
        axs[1, 2].set_xticklabels(time_labels)
        plt.colorbar(im6, ax=axs[1, 2], ticks=[0, 1], label='Different')
        
        plt.tight_layout()
        plt.show()
        
        # Print detailed statistics
        print("\n" + "="*60)
        print("Policy Comparison Statistics")
        print("="*60)
        print(f"\nBuy Policy (stops_left=2):")
        print(f"  Q-Learning accept rate: {q_policy_view[:, :, 2].mean():.2%}")
        print(f"  Optimal accept rate: {opt_policy_view[:, :, 2].mean():.2%}")
        print(f"  Agreement: {(1 - diff_buy.mean()):.2%}")
        print(f"  Mismatch: {diff_buy.mean():.2%}")
        
        print(f"\nSell Policy (stops_left=1):")
        print(f"  Q-Learning accept rate: {q_policy_view[:, :, 1].mean():.2%}")
        print(f"  Optimal accept rate: {opt_policy_view[:, :, 1].mean():.2%}")
        print(f"  Agreement: {(1 - diff_sell.mean()):.2%}")
        print(f"  Mismatch: {diff_sell.mean():.2%}")
        
        # Overall agreement
        total_agreement = 1 - (diff_buy.sum() + diff_sell.sum()) / (diff_buy.size + diff_sell.size)
        print(f"\nOverall Policy Agreement: {total_agreement:.2%}")
        print("="*60)
    
    def pipeline(self, compare_optimal=True):
        self.plot_visit_counts()
        self.plot_episode_rewards()
        self.visualise_Q_table()
        self.visualise_policy()
        if compare_optimal and self.optimal_policy is not None:
            self.compare_with_optimal()

    @staticmethod
    def create_policy_evolution_gif(
        snapshots: List[np.ndarray],
        episodes: List[int],
        rewards: List[float],
        output_path: str = "policy_evolution.gif",
        optimal_policy: Optional[np.ndarray] = None,
        fps: int = 5,
        dpi: int = 100,
        figsize: Tuple[int, int] = (14, 6)
    ) -> str:
        """
        Create an animated GIF showing how buy/sell policies evolve during training.
        
        Args:
            snapshots: List of policy arrays captured during training.
                      Each has shape (n_states, max_time+1, max_stops+1)
            episodes: List of episode numbers corresponding to each snapshot
            rewards: List of rolling average rewards at each snapshot
            output_path: Path to save the GIF file
            optimal_policy: Optional optimal policy to show agreement percentage
            fps: Frames per second for the GIF
            dpi: Resolution of each frame
            figsize: Figure size (width, height) in inches
            
        Returns:
            Path to the saved GIF file
        """
        try:
            from PIL import Image
        except ImportError:
            raise ImportError(
                "PIL (Pillow) is required for GIF creation. "
                "Install with: pip install Pillow"
            )
        
        if len(snapshots) == 0:
            raise ValueError("No snapshots provided. Run training with PolicySnapshotCallback first.")
        
        print(f"Creating GIF with {len(snapshots)} frames...")
        
        frames = []
        
        for i, (policy, episode, reward) in enumerate(zip(snapshots, episodes, rewards)):
            # Create figure for this frame
            fig, axs = plt.subplots(1, 2, figsize=figsize)
            
            # Exclude time_left=0
            policy_buy = policy[:, 1:, 2]   # Buy policy (stops_left=2)
            policy_sell = policy[:, 1:, 1]  # Sell policy (stops_left=1)
            
            max_time = policy_buy.shape[1]
            time_ticks = np.arange(0, max_time, max(1, max_time//10))
            time_labels = np.arange(1, max_time+1, max(1, max_time//10))
            
            # Buy policy (left subplot)
            im1 = axs[0].imshow(policy_buy, cmap='RdYlGn', vmin=0, vmax=1, 
                               aspect='auto', origin='lower')
            axs[0].set_xlabel('Time Left', fontsize=11)
            axs[0].set_ylabel('Offer Value Index', fontsize=11)
            axs[0].set_xticks(time_ticks)
            axs[0].set_xticklabels(time_labels)
            
            buy_title = f'Buy Policy\nAccept Rate: {policy_buy.mean():.1%}'
            if optimal_policy is not None:
                opt_buy = optimal_policy[:, 1:, 2]
                buy_agreement = (policy_buy == opt_buy).mean()
                buy_title += f' | Agreement: {buy_agreement:.1%}'
            axs[0].set_title(buy_title, fontsize=11, fontweight='bold')
            
            cbar1 = plt.colorbar(im1, ax=axs[0], orientation='vertical', pad=0.02)
            cbar1.set_ticks([0, 1])
            cbar1.set_ticklabels(['Reject', 'Accept'])
            
            # Sell policy (right subplot)
            im2 = axs[1].imshow(policy_sell, cmap='RdYlGn', vmin=0, vmax=1, 
                               aspect='auto', origin='lower')
            axs[1].set_xlabel('Time Left', fontsize=11)
            axs[1].set_ylabel('Offer Value Index', fontsize=11)
            axs[1].set_xticks(time_ticks)
            axs[1].set_xticklabels(time_labels)
            
            sell_title = f'Sell Policy\nAccept Rate: {policy_sell.mean():.1%}'
            if optimal_policy is not None:
                opt_sell = optimal_policy[:, 1:, 1]
                sell_agreement = (policy_sell == opt_sell).mean()
                sell_title += f' | Agreement: {sell_agreement:.1%}'
            axs[1].set_title(sell_title, fontsize=11, fontweight='bold')
            
            cbar2 = plt.colorbar(im2, ax=axs[1], orientation='vertical', pad=0.02)
            cbar2.set_ticks([0, 1])
            cbar2.set_ticklabels(['Reject', 'Accept'])
            
            # Main title with episode and reward info
            main_title = f'Policy Evolution - Episode {episode:,}'
            if reward != 0:
                main_title += f' | Avg Reward: {reward:.4f}'
            if optimal_policy is not None:
                overall_agreement = ((policy_buy == opt_buy).sum() + (policy_sell == opt_sell).sum()) / \
                                   (policy_buy.size + policy_sell.size)
                main_title += f' | Overall Agreement: {overall_agreement:.1%}'
            
            fig.suptitle(main_title, fontsize=13, fontweight='bold', y=1.02)
            
            plt.tight_layout()
            
            # Save frame to buffer
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
            buf.seek(0)
            frames.append(Image.open(buf).copy())
            buf.close()
            plt.close(fig)
            
            # Progress indicator
            if (i + 1) % 10 == 0 or i == len(snapshots) - 1:
                print(f"  Rendered frame {i + 1}/{len(snapshots)}")
        
        # Save as GIF
        duration = int(1000 / fps)  # Duration per frame in milliseconds
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=0  # 0 = infinite loop
        )
        
        print(f"✓ GIF saved to: {output_path}")
        print(f"  - {len(frames)} frames at {fps} FPS")
        print(f"  - Duration: {len(frames) / fps:.1f} seconds")
        
        return output_path

    @staticmethod
    def create_policy_evolution_gif_with_callback(
        callback,  # PolicySnapshotCallback instance
        output_path: str = "policy_evolution.gif",
        optimal_policy: Optional[np.ndarray] = None,
        fps: int = 5,
        dpi: int = 100,
        figsize: Tuple[int, int] = (14, 6)
    ) -> str:
        """
        Convenience method to create GIF directly from a PolicySnapshotCallback.
        
        Args:
            callback: PolicySnapshotCallback instance that was used during training.
                      Must have a get_snapshots() method returning (policies, episodes, rewards).
            output_path: Path to save the GIF file
            optimal_policy: Optional optimal policy to show agreement percentage
            fps: Frames per second for the GIF
            dpi: Resolution of each frame
            figsize: Figure size (width, height) in inches
            
        Returns:
            Path to the saved GIF file
            
        Example:
            >>> from callbacks import PolicySnapshotCallback
            >>> snapshot_cb = PolicySnapshotCallback(snapshot_interval=500)
            >>> agent.train(env, n_episodes=10000, callbacks=[snapshot_cb])
            >>> visualizer.create_policy_evolution_gif_with_callback(
            ...     snapshot_cb, 
            ...     output_path="training_evolution.gif",
            ...     optimal_policy=opt_policy
            ... )
        """
        snapshots, episodes, rewards = callback.get_snapshots()
        return visualizer.create_policy_evolution_gif(
            snapshots=snapshots,
            episodes=episodes,
            rewards=rewards,
            output_path=output_path,
            optimal_policy=optimal_policy,
            fps=fps,
            dpi=dpi,
            figsize=figsize
        )
