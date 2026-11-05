import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate


def plot_values(V_1, V_2, state):
    horizon = V_1.shape[0]
    
    time_steps_v1 = np.arange(1, horizon - 1)  # skip first value
    time_steps_v2 = np.arange(horizon - 2) 
    fig, ax = plt.subplots(figsize=(12, 6))
    time_steps = np.arange(horizon)
    ax.plot(time_steps_v1, V_1[1:-1, state], marker="o", label="V1")
    ax.plot(time_steps_v2, V_2[:-2, state], marker="o", label="V2")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Value")
    ax.set_title(f"Values for State {state}")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()


def plot_q_values(Q, state, env, title=None):
    """
    Plot Q-values for a specific state across all time steps, actions, and z-values.
    Each (action, z) pair uses a different color.
    """
    horizon = Q.shape[0]
    action_names = env.actions
    num_actions = len(action_names)

    fig, ax = plt.subplots(figsize=(12, 6))
    time_steps = np.arange(horizon)
    # Use 4 distinct colors for the 4 (action, z) pairs
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    line_idx = 0

    for a_idx, action_name in enumerate(action_names):
        for z_index, z_val in enumerate([2, 1]):  # z=2 then z=1
            color = colors[line_idx % len(colors)]
            ax.plot(
                time_steps,
                Q[:, state, z_index, a_idx],
                marker="o",
                label=f"Action: {action_name}, z={z_val}",
                color=color,
            )
            line_idx += 1

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Q-Value")
    ax.set_title(title or f"Q-values for State {state} (Value: {env.states[state]})")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    return fig, ax


def plot_final_q_values(agent, states_to_plot=None):
    env = agent.env
    Q = agent.Q
    if states_to_plot is None:
        max_state = len(env.states) - 1
        states_to_plot = [
            0,
            max_state // 4,
            max_state // 2,
            3 * max_state // 4,
            max_state,
        ]

    fig, ax = plt.subplots(figsize=(14, 8))
    line_styles = ["-", "--"]
    z_labels = [2, 1]
    markers = ["o", "s", "^", "D", "x"]
    colors = ["blue", "green"]

    for i, state in enumerate(states_to_plot):
        for a_idx, action_name in enumerate(env.actions):
            for z_index, z in enumerate(z_labels):
                ax.plot(
                    range(Q.shape[0]),
                    Q[:, state, z_index, a_idx],
                    linestyle=line_styles[z_index],
                    color=colors[a_idx],
                    marker=markers[i % len(markers)],
                    label=f"S={env.states[state]} - {action_name}, z={z}",
                )

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Q-Value")
    ax.set_title("Q-values for Different States, Actions, and z")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
    ax.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    return fig, ax


def print_q_table(
    Q, env, time_step=0, states_to_show=None, headers="firstrow", tablefmt="grid"
):
    if states_to_show is None:
        states_to_show = list(range(len(env.states)))
    action_names = env.actions
    table_data = [["State Value", "z"] + action_names]

    for state in states_to_show:
        state_value = env.states[state]
        for z_index, z in enumerate([2, 1]):
            row = [f"S={state_value}", f"{z}"]
            for a_idx in range(len(action_names)):
                q_value = Q[time_step, state, z_index, a_idx]
                row.append(f"{q_value:.4f}")
            table_data.append(row)

    print(f"Q-values at Time Step {time_step}:")
    print(tabulate(table_data, headers=headers, tablefmt=tablefmt))


def print_q_table_all_timesteps(
    Q, env, states_to_show=None, actions_to_show=None, tablefmt="grid"
):
    if states_to_show is None:
        max_state = len(env.states) - 1
        states_to_show = list(range(len(env.states)))
        # states_to_show = [0, max_state // 4, max_state // 2, 3 * max_state // 4, max_state]

    if actions_to_show is None:
        actions_to_show = list(range(len(env.actions)))

    action_names = [env.actions[i] for i in actions_to_show]
    horizon = Q.shape[0]
    header_row = ["State, z, Action"] + list(range(horizon))

    for state in states_to_show:
        state_value = env.states[state]
        for z in [1, 2]:
            for a_idx in actions_to_show:
                action_name = env.actions[a_idx]
                row = [f"S={state_value}, z={z}, A={action_name}"]
                for t in range(horizon):
                    q_value = Q[t, state, z - 1, a_idx]
                    row.append(f"{q_value:.4f}")
                print(
                    tabulate([header_row, row], headers="firstrow", tablefmt=tablefmt)
                )
                print()


def print_optimal_actions(Q, env, time_step=0, states_to_show=None, tablefmt="grid"):
    if states_to_show is None:
        states_to_show = list(range(len(env.states)))

    table_data = [["State Value", "z", "Optimal Action", "Q-Value"]]

    for state in states_to_show:
        state_value = env.states[state]
        for z_index, z in enumerate([2, 1]):
            q_values = Q[time_step, state, z_index]
            optimal_action_idx = np.argmax(q_values)
            optimal_action_name = env.actions[optimal_action_idx]
            optimal_q_value = q_values[optimal_action_idx]
            table_data.append(
                [f"{state_value}", z, optimal_action_name, f"{optimal_q_value:.4f}"]
            )

    print(f"Optimal Actions at Time Step {time_step}:")
    print(tabulate(table_data, headers="firstrow", tablefmt=tablefmt))


def plot_q_history_for_state(Q_values_history, state, time, env, title=None):
    """
    Plot the evolution of Q-values across training updates for a given state,
    showing separate lines for each (action, z) pair, each in a different color.

    Parameters:
    -----------
    Q_values_history : list of np.ndarray
        History of Q-arrays from the agent, each of shape (T, S, 2, A)
    state : int
        The state index to visualize
    time : int
        The time step to visualize
    env : environment object
        Should have .states (list of state values) and .actions (list of action names)
    title : str, optional
        Custom title for the plot

    Returns:
    --------
    fig, ax : matplotlib figure and axis
    """
    num_updates = len(Q_values_history)
    action_names = env.actions
    num_actions = len(action_names)
    z_values = [1, 2]  # z=1 means one stopping left, z=2 means two stoppings left

    fig, ax = plt.subplots(figsize=(12, 6))
    # Use a color for each (action, z) pair
    import matplotlib.cm as cm

    num_lines = num_actions * len(z_values)
    color_map = cm.get_cmap("tab10", num_lines)
    line_idx = 0

    for a_idx, action_name in enumerate(action_names):
        for z_index, z in enumerate(z_values):
            values = [Q[time, state, z_index, a_idx] for Q in Q_values_history]
            label = f"Action: {action_name}, z={z}"
            ax.plot(
                range(num_updates),
                values,
                color=color_map(line_idx),
                linestyle="-",
                label=label,
            )
            line_idx += 1

    ax.set_xlabel("Update Step")
    ax.set_ylabel("Q-Value")
    ax.set_title(
        title or f"Q-Value History for State {state} (Value: {env.states[state]})"
    )
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend()
    plt.tight_layout()
    return fig, ax


def plot_all_values(V_1_vi, V_2_vi, V_1_q, V_2_q, state):
    horizon = V_1_vi.shape[0]
   
    time_steps_v1 = np.arange(1, horizon - 1)  # skip first value
    time_steps_v2 = np.arange(horizon - 2)  # skip last value

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(time_steps_v1, V_1_vi[1 : horizon - 1, state], marker="o", label="V1 (VI)")
    ax.plot(time_steps_v2, V_2_vi[: horizon - 2, state], marker="o", label="V2 (VI)")
    ax.plot(time_steps_v1, V_1_q[1 : horizon - 1, state], marker="x", label="V1 (Q)")
    ax.plot(time_steps_v2, V_2_q[: horizon - 2, state], marker="x", label="V2 (Q)")

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Value")
    ax.set_title(f"Value Functions for State {state}")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


# Example usage:
# agent, env, Q, history = run_experiment()
#
# # Print Q-values for time step 0
# print_q_table(Q, env, time_step=0, states_to_show=[0, 10, 20, 30, 40, 50])
#
# # Print Q-values for all time steps for selected states
# print_q_table_all_timesteps(Q, env, states_to_show=[0, 50, 100])
#
# # Print optimal actions for time step 0
# print_optimal_actions(Q, env, time_step=0, states_to_show=list(range(0, 101, 10)))
