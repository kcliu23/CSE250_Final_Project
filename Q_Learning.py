import gymnasium as gym
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches
import pandas as pd
import os


if not os.path.exists("./results_Q"):
    os.makedirs("./results_Q")

def initialize_q_table(state_size, action_size):
    return np.zeros((state_size, action_size))

def choose_action(env, q_table, state, epsilon):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample() 
    else:
        return np.argmax(q_table[state, :]) 

def update_q_value(q_table, state, action, reward, new_state, terminated,
                   learning_rate, discount_factor):
    
    if terminated:
        target = reward 
    else:
        target = reward + discount_factor * np.max(q_table[new_state, :]) 

    td_error = target - q_table[state, action]
    q_table[state, action] = q_table[state, action] + learning_rate * td_error
    
    return q_table, td_error

def train_agent(env, q_table, hyperparameters, verbose=True):
    if verbose:
        print(f"Training with: alpha={hyperparameters['learning_rate']}, gamma={hyperparameters['discount_factor']}...")
    
    all_rewards = []
    all_episode_lengths = []
    all_td_errors = []
    
    num_episodes = hyperparameters['num_episodes']
    max_steps_per_episode = hyperparameters['max_steps_per_episode']
    learning_rate = hyperparameters['learning_rate']
    discount_factor = hyperparameters['discount_factor']
    epsilon = hyperparameters['epsilon']
    max_epsilon = hyperparameters['max_epsilon']
    min_epsilon = hyperparameters['min_epsilon']
    epsilon_decay_rate = hyperparameters['epsilon_decay_rate']
    
    for episode in range(num_episodes):
        state, info = env.reset()
        terminated = False
        truncated = False
        episode_reward = 0
        episode_td_errors = []
        
        for step in range(max_steps_per_episode):
            action = choose_action(env, q_table, state, epsilon)
            new_state, reward, terminated, truncated, info = env.step(action)
            
            q_table, td_error = update_q_value(q_table, state, action, reward, new_state, terminated,
                                             learning_rate, discount_factor)
            
            state = new_state
            episode_reward += reward
            episode_td_errors.append(td_error)
            
            if terminated or truncated:
                break
        
        all_rewards.append(episode_reward)
        all_episode_lengths.append(step + 1)
        if episode_td_errors:
            all_td_errors.append(np.mean(episode_td_errors))
        else:
            all_td_errors.append(0)
            
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-epsilon_decay_rate * episode)
        
        if verbose and (episode + 1) % 20000 == 0:
            avg_reward = np.mean(all_rewards[-1000:])
            print(f"Ep {episode + 1}/{num_episodes}, Avg Reward: {avg_reward:.2f}")

    return q_table, all_rewards, all_episode_lengths, all_td_errors

def plot_training(all_rewards, all_lengths, all_errors, filename, rolling_length=500):
    print(f"Saving plot to {filename}...")
    fig, axs = plt.subplots(ncols=3, figsize=(15, 5))
    
    # Rewards
    reward_series = pd.Series(all_rewards)
    reward_avg = reward_series.rolling(window=rolling_length, min_periods=rolling_length).mean()
    axs[0].plot(reward_avg, label=f'{rolling_length}-ep rolling avg', color='blue')
    axs[0].set_title("Episode Rewards")
    axs[0].set_xlabel("Episode")
    axs[0].set_ylabel("Reward")
    axs[0].grid(True)
    
    # Lengths
    length_series = pd.Series(all_lengths)
    length_avg = length_series.rolling(window=rolling_length, min_periods=rolling_length).mean()
    axs[1].plot(length_avg, label=f'{rolling_length}-ep rolling avg', color='orange')
    axs[1].set_title("Episode Lengths")
    axs[1].set_xlabel("Episode")
    axs[1].grid(True)
    
    # Errors
    error_series = pd.Series(all_errors)
    error_avg = error_series.rolling(window=rolling_length, min_periods=rolling_length).mean()
    axs[2].plot(error_avg, label=f'{rolling_length}-ep rolling avg', color='green')
    axs[2].set_title("Training Error (TD Error)")
    axs[2].set_xlabel("Episode")
    axs[2].grid(True)
    
    plt.tight_layout() 
    plt.savefig(f"./results_Q/{filename}")
    plt.close()

def plot_comparison(baseline_rewards, unstable_rewards):
    print("Generating Stability Comparison Plot...")
    plt.figure(figsize=(10, 6))
    s1 = pd.Series(baseline_rewards)
    plt.plot(s1.rolling(window=1000).mean(), label="Baseline (alpha=0.1)", color='blue', linewidth=2)
    s2 = pd.Series(unstable_rewards)
    plt.plot(s2.rolling(window=50).mean(), label="High LR (alpha=0.99)", color='red', alpha=0.5, linewidth=1)
    plt.title("Stability Comparison: Baseline vs. High Learning Rate")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.savefig("./results_Q/comparison_stability.png")
    plt.close()


def run_sensitivity_analysis(env, base_hyperparams):
    print("\n" + "="*40)
    print("STARTING SENSITIVITY ANALYSIS (PLOTS)")
    print("="*40)
    
    state_size = env.observation_space.n
    action_size = env.action_space.n
    

    gammas = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.99]
    gamma_results = []
    
    print("Testing Discount Factors (Gamma)...")
    for g in gammas:
        # Create fresh q-table
        q_table = initialize_q_table(state_size, action_size)
        params = base_hyperparams.copy()
        params['discount_factor'] = g
        # Reduce episodes slightly for speed, e.g., 50k is usually enough to see convergence
        params['num_episodes'] = 50000 
        
        _, rewards, _, _ = train_agent(env, q_table, params, verbose=False)
        final_score = np.mean(rewards[-1000:]) # Average of last 1000
        gamma_results.append(final_score)
        print(f" -> Gamma={g}: Score={final_score:.2f}")

    plt.figure(figsize=(8, 5))
    plt.plot(gammas, gamma_results, marker='o', linestyle='-', color='purple', linewidth=2)
    plt.title("Sensitivity: Discount Factor vs. Average Reward")
    plt.xlabel("Discount Factor (Gamma)")
    plt.ylabel("Final Average Reward")
    plt.grid(True)
    plt.savefig("./results_Q/sensitivity_gamma.png")
    plt.close()

    alphas = [0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
    alpha_results = []
    
    print("Testing Learning Rates (Alpha)...")
    for a in alphas:
        q_table = initialize_q_table(state_size, action_size)
        params = base_hyperparams.copy()
        params['learning_rate'] = a
        params['num_episodes'] = 50000
        
        _, rewards, _, _ = train_agent(env, q_table, params, verbose=False)
        final_score = np.mean(rewards[-1000:])
        alpha_results.append(final_score)
        print(f" -> Alpha={a}: Score={final_score:.2f}")

    plt.figure(figsize=(8, 5))
    plt.plot(alphas, alpha_results, marker='s', linestyle='-', color='darkorange', linewidth=2)
    plt.title("Sensitivity: Learning Rate vs. Average Reward")
    plt.xlabel("Learning Rate (Alpha)")
    plt.ylabel("Final Average Reward")
    plt.grid(True)
    plt.savefig("./results_Q/sensitivity_alpha.png")
    plt.close()
    
    print("Sensitivity plots saved to ./results_Q/")

def create_grids_taxi(env, q_table, passenger_loc, dest_loc):
    value_grid_goto = np.zeros((5, 5))
    policy_grid_goto = np.zeros((5, 5), dtype=int)
    value_grid_dropoff = np.zeros((5, 5))
    policy_grid_dropoff = np.zeros((5, 5), dtype=int)

    for r in range(5):
        for c in range(5):
            state_goto = env.unwrapped.encode(r, c, passenger_loc, dest_loc)
            value_grid_goto[r, c] = np.max(q_table[state_goto, :])
            policy_grid_goto[r, c] = np.argmax(q_table[state_goto, :])
            state_dropoff = env.unwrapped.encode(r, c, 4, dest_loc)
            value_grid_dropoff[r, c] = np.max(q_table[state_dropoff, :])
            policy_grid_dropoff[r, c] = np.argmax(q_table[state_dropoff, :])
            
    return (value_grid_goto, policy_grid_goto), (value_grid_dropoff, policy_grid_dropoff)

def create_plots_taxi(value_grid, policy_grid, title: str):
    action_labels = ["South", "North", "East", "West", "Pickup", "Dropoff"]
    cmap = ListedColormap(["lightgreen", "magenta", "grey"])
    
    legend_patches = [
        mpatches.Patch(color="lightgreen", label="Move"),
        mpatches.Patch(color="magenta", label="Pickup"),
        mpatches.Patch(color="grey", label="Dropoff")
    ]

    policy_grid_types = np.zeros((5, 5), dtype=int)
    for r in range(5):
        for c in range(5):
            action = policy_grid[r, c]
            if 0 <= action <= 3: policy_grid_types[r, c] = 0
            elif action == 4: policy_grid_types[r, c] = 1
            elif action == 5: policy_grid_types[r, c] = 2
            
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = BoundaryNorm(bounds, cmap.N)

    fig, axs = plt.subplots(ncols=2, figsize=(13, 6))
    fig.suptitle(title, fontsize=16)

    ax1 = axs[0]
    ax1.set_title("V(s) - State Values")
    sns.heatmap(value_grid, annot=True, fmt=".2f", cmap="viridis", ax=ax1, cbar=True)
    ax1.set_xlabel("Column")
    ax1.set_ylabel("Row")

    ax2 = axs[1]
    ax2.set_title("Policy (Best Action)")
    sns.heatmap(policy_grid_types, annot=False, cmap=cmap, norm=norm, fmt="d", ax=ax2, cbar=False)

    for r in range(5):
        for c in range(5):
            ax2.text(c + 0.5, r + 0.5, action_labels[policy_grid[r, c]], 
                        ha='center', va='center', color='black', fontsize=9)
            
    ax2.set_xlabel("Column")
    ax2.set_ylabel("Row")
    ax2.legend(handles=legend_patches, bbox_to_anchor=(1.5, 1))
    
    plt.tight_layout()
    return fig

def main():
    env = gym.make("Taxi-v3", render_mode=None)
    state_size = env.observation_space.n
    action_size = env.action_space.n

    print("\n>>> Running Experiment 1: Baseline")
    q_table_base = initialize_q_table(state_size, action_size)
    hyperparams_base = {
        'num_episodes': 100000, 'max_steps_per_episode': 100,
        'learning_rate': 0.1, 'discount_factor': 0.99,
        'epsilon': 1.0, 'max_epsilon': 1.0, 'min_epsilon': 0.01, 'epsilon_decay_rate': 0.001
    }
    q_table_base, rewards_base, lengths_base, errors_base = train_agent(env, q_table_base, hyperparams_base)
    plot_training(rewards_base, lengths_base, errors_base, "q_learning_all_plots_baseline.png")

    print("\n>>> Running Experiment 2: Low Discount Factor (gamma=0.1)")
    q_table_myopic = initialize_q_table(state_size, action_size)
    hyperparams_myopic = hyperparams_base.copy()
    hyperparams_myopic['discount_factor'] = 0.1 
    q_table_myopic, rewards_myopic, lengths_myopic, errors_myopic = train_agent(env, q_table_myopic, hyperparams_myopic)
    plot_training(rewards_myopic, lengths_myopic, errors_myopic, "q_learning_all_plots_df01.png")

    print("\n>>> Running Experiment 3: High Learning Rate (alpha=0.99)")
    q_table_unstable = initialize_q_table(state_size, action_size)
    hyperparams_unstable = hyperparams_base.copy()
    hyperparams_unstable['learning_rate'] = 0.99 
    q_table_unstable, rewards_unstable, lengths_unstable, errors_unstable = train_agent(env, q_table_unstable, hyperparams_unstable)
    plot_training(rewards_unstable, lengths_unstable, errors_unstable, "q_learning_all_plots_lr99.png")

    plot_comparison(rewards_base, rewards_unstable)

    run_sensitivity_analysis(env, hyperparams_base)

    print("\n>>> Generating Policy Maps (Baseline Agent)")
    temp_env = gym.make("Taxi-v3")
    
    (vg_R, pg_R), (vg_B, pg_B) = create_grids_taxi(temp_env, q_table_base, 0, 2)
    fig1 = create_plots_taxi(vg_R, pg_R, "Phase 1: Policy to pick up at R(0,0)")
    fig2 = create_plots_taxi(vg_B, pg_B, "Phase 2: Policy to drop off at B(4,0)")
    
    (vg_Y, pg_Y), (vg_R_dest, pg_R_dest) = create_grids_taxi(temp_env, q_table_base, 3, 0)
    fig3 = create_plots_taxi(vg_Y, pg_Y, "Phase 1: Policy to pick up at Y(4,3)")
    fig4 = create_plots_taxi(vg_R_dest, pg_R_dest, "Phase 2: Policy to drop off at R(0,0)")
    
    fig1.savefig("./results_Q/policy_1a_R_goto.png")
    fig2.savefig("./results_Q/policy_1b_B_dropoff.png")
    fig3.savefig("./results_Q/policy_2a_Y_goto.png")
    fig4.savefig("./results_Q/policy_2b_R_dropoff.png")
    
    temp_env.close()
    env.close()

    print("\n" + "="*30)
    print("FINAL RESULTS SUMMARY")
    print("="*30)
    print(f"Baseline Final Avg Reward: {np.mean(rewards_base[-1000:]):.2f}")
    print(f"Gamma=0.1 Avg:    {np.mean(rewards_myopic[-1000:]):.2f}")
    print(f"Alpha=0.99 Avg: {np.mean(rewards_unstable[-1000:]):.2f}")
    print("="*30)
    

if __name__ == "__main__":
    main()