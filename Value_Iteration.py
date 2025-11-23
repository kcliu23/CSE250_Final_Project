import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
import pandas as pd

def value_iteration(env, discount_factor=0.99, theta=1e-9, max_iterations=1000):
    """
    Runs Value Iteration to find the optimal Value Function V* and Policy.
    Now returns 'iterations' count for comparison.
    """
    state_size = env.observation_space.n
    action_size = env.action_space.n
    
    value_table = np.zeros(state_size)
    deltas = [] 
    
    iteration = 0
    
    # --- The Training Loop ---
    while iteration < max_iterations:
        delta = 0
        
        # Sweep through all 500 states
        for s in range(state_size):
            v_old = value_table[s]
            
            # Check every action's possible outcome
            action_values = []
            for a in range(action_size):
                q_sa = 0
                # Sum over all possible next states (prob, next_state, reward, done)
                for prob, next_state, reward, terminated in env.unwrapped.P[s][a]:
                    q_sa += prob * (reward + discount_factor * value_table[next_state])
                action_values.append(q_sa)
            
            # Update V(s) to the max value found
            value_table[s] = np.max(action_values)
            delta = max(delta, abs(v_old - value_table[s]))
            
        deltas.append(delta)
        iteration += 1
        
        if delta < theta:
            break
            
    # --- Policy Extraction ---
    policy = np.zeros(state_size, dtype=int)
    for s in range(state_size):
        action_values = []
        for a in range(action_size):
            q_sa = 0
            for prob, next_state, reward, terminated in env.unwrapped.P[s][a]:
                q_sa += prob * (reward + discount_factor * value_table[next_state])
            action_values.append(q_sa)
        policy[s] = np.argmax(action_values)
        
    return value_table, policy, deltas, iteration

def evaluate_policy(env, policy, episodes=100):
    """
    Quantitatively measures how good the policy is.
    Runs 100 episodes without exploration and calculates average reward.
    """
    total_rewards = []
    total_steps = []
    
    for _ in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        steps = 0
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            action = policy[state]
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            steps += 1
            if steps > 100: break # Safety break
            
        total_rewards.append(episode_reward)
        total_steps.append(steps)
        
    avg_reward = np.mean(total_rewards)
    avg_steps = np.mean(total_steps)
    return avg_reward, avg_steps

def run_ablation_study(env):
    """
    Compares different Hyperparameters (Gamma).
    """
    print("\n--- Running Ablation Study (Hyperparameter Comparison) ---")
    
    # We will test how Discount Factor affects convergence speed
    gammas = [0.1, 0.5, 0.8, 0.9, 0.99]
    results_iters = []
    results_rewards = []
    
    for gamma in gammas:
        print(f"Testing Gamma (Discount Factor): {gamma}")
        _, policy, _, iterations = value_iteration(env, discount_factor=gamma)
        
        # Evaluate how good this policy is
        avg_reward, _ = evaluate_policy(env, policy, episodes=50)
        
        results_iters.append(iterations)
        results_rewards.append(avg_reward)
    
    # --- Plot 1: Gamma vs. Convergence Speed ---
    plt.figure(figsize=(10, 5))
    plt.plot(gammas, results_iters, marker='o', color='purple')
    plt.title("Impact of Discount Factor (Gamma) on Convergence Speed")
    plt.xlabel("Discount Factor (Gamma)")
    plt.ylabel("Iterations to Converge")
    plt.grid(True)
    plt.savefig("./results_V/ablation_gamma_vs_iterations.png")
    print("Saved ablation plot: ablation_gamma_vs_iterations.png")
    
    # --- Plot 2: Gamma vs. Final Score ---
    plt.figure(figsize=(10, 5))
    plt.plot(gammas, results_rewards, marker='s', color='green')
    plt.title("Impact of Discount Factor (Gamma) on Policy Quality")
    plt.xlabel("Discount Factor (Gamma)")
    plt.ylabel("Average Reward (50 episodes)")
    plt.grid(True)
    plt.savefig("./results_V/ablation_gamma_vs_reward.png")
    print("Saved ablation plot: ablation_gamma_vs_reward.png")

def visualize_vi_policy_maps(env, value_table, policy, scenario_name, passenger_loc, dest_loc):
    """
    Creates specific heatmaps for Value Iteration results.
    """
    # Setup grids
    value_grid_goto = np.zeros((5, 5))
    policy_grid_goto = np.zeros((5, 5), dtype=int)
    value_grid_dropoff = np.zeros((5, 5))
    policy_grid_dropoff = np.zeros((5, 5), dtype=int)

    for r in range(5):
        for c in range(5):
            # Phase 1: Go to Passenger
            state_goto = env.unwrapped.encode(r, c, passenger_loc, dest_loc)
            value_grid_goto[r, c] = value_table[state_goto]
            policy_grid_goto[r, c] = policy[state_goto]
            
            # Phase 2: Go to Destination (Pass inside taxi)
            state_dropoff = env.unwrapped.encode(r, c, 4, dest_loc)
            value_grid_dropoff[r, c] = value_table[state_dropoff]
            policy_grid_dropoff[r, c] = policy[state_dropoff]

    # --- Plotting ---
    fig, axs = plt.subplots(ncols=2, figsize=(13, 6))
    fig.suptitle(f"VI Optimal Policy: {scenario_name}", fontsize=16)
    
    action_labels = ["South", "North", "East", "West", "Pickup", "Dropoff"]
    cmap = ListedColormap(["lightgreen", "magenta", "grey"])
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = BoundaryNorm(bounds, cmap.N)
    
    legend_patches = [
        mpatches.Patch(color="lightgreen", label="Move"),
        mpatches.Patch(color="magenta", label="Pickup"),
        mpatches.Patch(color="grey", label="Dropoff")
    ]

    # Map 1: Phase 1 Policy
    axs[0].set_title("Phase 1: Go to Passenger")
    types_goto = np.zeros((5, 5), dtype=int)
    for r in range(5):
        for c in range(5):
            a = policy_grid_goto[r, c]
            types_goto[r, c] = 0 if a <= 3 else (1 if a == 4 else 2)
            
    sns.heatmap(types_goto, annot=False, cmap=cmap, norm=norm, cbar=False, ax=axs[0])
    for r in range(5):
        for c in range(5):
            axs[0].text(c + 0.5, r + 0.5, action_labels[policy_grid_goto[r,c]], 
                       ha='center', va='center', color='black', fontsize=9)
    axs[0].set_xlabel("Col")
    axs[0].set_ylabel("Row")

    # Map 2: Phase 2 Policy
    axs[1].set_title("Phase 2: Go to Destination")
    types_drop = np.zeros((5, 5), dtype=int)
    for r in range(5):
        for c in range(5):
            a = policy_grid_dropoff[r, c]
            types_drop[r, c] = 0 if a <= 3 else (1 if a == 4 else 2)

    sns.heatmap(types_drop, annot=False, cmap=cmap, norm=norm, cbar=False, ax=axs[1])
    for r in range(5):
        for c in range(5):
            axs[1].text(c + 0.5, r + 0.5, action_labels[policy_grid_dropoff[r,c]], 
                       ha='center', va='center', color='black', fontsize=9)
    axs[1].set_xlabel("Col")
    axs[1].set_ylabel("Row")
    
    fig.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(1.15, 0.95))
    plt.tight_layout()
    return fig

def watch_vi_agent(env, policy, start_row, start_col, passenger_loc, dest_loc):
    """Visualizes the Optimal Policy for a specific scenario."""
    env.reset() 
    state = env.unwrapped.encode(start_row, start_col, passenger_loc, dest_loc)
    env.unwrapped.s = state
    
    print(f"\nWatching VI Agent: ({start_row},{start_col}) -> Pass({passenger_loc}) -> Dest({dest_loc})")
    
    time.sleep(1)
    
    terminated = False
    truncated = False
    steps = 0
    total_reward = 0
    
    while not (terminated or truncated):
        action = policy[state]
        state, reward, terminated, truncated, info = env.step(action)
        steps += 1
        total_reward += reward
        time.sleep(0.5)
        
    print(f"Finished in {steps} steps. Total Reward: {total_reward}")

def plot_convergence(deltas):
    plt.figure(figsize=(10, 6))
    plt.plot(deltas)
    plt.yscale('log')
    plt.title("Value Iteration Convergence (Delta per Iteration)")
    plt.xlabel("Iteration")
    plt.ylabel("Max Value Change (Delta) - Log Scale")
    plt.grid(True, which="both", ls="-")
    plt.savefig("./results_V/vi_convergence_plot.png")
    print("Saved convergence plot: vi_convergence_plot.png")

def main():
    # 1. Setup
    env = gym.make("Taxi-v3", render_mode=None)
    
    # --- PART A: Main Training ---
    print("--- Part A: Running Main Value Iteration ---")
    value_table, policy, deltas, iterations = value_iteration(env, discount_factor=0.99)
    print(f"Converged in {iterations} iterations.")
    
    # --- PART B: Quantitative Evaluation (New!) ---
    print("\n--- Part B: Quantitative Evaluation ---")
    avg_reward, avg_steps = evaluate_policy(env, policy, episodes=100)
    print(f"FINAL SCORE: Average Reward over 100 episodes: {avg_reward:.2f}")
    print(f"FINAL SCORE: Average Steps per episode: {avg_steps:.2f}")
    
    # --- PART C: Plotting ---
    plot_convergence(deltas)
    
    temp_env = gym.make("Taxi-v3")
    fig1 = visualize_vi_policy_maps(temp_env, value_table, policy, "R to B", 0, 2)
    fig1.savefig("./results_V/vi_policy_R_to_B.png")
    
    fig2 = visualize_vi_policy_maps(temp_env, value_table, policy, "Y to R", 3, 0)
    fig2.savefig("./results_V/vi_policy_Y_to_R.png")
    
    # --- PART D: Ablation Study (New!) ---
    run_ablation_study(env)
    
    print("\nAll experiments complete. Check './results_V/' folder.")
    
    # --- PART E: Watch It Run ---
    # Uncomment below if you want to see the animation
    # env_human = gym.make("Taxi-v3", render_mode="human")
    # watch_vi_agent(env_human, policy, 0, 1, 0, 2) 
    # env_human.close()

if __name__ == "__main__":
    main()