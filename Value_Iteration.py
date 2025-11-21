import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm

def value_iteration(env, discount_factor=0.99, theta=1e-9):
    """
    Runs the Value Iteration algorithm to find the optimal Value Function V*.
    """
    state_size = env.observation_space.n
    action_size = env.action_space.n
    
    # Initialize V(s) to zeros
    value_table = np.zeros(state_size)
    deltas = [] # To track convergence speed
    
    iteration = 0
    print("Starting Value Iteration...")
    
    while True:
        delta = 0
        
        # Sweep through all states
        for s in range(state_size):
            v_old = value_table[s]
            
            # Calculate the value of all actions from this state
            action_values = []
            for a in range(action_size):
                q_sa = 0
                
                ### --- FIX 1: Use env.unwrapped.P --- ###
                for prob, next_state, reward, terminated in env.unwrapped.P[s][a]:
                    q_sa += prob * (reward + discount_factor * value_table[next_state])
                
                action_values.append(q_sa)
            
            # Update value table with the best action's value
            value_table[s] = np.max(action_values)
            
            # Track largest change
            delta = max(delta, abs(v_old - value_table[s]))
            
        deltas.append(delta)
        iteration += 1
        
        if delta < theta:
            print(f"Value Iteration converged after {iteration} iterations.")
            break
            
    # --- Extract Optimal Policy ---
    print("Extracting optimal policy...")
    policy = np.zeros(state_size, dtype=int)
    
    for s in range(state_size):
        action_values = []
        for a in range(action_size):
            q_sa = 0
            
            ### --- FIX 2: Use env.unwrapped.P here too --- ###
            for prob, next_state, reward, terminated in env.unwrapped.P[s][a]:
                q_sa += prob * (reward + discount_factor * value_table[next_state])
            
            action_values.append(q_sa)
        
        # The best action is the one that maximizes the expected value
        policy[s] = np.argmax(action_values)
        
    return value_table, policy, deltas
def plot_convergence(deltas):
    """Plots the 'Delta' (max change) over iterations."""
    plt.figure(figsize=(10, 6))
    plt.plot(deltas)
    plt.yscale('log') # Log scale makes it easier to see small deltas
    plt.title("Value Iteration Convergence (Delta per Iteration)")
    plt.xlabel("Iteration")
    plt.ylabel("Max Value Change (Delta) - Log Scale")
    plt.grid(True, which="both", ls="-")
    plt.savefig("./results_V/vi_convergence_plot.png")
    plt.show()

def visualize_vi_policy_maps(env, value_table, policy, scenario_name, passenger_loc, dest_loc):
    """
    Creates specific heatmaps for Value Iteration results.
    Identical to the Q-Learning visualization for comparison.
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
    
    # 1. Initialize the environment (Satisfies the ResetNeeded error)
    env.reset() 
    
    # 2. Manually set our specific scenario
    state = env.unwrapped.encode(start_row, start_col, passenger_loc, dest_loc)
    env.unwrapped.s = state
    
    print(f"\nWatching VI Agent: ({start_row},{start_col}) -> Pass({passenger_loc}) -> Dest({dest_loc})")
    
    # 3. Now we can render safely
    env.render()
    time.sleep(1)
    
    terminated = False
    truncated = False
    steps = 0
    
    while not (terminated or truncated):
        action = policy[state]
        state, reward, terminated, truncated, info = env.step(action)
        steps += 1
        time.sleep(0.5)
        
    print(f"Finished in {steps} steps.")
def main():
    # 1. Setup Environment
    env = gym.make("Taxi-v3", render_mode=None)
    
    # 2. Run Value Iteration
    value_table, policy, deltas = value_iteration(env)
    
    # 3. Plot Convergence
    plot_convergence(deltas)
    
    # 4. Generate Comparison Heatmaps
    print("Generating Policy Maps...")
    temp_env = gym.make("Taxi-v3")
    
    # Scenario 1: R(0,0) to B(4,0)
    fig1 = visualize_vi_policy_maps(temp_env, value_table, policy, "R to B", 0, 2)
    fig1.savefig("./results_V/vi_policy_R_to_B.png")
    
    # Scenario 2: Y(4,3) to R(0,0)
    fig2 = visualize_vi_policy_maps(temp_env, value_table, policy, "Y to R", 3, 0)
    fig2.savefig("./results_V/vi_policy_Y_to_R.png")
    
    plt.show()
    
    # 5. Watch the Agent
    env_human = gym.make("Taxi-v3", render_mode="human")
    watch_vi_agent(env_human, policy, 0, 1, 0, 2) # R to B example
    watch_vi_agent(env_human, policy, 2, 2, 3, 0) # Y to R example
    env_human.close()

if __name__ == "__main__":
    main()