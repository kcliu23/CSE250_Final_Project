import gymnasium as gym
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import pandas as pd

def initialize_q_table(state_size, action_size):
    
    return np.zeros((state_size, action_size))

def choose_action(env, q_table, state, epsilon):

    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample() 
    else:
        return np.argmax(q_table[state, :]) 

def update_q_value(q_table, state, action, reward, new_state, terminated,
                   learning_rate, discount_factor):
    

    #New Value = Old Value + learning_rate * (New Guess - Old Value)

    if terminated:
        target = reward # immediate reward 
    else:
        target = reward + discount_factor * np.max(q_table[new_state, :]) #future reward

    # temporal-difference error measures how "wrong" or "surprised" the agent was.
    # td_error = target - old_Q_value
    # If td_error is large:The agent's expectation was far from reality.
    # If td_error is small:  The agent already had a good estimate.
    td_error = target - q_table[state, action]
    
    q_table[state, action] = q_table[state, action] + learning_rate * td_error
    
    return q_table, td_error

def train_agent(env, q_table, hyperparameters):
    
    print("Training Q-Learning agent...")
    
    
    all_rewards = []
    all_episode_lengths = []
    all_td_errors = []
    
    # Unpack hyperparameters
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
        all_td_errors.append(np.mean(episode_td_errors)) 
            
        
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-epsilon_decay_rate * episode)
        
        if (episode + 1) % 1000 == 0:
            avg_reward = np.mean(all_rewards[-1000:])
            avg_length = np.mean(all_episode_lengths[-1000:])
            print(f"Ep {episode + 1}, Avg Reward: {avg_reward:.2f}, Avg Length: {avg_length:.2f}")

    print("Training finished!")
    return q_table, all_rewards, all_episode_lengths, all_td_errors

def plot_training(all_rewards, all_lengths, all_errors, rolling_length=500):
 
    print("Plotting training results...")
    
   
    fig, axs = plt.subplots(ncols=3, figsize=(15, 5))
    
   
    axs[0].set_title("Episode Rewards")
    reward_series = pd.Series(all_rewards)
    reward_avg = reward_series.rolling(window=rolling_length, min_periods=rolling_length).mean()
    axs[0].plot(reward_avg, label=f'{rolling_length}-ep rolling avg')
    axs[0].set_xlabel("Episode")
    axs[0].set_ylabel("Average Reward")
    
    
    axs[1].set_title("Episode Lengths")
    length_series = pd.Series(all_lengths)
    length_avg = length_series.rolling(window=rolling_length, min_periods=rolling_length).mean()
    axs[1].plot(length_avg, label=f'{rolling_length}-ep rolling avg')
    axs[1].set_xlabel("Episode")
    axs[1].set_ylabel("Average Length")
    
    
    axs[2].set_title("Training Error (TD Error)")
    error_series = pd.Series(all_errors)
    error_avg = error_series.rolling(window=rolling_length, min_periods=rolling_length).mean()
    axs[2].plot(error_avg, label=f'{rolling_length}-ep rolling avg')
    axs[2].set_xlabel("Episode")
    axs[2].set_ylabel("Average TD Error")
    
    plt.tight_layout() 
    plt.savefig("./results/q_learning_all_plots_df01.png")
    # plt.show()

def watch_agent(q_table):
    
    print("\nWatching trained agent...")
    env_human = gym.make("Taxi-v3", render_mode=None)
    state, info = env_human.reset()
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = np.argmax(q_table[state, :]) 
        state, reward, terminated, truncated, info = env_human.step(action)
        time.sleep(0.5)

    env_human.close()


def watch_agent_in_scenario(q_table, start_row, start_col, passenger_loc, dest_loc):

    env_human = gym.make("Taxi-v3", render_mode=None)
    
    # We must call reset() once to initialize the environment and renderer
    # The state it returns is random, so we will ignore it.
    state, info = env_human.reset()
    
    # --- Encode and Set Our Desired State ---
    try:
        # Get the integer representing our desired scenario
        desired_state = env_human.unwrapped.encode(start_row, start_col, passenger_loc, dest_loc)
    except Exception as e:
        print(f"Error encoding state: {e}")
        print("Please check your row/col/pass/dest values.")
        env_human.close()
        return
        
    # Manually set the environment's current state to our desired one
    env_human.unwrapped.s = desired_state
    state = desired_state # This is the state our loop will start with
    
    # --- Run the Simulation ---
    # The render window will show the random reset state until the first
    # step. We can add a pause to make this clear.
    scenario_name = f"Taxi at ({start_row},{start_col}), Pass at {passenger_loc}, Dest at {dest_loc}"
    print(f"\nWatching agent in scenario: {scenario_name}")
    print("Press Enter to start the simulation...")
    time.sleep(2)
    
    terminated = False
    truncated = False
    total_reward = 0
    
    while not (terminated or truncated):
        # Exploit: Always take the best action
        action = np.argmax(q_table[state, :])
        state, reward, terminated, truncated, info = env_human.step(action)
        total_reward += reward
        time.sleep(0.5)

    print(f"Scenario finished with a total reward of: {total_reward}")
    env_human.close()


def create_grids_taxi(env, q_table, passenger_loc, dest_loc):
    """
    Creates value and policy grids for both phases of a Taxi scenario.
    
    Phase 1: "Go to Passenger" (Passenger is at passenger_loc)
    Phase 2: "Go to Destination" (Passenger is in the taxi)
    """
    
    # --- Create 5x5 grids for both phases ---
    value_grid_goto = np.zeros((5, 5))
    policy_grid_goto = np.zeros((5, 5), dtype=int)
    
    value_grid_dropoff = np.zeros((5, 5))
    policy_grid_dropoff = np.zeros((5, 5), dtype=int)

    for r in range(5):
        for c in range(5):
            # --- Phase 1: Go to Passenger ---
            # State: (r, c, passenger_loc, dest_loc)
            state_goto = env.unwrapped.encode(r, c, passenger_loc, dest_loc)
            value_grid_goto[r, c] = np.max(q_table[state_goto, :])
            policy_grid_goto[r, c] = np.argmax(q_table[state_goto, :])
            
            # --- Phase 2: Go to Destination ---
            # State: (r, c, 4, dest_loc) (4 = "in taxi")
            state_dropoff = env.unwrapped.encode(r, c, 4, dest_loc)
            value_grid_dropoff[r, c] = np.max(q_table[state_dropoff, :])
            policy_grid_dropoff[r, c] = np.argmax(q_table[state_dropoff, :])
            
    return (value_grid_goto, policy_grid_goto), (value_grid_dropoff, policy_grid_dropoff)


def create_plots_taxi(value_grid, policy_grid, title: str):
    """
    Creates a plot with two 2D heatmaps:
    1. V(s) - State Values
    2. Policy (Best Action)
    """
    
    # --- Setup ---
    action_labels = ["South", "North", "East", "West", "Pickup", "Dropoff"]

    ### --- THIS IS THE FIX --- ###
    
    # 1. Create our custom colormap
    # 0=Move (lightgreen), 1=Pickup (magenta), 2=Dropoff (grey)
    cmap = ListedColormap(["lightgreen", "magenta", "grey"])
    
    # 2. Create the legend handles to match
    legend_patches = [
        mpatches.Patch(color="lightgreen", label="Move (0-3)"),
        mpatches.Patch(color="magenta", label="Pickup (4)"),
        mpatches.Patch(color="grey", label="Dropoff (5)")
    ]

    # 3. Map the action indices (0-5) to types (0-2) for coloring
    policy_grid_types = np.zeros((5, 5), dtype=int)
    for r in range(5):
        for c in range(5):
            action = policy_grid[r, c]
            if 0 <= action <= 3: policy_grid_types[r, c] = 0   # Type 0 = Move
            elif action == 4: policy_grid_types[r, c] = 1   # Type 1 = Pickup
            elif action == 5: policy_grid_types[r, c] = 2   # Type 2 = Dropoff

    # --- Create Figure ---
    fig, axs = plt.subplots(ncols=2, figsize=(13, 6))
    fig.suptitle(title, fontsize=16)

    # --- 1. Plot the V(s) - State Values Heatmap ---
    ax1 = axs[0]
    ax1.set_title("V(s) - State Values")
    sns.heatmap(
        value_grid,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        ax=ax1,
        cbar=True
    )
    ax1.set_xlabel("Column")
    ax1.set_ylabel("Row")

    # --- 2. Plot the 2D Policy Heatmap ---
    ax2 = axs[1]
    ax2.set_title("Policy (Best Action)")
    
    sns.heatmap(
        policy_grid_types, # Use the 0-2 types for color
        annot=False,
        cmap=cmap,
        fmt="d",
        ax=ax2,
        cbar=False,
        vmin=0,
        vmax=2,
    )
    
    # Add text labels from the 0-5 action grid
    for r in range(5):
        for c in range(5):
            ax2.text(c + 0.5, r + 0.5, action_labels[policy_grid[r, c]], 
                        ha='center', va='center', color='black', fontsize=9)
            
    ax2.set_xlabel("Column")
    ax2.set_ylabel("Row")
    ax2.legend(handles=legend_patches, bbox_to_anchor=(1.5, 1))
    
    plt.tight_layout() # Prevent labels from overlapping
    
    return fig

def main():
    # --- 1. Setup ---
    env = gym.make("Taxi-v3", render_mode=None)
    state_size = env.observation_space.n
    action_size = env.action_space.n
    
    # --- 2. Initialize Q-Table ---
    q_table = initialize_q_table(state_size, action_size)
    
    # --- 3. Set Hyperparameters ---
    hyperparameters = {
        'num_episodes': 100000,         
        'max_steps_per_episode': 100,      
        'learning_rate': 0.1,           
        'discount_factor': 0.1, # How much to value future rewards (0.0=only care about now, 1.0=care about all future).
        'epsilon': 1.0,          # The starting probability of taking a random action (100% random).
        'max_epsilon': 1.0,      # The maximum value epsilon can be (the starting value).
        'min_epsilon': 0.01,     # The minimum value epsilon will decay to (1% random) to ensure a little exploration.
        'epsilon_decay_rate': 0.001 # How fast epsilon drops from max_epsilon to min_epsilon .
    }
    
    q_table, rewards, lengths, errors = train_agent(env, q_table, hyperparameters)
    env.close()
    
    
    plot_training(rewards, lengths, errors)
    

    print("Generating policy plots...")
    temp_env = gym.make("Taxi-v3")
    

    (vg_goto_R, pg_goto_R), (vg_drop_B, pg_drop_B) = create_grids_taxi(temp_env, q_table, 
                                                                    passenger_loc=0, dest_loc=2)
    
    fig1 = create_plots_taxi(vg_goto_R, pg_goto_R, "Phase 1: Policy to pick up at R(0,0)")
    fig2 = create_plots_taxi(vg_drop_B, pg_drop_B, "Phase 2: Policy to drop off at B(4,0)")
    

    (vg_goto_Y, pg_goto_Y), (vg_drop_R, pg_drop_R) = create_grids_taxi(temp_env, q_table, 
                                                                    passenger_loc=3, dest_loc=0)
    
    fig3 = create_plots_taxi(vg_goto_Y, pg_goto_Y, "Phase 1: Policy to pick up at Y(4,3)")
    fig4 = create_plots_taxi(vg_drop_R, pg_drop_R, "Phase 2: Policy to drop off at R(0,0)")
    
     # Save all figures


    fig1.savefig("./results/policy_1a_R_goto.png")
    fig2.savefig("./results/policy_1b_B_dropoff.png")
    fig3.savefig("./results/policy_2a_Y_goto.png")
    fig4.savefig("./results/policy_2b_R_dropoff.png")
    # plt.show() # Show all figures
    
    temp_env.close()

    # --- 7. Watch Trained Agent in Specific Scenarios ---
    # The four pickup/dropoff spots (R, G, B, Y) and the "in-taxi" status
    # 0 = R (at row 0, col 0)
    # 1 = G (at row 0, col 4)
    # 2 = B (at row 4, col 0)
    # 3 = Y (at row 4, col 3)
    # 4 = In Taxi 

    watch_agent_in_scenario(q_table, 
                            start_row=0, start_col=1, 
                            passenger_loc=0, dest_loc=2) # R to B

    watch_agent_in_scenario(q_table, 
                            start_row=2, start_col=2, 
                            passenger_loc=3, dest_loc=0) # Y to R

    # --- 8. Watch Trained Agent ---
    watch_agent(q_table)


if __name__ == "__main__":
    main()