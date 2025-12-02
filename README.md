# Reinforcement Learning in Taxi-v3 (Gymnasium)

This repository contains the final project implementation for **CSE 250A: Principles of Artificial Intelligence: Probabilistic Reasoning and Learning**. 

The project investigates the derivation of optimal policies for the `Taxi-v3` environment using two contrasting Reinforcement Learning (RL) approaches:
1.  **Model-Based Planning:** Value Iteration (VI)
2.  **Model-Free Learning:** Q-Learning

## 🚖 Environment: Taxi-v3
The Taxi problem involves navigating a 5x5 grid world to pick up a passenger from one of four locations (R, G, Y, B) and drop them off at a destination.
* **State Space:** 500 discrete states.
* **Action Space:** 6 discrete actions (South, North, East, West, Pickup, Dropoff).
* **Rewards:** +20 for successful dropoff, -1 per step, -10 for illegal actions.

## 📁 Repository Structure

    ├── Value_Iteration.py      # Implementation of Model-Based Value Iteration
    ├── q_learning.py           # Implementation of Model-Free Q-Learning
    ├── results_Q/                # Stores generated plots and policy heatmaps
    │   ├── q_learning_all_plots_baseline.png
    │   ├── comparison_stability.png
    │   └── ... (other plots)
    ├── results_V/        # Stores VI-specific plots
    │   ├── vi_convergence_plot.png
    │   └── ...
    ├── Docs/
    |   ├── milestone1.pdf  
    │   └── milestone1.docx
    └── README.md

##  How to Run the Code

### Prerequisites
You need Python 3.x and the following libraries installed:

    pip install gymnasium numpy matplotlib seaborn pandas

### 1. Run Value Iteration (Model-Based)
This script calculates the optimal policy using the known transition matrix $P(s'|s,a)$. It converges in < 2 seconds.

    python Value_Iteration.py

* **Output:** Generates convergence plots and optimal policy heatmaps in the `results_V/` folder.

### 2. Run Q-Learning (Model-Free)
This script trains an agent without knowledge of the transition dynamics. It runs multiple experiments (Baseline, Myopic, Aggressive) and generates sensitivity analysis plots.

    python Q_learning.py

* **Output:** Generates training curves, stability comparison plots, and sensitivity analysis figures in the `results_Q/` folder.

### 3. Run Sensitivity Analysis for VI
(Optional) To regenerate the specific ablation study plots for Value Iteration:

    python run_ablation.py

## Key Results

### Value Iteration
* **Convergence:** Converged to optimal $V^*$ in **1,183 iterations** ($\approx$ 0.5 seconds).
* **Optimal Policy:** Successfully navigates shortest paths.

### Q-Learning
* **Baseline ($\alpha=0.1, \gamma=0.99$):** Converged to optimal average reward ($\approx$ 7.4) after ~40,000 episodes.
* **Robustness:** The agent proved surprisingly robust to high learning rates ($\alpha=0.99$), showing that the Taxi-v3 environment is sufficiently deterministic that aggressive updates do not destabilize the policy.
* **Sensitivity:** The agent requires a discount factor $\gamma \ge 0.3$ to learn the optimal path; values below this threshold result in myopic behavior.

## Authors
* Tzu Ping Chen
* Kai Cheng Liu
* Cheng-Yang Wu
* Chih Yun Lin

**University of California, San Diego**