import numpy as np
import pickle as pkl

# -----------------------------
# Load learned Q-tables
# -----------------------------
Q_qlearning = pkl.load(open("frozenlake_q_table.pkl", "rb"))
Q_sarsa = pkl.load(open("frozenlake_sarsa_q_table.pkl", "rb"))

GRID_SIZE = 8

# -----------------------------
# Extract greedy policy
# π(s) = argmax_a Q(s, a)
# -----------------------------
def extract_policy(Q):
    return np.argmax(Q, axis=1)

# -----------------------------
# Visualize policy on 8x8 grid
# -----------------------------
def visualize_policy(policy, title):
    arrows = {
        0: "←",  # LEFT
        1: "↓",  # DOWN
        2: "→",  # RIGHT
        3: "↑"   # UP
    }

    print(f"\n{title}")
    grid = policy.reshape(GRID_SIZE, GRID_SIZE)

    for row in grid:
        print(" ".join(arrows[a] for a in row))

# -----------------------------
# Extract policies
# -----------------------------
policy_qlearning = extract_policy(Q_qlearning)
policy_sarsa = extract_policy(Q_sarsa)

# -----------------------------
# Visualize
# -----------------------------
visualize_policy(policy_qlearning, "Q-Learning Greedy Policy (8x8)")
visualize_policy(policy_sarsa, "SARSA Greedy Policy (8x8)")
