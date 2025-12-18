import gymnasium as gym
import numpy as np
import pickle as pkl
import time

# -----------------------------
# Load environment
# -----------------------------
env = gym.make(
    "FrozenLake-v1",
    map_name="8x8",
    is_slippery=True,
    render_mode="human"   
)

# -----------------------------
# Load Q-tables
# -----------------------------
q_learning_q = pkl.load(open("frozenlake_q_table.pkl", "rb"))
sarsa_q = pkl.load(open("frozenlake_sarsa_q_table.pkl", "rb"))

# -----------------------------
# Greedy policy
# -----------------------------
def greedy_policy(Q, state):
    max_q = np.max(Q[state])
    best_actions = np.where(Q[state] == max_q)[0]
    return np.random.choice(best_actions)

# -----------------------------
# Evaluation function
# -----------------------------
def evaluate_agent(Q, num_episodes=1000, visualize=False):
    successes = 0

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            if visualize:
                time.sleep(0.3)

            action = greedy_policy(Q, state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            total_reward += reward
            steps += 1

        if total_reward == 1:
            successes += 1

        print(
            "Episode:", episode,
            "Steps:", steps,
            "Reward:", total_reward
        )

    success_rate = successes / num_episodes
    return success_rate

# -----------------------------
# Evaluate Q Learning
# -----------------------------
print("\nEvaluating Q-Learning Policy\n")
q_learning_success = evaluate_agent(
    q_learning_q,
    num_episodes=5,      
    visualize=True
)

# -----------------------------
# Evaluate SARSA
# -----------------------------
print("\nEvaluating SARSA Policy\n")
sarsa_success = evaluate_agent(
    sarsa_q,
    num_episodes=5,     
    visualize=True
)

env.close()

print("\nFinal Results:")
print("Q-Learning Success Rate:", q_learning_success)
print("SARSA Success Rate:", sarsa_success)
