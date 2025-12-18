#Q Learning
import gymnasium as gym
import numpy as np
import pickle as pkl

# -----------------------------
# Create FrozenLake Environment
# -----------------------------
env = gym.make(
    "FrozenLake-v1",
    map_name="8x8",
    is_slippery=True
)

n_states = env.observation_space.n
n_actions = env.action_space.n

# -----------------------------
# Initialize Q-table
# -----------------------------
q_table = np.zeros((n_states, n_actions))

# -----------------------------
# Hyperparameters
# -----------------------------
EPSILON = 1.0          # start with full exploration
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.999
ALPHA = 0.1
GAMMA = 0.99
NUM_EPISODES = 20000

# -----------------------------
# epsiolngreedy policy with tiebreaking
# -----------------------------
def policy(state, explore=0.0):
    if np.random.random() < explore:
        return np.random.randint(n_actions)

    max_q = np.max(q_table[state])
    best_actions = np.where(q_table[state] == max_q)[0]
    return np.random.choice(best_actions)

# -----------------------------
# Training Loop
# -----------------------------
success_count = 0

for episode in range(NUM_EPISODES):

    done = False
    total_reward = 0
    episode_length = 0

    state, _ = env.reset()

    while not done:
        
        action = policy(state, EPSILON)

        
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        
        next_action = policy(next_state, 0.0)

        
        q_table[state][action] += ALPHA * (
            reward + GAMMA * q_table[next_state][next_action]
            - q_table[state][action]
        )

        state = next_state
        total_reward += reward
        episode_length += 1

    
    if total_reward == 1:
        success_count += 1
        print(f"SUCCESS at episode {episode}")

    
    EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)

    
    if episode % 1000 == 0:
        print(
            f"Episode {episode} | "
            f"Epsilon {EPSILON:.3f} | "
            f"Successes {success_count}"
        )

# -----------------------------
# Save Q-table
# -----------------------------
env.close()
pkl.dump(q_table, open("frozenlake_q_table.pkl", "wb"))
