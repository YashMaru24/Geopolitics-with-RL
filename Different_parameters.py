import gymnasium as gym
import numpy as np

# -----------------------------
# Environment
# -----------------------------
env = gym.make(
    "FrozenLake-v1",
    map_name="8x8",
    is_slippery=True
)

n_states = env.observation_space.n
n_actions = env.action_space.n

# -----------------------------
# ε-greedy policy with tie-breaking
# -----------------------------
def policy(Q, state, epsilon):
    if np.random.random() < epsilon:
        return np.random.randint(n_actions)
    max_q = np.max(Q[state])
    best_actions = np.where(Q[state] == max_q)[0]
    return np.random.choice(best_actions)

# -----------------------------
# Train + Evaluate function
# -----------------------------
def train_and_evaluate(
    alpha=0.1,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_decay=None,
    episodes=20000,
    eval_episodes=1000
):
    Q = np.zeros((n_states, n_actions))
    epsilon = epsilon_start

    # Training
    for ep in range(episodes):
        state, _ = env.reset()
        done = False

        while not done:
            action = policy(Q, state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            best_next = np.max(Q[next_state])
            Q[state][action] += alpha * (
                reward + gamma * best_next - Q[state][action]
            )

            state = next_state

        if epsilon_decay is not None:
            epsilon = max(0.05, epsilon * epsilon_decay)

    # Evaluation (greedy)
    successes = 0
    for _ in range(eval_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = policy(Q, state, 0.0)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

        if total_reward == 1:
            successes += 1

    return successes / eval_episodes

# ======================================================
# EXPERIMENT 1: Exploration Strategy
# ======================================================
print("\n=== Experiment 1: Exploration Strategy ===")

fixed_eps_success = train_and_evaluate(
    epsilon_start=0.1,
    epsilon_decay=None
)

decay_eps_success = train_and_evaluate(
    epsilon_start=1.0,
    epsilon_decay=0.999
)

print(f"Fixed ε = 0.1 Success Rate: {fixed_eps_success:.3f}")
print(f"Decaying ε Success Rate: {decay_eps_success:.3f}")

# ======================================================
# EXPERIMENT 2: Learning Rate Sensitivity
# ======================================================
print("\n=== Experiment 2: Learning Rate Sensitivity ===")

for alpha in [0.05, 0.1, 0.3]:
    success = train_and_evaluate(alpha=alpha)
    print(f"α = {alpha:.2f} → Success Rate: {success:.3f}")

# ======================================================
# EXPERIMENT 3: Discount Factor Sensitivity
# ======================================================
print("\n=== Experiment 3: Discount Factor Sensitivity ===")

for gamma in [0.90, 0.95, 0.99]:
    success = train_and_evaluate(gamma=gamma)
    print(f"γ = {gamma:.2f} → Success Rate: {success:.3f}")

env.close()
