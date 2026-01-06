# Analysis of Geopolitical Factors over Global Supply Chain using Reinforcement Learning

This repository contains my submissions for two Reinforcement Learning assignments completed as part of **Winter in Data Science (WiDS) 2025**. The assignments focus on both **foundational tabular RL methods** and **continuous-control deep RL**, with an emphasis on experimentation, analysis, and environment modeling rather than just achieving high rewards.


## Assignment 1: Reinforcement Learning on FrozenLake

### Objective

To deeply understand **Q-learning (off-policy)** and **SARSA (on-policy)** algorithms by implementing them from scratch on the **FrozenLake environment**, and analyzing their behavior under different hyperparameter settings.

### Environment

* **FrozenLake ** for Q-learning experiments
* **FrozenLake ** for Q-learning vs SARSA comparison
* Stochastic transitions (`is_slippery=True`)

### Key Experiments

#### 1. Exploration Strategy

* Fixed ε greedy exploration
* Decaying εₜ strategy
* Plotted **success rate vs training episodes**

#### 2. Learning Rate Sensitivity

* Compared performance for multiple values of **α**
* Analyzed:

  * Stability of learning
  * Speed of convergence
  * Variance across runs

#### 3. Discount Factor Sensitivity

* γ ∈ {0.90, 0.95, 0.99}
* Studied:

  * Risk sensitive behavior near holes
  * Trade off between short term safety and long term reward

### Post-Training Analysis

* Extracted greedy policy:
  π(s) = arg maxₐ Q(s, a)
* Visualized learned policies on the grid
* Evaluated **empirical success probability over 1,000 episodes**

### Q-learning vs SARSA Comparison

* Compared:

  * Episode wise cumulative reward
  * Success rate
  * Final learned policy
* Highlighted behavioral differences near risky states
* Provided theoretical explanation covering:

  * On policy vs off policy updates
  * Role of exploration in value updates
  * Effect of stochastic transitions

---

##  Assignment 2: Driving on an Annular Track using Reinforcement Learning

### Objective

To design a **custom continuous-control driving environment**, model realistic physics, and evaluate suitable RL algorithms for minimizing lap time while staying within track boundaries.

### Environment Design

* **Annular (circular) track** with inner and outer radius
* Continuous state space (position, velocity, orientation, etc.)
* Action space:

  * Accelerate
  * Decelerate
  * Maintain speed
  * Steering angle δ ∈ [−90°, +90°]

### Reward Design

* Function of:

  * Distance from center of the track
  * Angular progress around the annulus
  * Penalties for going off-track or unstable motion

### RL Algorithm Analysis

Surveyed and compared suitability of:

* Tabular methods (not scalable)
* Deep Q-Networks (DQN)
* On-policy methods
* Off-policy methods
* Actor-Critic architectures

### Chosen Approach

* **Proximal Policy Optimization (PPO)**

**Reasons:**

* Stable updates in continuous action spaces
* Proven performance in control problems
* Widely used in driving and robotics research

### Implementation Highlights

* Custom environment built using **Gymnasium-style API**
* PPO agent trained on the environment
* Optional rendering for debugging and visualization



##  Notes

These assignments were completed as part of an academic course. The focus is on **understanding, experimentation, and analysis**, not just final performance metrics.

Feel free to explore the code, plots, and analysis files. Suggestions and discussions are welcome.

---

**Mentor:** Adarsh Prajapati
**Program:** Winter in Data Science (WiDS) 2025
