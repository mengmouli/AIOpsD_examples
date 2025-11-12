import numpy as np
import gymnasium as gym

# Create environment (text-based rendering)
env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="ansi")
n_states = env.observation_space.n
n_actions = env.action_space.n

# Initialize Q-table
Q = np.zeros((n_states, n_actions))

# Hyperparameters
alpha = 0.8      # learning rate
gamma = 0.95     # discount factor
epsilon = 0.1    # exploration rate
episodes = 2000

# Training loop
for episode in range(episodes):
    state, _ = env.reset()
    done = False
    while not done:
        # epsilon-greedy action selection
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state, :])

        next_state, reward, done, truncated, _ = env.step(action)

        # Q-learning update
        Q[state, action] += alpha * (
            reward + gamma * np.max(Q[next_state, :]) - Q[state, action]
        )

        state = next_state

# Test trained agent
state, _ = env.reset()
done = False
max_steps = 100  # safety cutoff
steps = 0

while not done and steps < max_steps:
    action = np.argmax(Q[state, :])
    state, reward, done, truncated, _ = env.step(action)
    print(env.render())  # show environment as ASCII grid
    steps += 1

print("Final reward:", reward)
print("Learned Q-table:")
print(Q)
