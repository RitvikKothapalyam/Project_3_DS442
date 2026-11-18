import gymnasium as gym

env = gym.make(
    "FrozenLake-v1",
    desc=None,
    map_name="4x4",
    is_slippery=True,
    render_mode="human",
)

state, _ = env.reset()

for _ in range(50):
    action = env.action_space.sample()
    state, reward, terminated, truncated, _ = env.step(action)

    if terminated or truncated:
        state, _ = env.reset()

env.close()