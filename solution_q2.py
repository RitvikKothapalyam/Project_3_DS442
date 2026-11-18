from collections import defaultdict

import gymnasium as gym

env = gym.make(
    "FrozenLake-v1",
    desc=None,
    map_name="4x4",
    is_slippery=True,
)

episodes = 1_000
action_counts = defaultdict(int)
transition_counts = defaultdict(int)
reward_totals = defaultdict(float)
terminal_transitions = set()
goals_reached = 0

for _ in range(episodes):
    state, _ = env.reset()

    while True:
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)

        transition = (state, action, next_state)
        action_counts[state, action] += 1
        transition_counts[transition] += 1
        reward_totals[transition] += float(reward)

        if terminated:
            terminal_transitions.add(transition)
            if float(reward) > 0:
                goals_reached += 1

        if terminated or truncated:
            break

        state = next_state

env.close()

# For each observed move, estimate its possible outcomes and rewards.
transition_model = defaultdict(list)

for transition, count in transition_counts.items():
    state, action, next_state = transition
    probability = count / action_counts[state, action]
    average_reward = reward_totals[transition] / count
    is_terminal = transition in terminal_transitions

    transition_model[state, action].append(
        (next_state, probability, average_reward, is_terminal)
    )

print(f"Random episodes completed: {episodes}")
print(f"Goals reached: {goals_reached}")
print(f"Observed state-action pairs: {len(transition_model)}")

num_states = 16
num_actions = 4
gamma = 0.99
tolerance = 0.000001


def action_value(state, action, values):
    total = 0.0
    for next_state, probability, reward, is_terminal in transition_model.get(
        (state, action), []
    ):
        future_value = 0.0 if is_terminal else values[next_state]
        total += probability * (reward + gamma * future_value)
    return total


values = [0.0] * num_states
sweeps = 0

while True:
    new_values = [
        max(action_value(state, action, values) for action in range(num_actions))
        for state in range(num_states)
    ]

    max_change = max(
        abs(new - old) for new, old in zip(new_values, values)
    )
    values = new_values
    sweeps += 1

    if max_change < tolerance:
        break

print(f"\nValue iteration converged after {sweeps} sweeps.")
print("Learned state values:")
for row in range(0, num_states, 4):
    print(" ".join(f"{value:.3f}" for value in values[row:row + 4]))

action_symbols = ["L", "D", "R", "U"]
policy = {}

for state in range(num_states):
    observed_actions = [
        action for action in range(num_actions)
        if (state, action) in transition_model
    ]

    if observed_actions:
        policy[state] = max(
            observed_actions,
            key=lambda action: action_value(state, action, values),
        )

print("\nLearned policy:")
for row in range(0, num_states, 4):
    cells = [
        action_symbols[policy[state]] if state in policy else "-"
        for state in range(row, row + 4)
    ]
    print(" ".join(cells))


demo_env = gym.make(
    "FrozenLake-v1",
    desc=None,
    map_name="4x4",
    is_slippery=True,
    render_mode="human",
)

for episode in range(1, 4):
    state, _ = demo_env.reset()

    while True:
        action = policy[state]
        state, reward, terminated, truncated, _ = demo_env.step(action)

        if terminated or truncated:
            break

    if terminated and float(reward) > 0:
        result = "Reached the goal"
    elif terminated:
        result = "Fell into a hole"
    else:
        result = "Reached the step limit"

    print(f"Demo episode {episode}: {result}")

demo_env.close()