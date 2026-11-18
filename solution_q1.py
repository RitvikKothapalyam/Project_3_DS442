from itertools import product

import gymnasium as gym

import random

env = gym.make("Blackjack-v1", natural=False, sab=False)

# Each state contains (player sum, dealer card, usable ace).
state_ranges = state_ranges = (range(32), range(11), range(2))

# Each state starts with two scores: [stick, hit].
q_table = {state: [0.0, 0.0] for state in product(*state_ranges)}

alpha = 0.1  
gamma = 0.95 

episodes = 10_000
epsilon = 0.1
wins = 0

for episode in range(1, episodes + 1):
    state, _ = env.reset()

    while True:
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            scores = q_table[state]
            best_score = max(scores)
            best_actions = [
                index for index, score in enumerate(scores)
                if score == best_score
            ]
            action = random.choice(best_actions)

        next_state, reward, terminated, truncated, _ = env.step(action)

        target = float(reward)
        if not terminated:
            target += gamma * max(q_table[next_state])

        old_value = q_table[state][action]
        q_table[state][action] = old_value + alpha * (target - old_value)

        if terminated or truncated:
            break

        state = next_state

    if terminated and float(reward) > 0:
        result = "Win"
        wins += 1
    elif terminated and float(reward) < 0:
        result = "Loss"
    elif terminated:
        result = "Draw"
    else:
        result = "Truncated"

    print(f"Episode {episode}: {result}")

print(f"Training win rate: {wins / episodes:.1%}")

env.close()

demo_env = gym.make(
    "Blackjack-v1",
    natural=False,
    sab=False,
    render_mode="human",
)

demo_wins = 0
demo_hands = 5

for hand in range(1, demo_hands + 1):
    state, _ = demo_env.reset()

    while True:
        scores = q_table[state]
        action = max(range(len(scores)), key=scores.__getitem__)
        state, reward, terminated, truncated, _ = demo_env.step(action)

        if terminated or truncated:
            break

    final_reward = float(reward)
    if terminated and final_reward > 0:
        result = "Win"
        demo_wins += 1
    elif terminated and final_reward < 0:
        result = "Loss"
    elif terminated:
        result = "Draw"
    else:
        result = "Truncated"

    print(f"Demo hand {hand}: {result}")

demo_env.close()
print(f"Demo wins: {demo_wins}/{demo_hands}")