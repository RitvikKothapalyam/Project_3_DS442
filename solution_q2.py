import gymnasium as gym
import random

# no picture for the random part so it goes fast
env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)

# Your code for Q2.2 which Executes Random Policy until 1000 episodes

# these dictionaries remember everything the bot sees
times_tried = {}       # how many times i did action a in state s
times_landed = {}      # how many times (s, a) took me to s2
reward_added = {}      # all the rewards i got for (s, a, s2) added up
game_enders = []       # moves that ended the game (hole or goal)
goals = 0

for episode in range(1000):
    state, info = env.reset()
    done = False

    while done == False:
        action = env.action_space.sample()  # just pick a random move
        new_state, reward, terminated, truncated, info = env.step(action)

        move = (state, action, new_state)

        if (state, action) not in times_tried:
            times_tried[(state, action)] = 0
        times_tried[(state, action)] = times_tried[(state, action)] + 1

        if move not in times_landed:
            times_landed[move] = 0
            reward_added[move] = 0
        times_landed[move] = times_landed[move] + 1
        reward_added[move] = reward_added[move] + reward

        if terminated:
            if move not in game_enders:
                game_enders.append(move)
            if reward > 0:
                goals = goals + 1

        state = new_state
        if terminated or truncated:
            done = True

env.close()

# now make T(s'|s,a) and R(s,a,s') from the data
# T = how many times i landed there / how many times i tried that move
# R = the average reward for that move
# model[(s, a)] is a list of [s2, T, R, did the game end]
model = {}
for move in times_landed:
    s = move[0]
    a = move[1]
    s2 = move[2]
    T = times_landed[move] / times_tried[(s, a)]
    R = reward_added[move] / times_landed[move]
    ended = move in game_enders

    if (s, a) not in model:
        model[(s, a)] = []
    model[(s, a)].append([s2, T, R, ended])

print("Random games played:", 1000)
print("Times it reached the goal:", goals)
print("State and action pairs it saw:", len(model))


# Your code for Q2.3 which implements Value Iteration

discount = 0.99
small_number = 0.000001  # stop when the values barely change


# works out how good it is to do action a in state s
def get_q(s, a, V):
    total = 0
    if (s, a) in model:
        for s2, T, R, ended in model[(s, a)]:
            if ended:
                later = 0  # nothing after the game ends
            else:
                later = V[s2]
            total = total + T * (R + discount * later)
    return total


V = [0] * 16  # every square starts at 0
rounds = 0

while True:
    new_V = []
    for s in range(16):
        best = get_q(s, 0, V)
        for a in range(1, 4):
            if get_q(s, a, V) > best:
                best = get_q(s, a, V)
        new_V.append(best)

    # find the biggest change this round
    biggest_change = 0
    for s in range(16):
        change = abs(new_V[s] - V[s])
        if change > biggest_change:
            biggest_change = change

    V = new_V
    rounds = rounds + 1

    if biggest_change < small_number:
        break

print()
print("Value iteration finished after", rounds, "rounds")
print("Values for each square:")
for row in range(4):
    line = ""
    for col in range(4):
        line = line + str(round(V[row * 4 + col], 3)) + "  "
    print(line)


# Your code for Q2.4 which implements Policy Extraction

# for every square pick the move with the best score
# 0 = left, 1 = down, 2 = right, 3 = up
arrows = ["L", "D", "R", "U"]
policy = {}

for s in range(16):
    best_move = None
    best_score = None
    for a in range(4):
        if (s, a) in model:  # only moves i actually saw
            score = get_q(s, a, V)
            if best_score == None or score > best_score:
                best_score = score
                best_move = a
    if best_move != None:
        policy[s] = best_move

print()
print("Best move for each square (- means no moves seen there):")
for row in range(4):
    line = ""
    for col in range(4):
        s = row * 4 + col
        if s in policy:
            line = line + arrows[policy[s]] + " "
        else:
            line = line + "- "
    print(line)


# Your code for Q2.5 which executes the optimal policy

# turn the picture on so we can watch it
env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True,
               render_mode="human")

for episode in range(3):
    state, info = env.reset()
    done = False

    while done == False:
        action = policy[state]  # use the best move, not a random one
        state, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            done = True

    if terminated and reward > 0:
        print("Try", episode + 1, ": made it to the present!")
    elif terminated:
        print("Try", episode + 1, ": fell in a hole")
    else:
        print("Try", episode + 1, ": ran out of steps")

env.close()
