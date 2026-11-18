import gymnasium as gym
import random

env = gym.make('Blackjack-v1', natural=False, sab=False)

###YOUR Q-LEARNING CODE BEGINS

# the q table holds a score for every state
# a state is (my cards total, dealer card, if i have an ace i can use)
# every state has 2 scores, one for stick (0) and one for hit (1)
q = {}
for my_total in range(32):
    for dealer_card in range(11):
        for ace in range(2):
            q[(my_total, dealer_card, ace)] = [0, 0]

# numbers for learning
learn_rate = 0.1     # how much the bot learns each time
discount = 0.95      # how much it cares about later rewards
explore = 0.1        # how often it tries a random move
games = 10000

wins = 0

for game in range(games):
    state, info = env.reset()
    done = False

    while done == False:
        # sometimes pick a random move so the bot tries new stuff
        if random.random() < explore:
            action = env.action_space.sample()
        else:
            # otherwise pick the move with the bigger score
            if q[state][0] > q[state][1]:
                action = 0
            elif q[state][1] > q[state][0]:
                action = 1
            else:
                action = random.choice([0, 1])  # tie so just guess

        new_state, reward, terminated, truncated, info = env.step(action)

        # this is the q learning part
        if terminated:
            goal = reward  # game is over so there is no next state
        else:
            goal = reward + discount * max(q[new_state])

        q[state][action] = q[state][action] + learn_rate * (goal - q[state][action])

        state = new_state
        if terminated or truncated:
            done = True

    # say if the bot won or lost
    if reward > 0:
        print("Game", game + 1, ": the bot WON")
        wins = wins + 1
    elif reward < 0:
        print("Game", game + 1, ": the bot lost")
    else:
        print("Game", game + 1, ": it was a tie")

print("The bot won", wins, "out of", games, "games")
print("Win rate:", round(wins / games * 100, 1), "%")

env.close()

# now watch the bot play a few games with the picture on
env = gym.make('Blackjack-v1', natural=False, sab=False, render_mode="human")

show_wins = 0
show_games = 5

for game in range(show_games):
    state, info = env.reset()
    done = False

    while done == False:
        # no random moves now, just the best one
        if q[state][1] > q[state][0]:
            action = 1
        else:
            action = 0

        state, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            done = True

    if reward > 0:
        print("Show game", game + 1, ": the bot WON")
        show_wins = show_wins + 1
    elif reward < 0:
        print("Show game", game + 1, ": the bot lost")
    else:
        print("Show game", game + 1, ": it was a tie")

print("The bot won", show_wins, "out of", show_games, "show games")

env.close()

###YOUR Q-LEARNING CODE ENDS
