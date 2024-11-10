

import random
from Player import Player
import numpy as np
import itertools
import matplotlib.pyplot as plt
import time
import pickle
from matplotlib.animation import FuncAnimation

reverse_team_dict = {1: "0", 2: "X"}
team_dict = {"0": 1, "X": 2}

def all_possible_states() -> tuple:
    """Gives out all possible states of the Tic Tac Toe table"""
    every_combinations_for_a_layer = tuple(itertools.product((0, 1, 2), repeat=3))
    all_combinations = list()
    for f_combo in every_combinations_for_a_layer:
        for s_combo in every_combinations_for_a_layer:
            for t_combo in every_combinations_for_a_layer:
                all_combinations.append((f_combo, s_combo, t_combo))
    return tuple(all_combinations)

def how_many_free_cells(table: tuple) -> int:
    count = 0
    for layer in table:
        for element in layer:
            if element == 0:
                count += 1
    return count

def check_for_wins(state: tuple) -> [bool, str]:

    # Check for horizontal lines
    for layers in state:
        if len(set(layers)) == 1 and layers[0] in (1, 2):
            return [True, reverse_team_dict[layers[0]]]
    # Check for vertical lines
    traspose = np.array(state).T.tolist()
    for layers in traspose:
        if len(set(layers)) == 1 and layers[0] in (1, 2):
            return [True, reverse_team_dict[layers[0]]]
    # Check for slash lines
    if state[0][0] in (1, 2) and state[0][0] == state[1][1] == state[2][2]:
        return [True, reverse_team_dict[state[0][0]]]
    if state[2][0] in (1, 2) and state[2][0] == state[1][1] == state[0][2]:
        return [True, reverse_team_dict[state[2][0]]]

    return [False, "None"]

def player_best_move(state: tuple, player_mark: int, whoPlays: str) -> [bool, int]:
    available_cells = dict()
    reversed_available_cell = dict()
    count = 0
    absolute_index = 0
    for row, layer in enumerate(state):
        for col, element in enumerate(layer):
            if element == 0:
                available_cells[count] = absolute_index
                reversed_available_cell[absolute_index] = count
                count += 1
            absolute_index += 1
    # Pick a random move by default
    best_move_found = False
    index_to_mark = random.choice(list(available_cells.values()))


    # Otherwise check for more winning options

    # Check for horizontal lines
    for row, layers in enumerate(state):
        if layers.count(player_mark) == 2 and layers.count(0) == 1:
            index_to_mark = layers.index(0) + (row * 3)
            best_move_found = True
            return best_move_found, reversed_available_cell[index_to_mark]

    # Check for vertical lines
    current_col = list()
    for col in range(3):
        current_col.clear()
        for layers in state:
            current_col.append(layers[col])
        if current_col.count(player_mark) == 2 and current_col.count(0) == 1:
            index_to_mark = (current_col.index(0) * 3) + col
            best_move_found = True
            return best_move_found, reversed_available_cell[index_to_mark]

    # Check for slash lines
    if state[0][0] == state[1][1] == player_mark and state[2][2] == 0:
        best_move_found = True
        index_to_mark = 8
    elif state[0][0] == state[2][2] == player_mark and state[1][1] == 0:
        best_move_found = True
        index_to_mark = 4
    elif state[1][1] == state[2][2] == player_mark and state[0][0] == 0:
        best_move_found = True
        index_to_mark = 0


    elif state[2][0] == state[1][1] == player_mark and state[0][2] == 0:
        best_move_found = True
        index_to_mark = 2
    elif state[2][0] == state[0][2] == player_mark and state[1][1] == 0:
        best_move_found = True
        index_to_mark = 4
    elif state[1][1] == state[0][2] == player_mark and state[2][0] == 0:
        best_move_found = True
        index_to_mark = 6

    return best_move_found, reversed_available_cell[index_to_mark]

def how_many_tris_available(state: tuple, player_mark: int) -> int:

    available_tris = 0

    # Check for horizontal lines
    for row, layers in enumerate(state):
        if layers.count(player_mark) == 2 and layers.count(0) == 1:
            available_tris += 1

    # Check for vertical lines
    current_col = list()
    for col in range(3):
        current_col.clear()
        for layers in state:
            current_col.append(layers[col])
        if current_col.count(player_mark) == 2 and current_col.count(0) == 1:
            available_tris += 1

    # Check for slash lines
    if state[0][0] == state[1][1] == player_mark and state[2][2] == 0:
        available_tris += 1
    elif state[0][0] == state[2][2] == player_mark and state[1][1] == 0:
        available_tris += 1
    elif state[1][1] == state[2][2] == player_mark and state[0][0] == 0:
        available_tris += 1


    elif state[2][0] == state[1][1] == player_mark and state[0][2] == 0:
        available_tris += 1
    elif state[2][0] == state[0][2] == player_mark and state[1][1] == 0:
        available_tris += 1
    elif state[1][1] == state[0][2] == player_mark and state[2][0] == 0:
        available_tris += 1

    return available_tris


def game_step(player: Player, computer: Player, whoPlays: int, moveToMakeNextTime: int,
              computerMakesTrisNextTurn: bool, reward: float, observation: tuple) -> [int, float, tuple, str, int, int, bool]:
    if whoPlays == 1:
        action = player_best_move(observation, 1, "player")[1]

        # TODO: Use the AI against itself

        # Update Table
        new_table = player.action(observation, action)
        whoPlays = team_dict["X"]
        currentPlayer = "player"
        return player, computer, action, reward, new_table, currentPlayer, whoPlays, moveToMakeNextTime, computerMakesTrisNextTurn

    else:
        if np.random.random() > epsilon:
            # action = np.argmax(q_table[observation])
            action = np.argmax(softmax(q_table[observation]))
        else:
            action = np.random.randint(0, how_many_free_cells(observation))

        opponentMakesTrisNextTurn, moveToBlockOpponent = player_best_move(observation, 1, "player")

        if opponentMakesTrisNextTurn:
            if action == moveToBlockOpponent:
                reward += BLOCK_MOVE_DONE_REWARD
            else:
                reward -= BLOCK_MOVE_NOT_DONE_PENALTY

        if computerMakesTrisNextTurn:
            if action != moveToMakeNextTime and moveToMakeNextTime >= 0:
                reward -= END_GAME_WRONG_MOVE_PENALTY
            elif action == moveToMakeNextTime and moveToMakeNextTime >= 0:
                howManyTrisSet = how_many_tris_available(observation, 2)
                reward += END_GAME_RIGHT_MOVE_REWARD
        else:
            computerMakesTrisNextTurn, moveToMakeNextTime = player_best_move(observation, 2, "computer")
            howManyTrisSet = how_many_tris_available(observation, 2)
            if computerMakesTrisNextTurn:
                reward += POTENTIAL_WINNING

        new_table = computer.action(observation, action)
        whoPlays = team_dict["0"]
        currentPlayer = "computer"

        return player, computer, action, reward, new_table, currentPlayer, whoPlays, moveToMakeNextTime, computerMakesTrisNextTurn


def update_q_values(q_table: tuple, action: int, reward: float, observation: tuple, new_observation: tuple) -> tuple:
    global LEARNING_RATE, DISCOUNT_RATE
    # max_future_q = np.max(q_table[new_observation])
    max_future_q = q_table[new_observation][np.argmax(softmax(q_table[new_observation]))]
    current_q = q_table[observation][action]

    new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
    q_table[observation][action] = new_q

    return q_table


def compute_reward_end_game(player: Player, computer: Player, currentPlayer: str, whoWins: str,
                            action: int, reward: float,
                            episode_reward: float, q_table: tuple, step: int,
                            observation: tuple, new_observation: tuple) -> [float, float, tuple]:

    global LOSE_PENALTY, WINNING_REWARD, EARLY_WIN_BONUS

    if whoWins == "0":
        # Player wins
        reward -= LOSE_PENALTY
        player.increment_win_game()
        computer.increment_lose_game()

    elif whoWins == "X":
        # Computer wins
        reward += WINNING_REWARD + EARLY_WIN_BONUS * (9 - step)
        player.increment_lose_game()
        computer.increment_win_game()

    episode_reward += reward

    if currentPlayer == "computer" and len(q_table[new_observation]) > 0:
        q_table = update_q_values(q_table, action, reward, observation, new_observation)

    return player, computer, reward, episode_reward, q_table


def compute_reward_draw(player: Player, computer: Player, currentPlayer: str, action: int,reward: float,
                        episode_reward: float, q_table: tuple,
                        observation: tuple, new_observation: tuple) -> [float, float, tuple]:

    global DRAWING_REWARD

    reward += DRAWING_REWARD
    player.increment_draw_game()
    computer.increment_draw_game()
    episode_reward += reward
    if currentPlayer == "computer" and len(q_table[new_observation]) > 0:
        q_table = update_q_values(q_table, action, reward, observation, new_observation)
    return player, computer, reward, episode_reward, q_table

def softmax(q_values: np.array) -> np.array:
    exps = np.exp(q_values - np.max(q_values))
    return exps / np.sum(exps)


def training(q_table: dict) -> list:

    global SHOW_EVERY, LEARNING_RATE, epsilon, EPISODES, EPS_DECAY, MOVE_PENALTY, WINNING_REWARD, \
        LOSE_PENALTY, BLOCK_MOVE_NOT_DONE_PENALTY, TABLE, DISCOUNT, DRAWING_REWARD, POTENTIAL_WINNING, \
        BLOCK_MOVE_DONE_REWARD, MIN_EPSILON

    episode_rewards = list()
    player = Player("0", TABLE)
    computer = Player("X", TABLE)

    epsilon = INITIAL_EPSILON
    learning_rate = LEARNING_RATE
    discount = DISCOUNT

    for episode in range(EPISODES):
        whoPlays = random.choice([1, 2])

        if episode % SHOW_EVERY == 0:
            print(f"on # {episode}, epsilon: {epsilon}")
            print(f"{SHOW_EVERY} ep mean: {np.mean(episode_rewards[-SHOW_EVERY:])}")

        # print(f"Game number {episode}")
        episode_reward = 0
        player.init_table()
        computer.init_table()
        moveToMakeNextTime = -1
        computerMakesTrisNextTurn = False
        currentPlayer = None

        for step in range(9):
            obs = player.observe()  # same as computer.observe(), since the table is the same
            reward = -MOVE_PENALTY  # by default penalty every move

            (player, computer, action, reward, new_table, currentPlayer, whoPlays, moveToMakeNextTime,
             computerMakesTrisNextTurn) = game_step(player, computer, whoPlays, moveToMakeNextTime,
                                                    computerMakesTrisNextTurn, reward, observation=obs)
            player.set_current_table(new_table)
            computer.set_current_table(new_table)

            new_obs = player.observe()

            ### REWARD PART ###
            game_is_ended, whoWins = check_for_wins(new_table)

            if game_is_ended:
                (player, computer, reward,
                 episode_reward, q_table) = compute_reward_end_game(player, computer, currentPlayer, whoWins,
                                                                    action, reward, episode_reward, q_table, step,
                                                                    observation=obs, new_observation=new_obs)
                break

            elif len(q_table[new_obs]) == 0 and whoWins == "None":
                (player, computer, reward,
                 episode_reward, q_table) = compute_reward_draw(player, computer, currentPlayer, action, reward,
                                                                episode_reward, q_table,
                                                                observation=obs, new_observation=new_obs)
                break
            else:
                episode_reward += reward
                if currentPlayer == "computer" and len(q_table[new_obs]) > 0:
                    q_table = update_q_values(q_table, action, reward, observation=obs, new_observation=new_obs)


        episode_rewards.append(episode_reward)

        # Decay epsilon
        if epsilon > MIN_EPSILON:
            epsilon = max(MIN_EPSILON, INITIAL_EPSILON * np.exp(-0.0001 * episode))
        else:
            epsilon = MIN_EPSILON

    return episode_rewards, q_table, player, computer



LOSE_PENALTY = 10
WINNING_REWARD = 20
DRAWING_REWARD = 1

BLOCK_MOVE_NOT_DONE_PENALTY = 10
BLOCK_MOVE_DONE_REWARD = 6
EARLY_WIN_BONUS = 4

MOVE_PENALTY = 1
POTENTIAL_WINNING = 2

END_GAME_RIGHT_MOVE_REWARD = 2
END_GAME_WRONG_MOVE_PENALTY = 7

INITIAL_EPSILON = 0.9
epsilon = 0.9
EPS_DECAY = 0.98
MIN_EPSILON = 0.01
SHOW_EVERY = 5000


LEARNING_RATE = 0.1
DISCOUNT = 0.99
EPISODES = 100_000
GENERATIONS = 4

# Table grid of the Tic Tac Toe game
TABLE = ( (0, 0, 0),
          (0, 0, 0),
          (0, 0, 0) )

# start_q_table = "qtable_gen_19.pickle"
start_q_table = None

# Here we map every possible state to a list of values. We put how many values as
# the number of free cells in the current state. When generating an action
# we will consider the argmax of that list of random uniform numbers, then the index of that
# value will be the index of the first free cell in the state
if start_q_table is None:
    q_table = {state: [np.random.uniform(-20, 20) for i in range(how_many_free_cells(state))] for state in all_possible_states()}
else:
    with open(start_q_table, "rb") as file:
        q_table = pickle.load(file)


all_moving_averages = []

for gen in range(GENERATIONS):

    print(f"##### Generation {gen} #####")
    epsilon = 0.9

    if start_q_table is None:
        q_table = {state: [np.random.uniform(-10, 10) for i in range(how_many_free_cells(state))] for state in
                   all_possible_states()}
    else:
        with open(start_q_table, "rb") as file:
            q_table = pickle.load(file)

    episode_rewards, q_table, player, computer = training(q_table)
    print(f"Player stats {player}")
    print(f"Computer stats {computer}")
    moving_average = np.convolve(episode_rewards, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode="valid")
    all_moving_averages.append(moving_average)

    plt.plot([i for i in range(len(moving_average))], moving_average)
    plt.ylabel(f"reward {SHOW_EVERY}ma")
    genth = gen
    plt.xlabel("episode #")
    plt.savefig(f"plots/reward_gen_{genth}.png")
    plt.show()
    start_q_table = f"qtable_gen_{genth}.pickle"
    with open(start_q_table, "wb") as file:
        pickle.dump(q_table, file)

