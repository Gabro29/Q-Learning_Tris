# Tic-Tac-Toe solved with Q-Learning

## Overview
In this project I use Q-learning to improve the gameplay of the computer over 
multiple episodes and generations. The program features a reinforcement learning 
approach where two players (human and computer) play against each other, with the
computer agent learning to make optimal moves over time. The AI's progress is tracked
through reward mechanisms and dynamic updates to its Q-table.

## Requirements
- Python 3.x 
- Required libraries:
  - numpy: for efficient numerical calculations 
  - matplotlib: for visualizing the AI's learning progress 
  - itertools: for generating all possible game states 
  - pickle: for saving/loading Q-table data

## File Structure
- Player.py: Defines the Player class, handling game actions and table updates. 
- TicTacToe.py: Main AI logic and Q-learning algorithms.

## Key Components
The program uses several hyperparameters and constants to guide the AI's learning process:
- Rewards and Penalties: Includes win rewards, lose penalties, draw rewards and specific penalties for inefficient or incorrect moves.

## Core Functions
- all_possible_states: Generates all possible board configurations.
- how_many_free_cells: Counts empty cells on the board.
- check_for_wins: Checks for a winning state on the board.
- player_best_move: Computes the best move based on the current board state.
- game_step: Executes a single game move for either the player or computer.
- update_q_values: Updates Q-values in the Q-table based on actions, rewards, and observations.
- compute_reward_end_game: Calculates rewards at the end of the game based on the outcome.
- training: Main function for training the AI by iterating over episodes and generations.

## Q-table
The Q-table is a dictionary where:
- Keys are board states.
- Values are lists of Q-values corresponding to each possible move.

## Training Process
- Initialization: The program initializes a Player and Computer instance, as well as the Q-table.
- Exploration vs Exploitation: The AI uses epsilon-greedy strategy to balance exploration of new moves and exploitation of known rewarding moves.
- Reward Calculation: The AI earns or loses points based on the game's outcome and individual moves.
- Q-value Update: After each move, the Q-table is updated using the Bellman equation to reinforce moves that maximize future rewards.
- Epsilon Decay: Epsilon is decayed after each episode to gradually reduce exploration.


## Saving and Loading the Q-table
The Q-table is saved at the end of each generation as qtable_gen_X.pickle, 
allowing the AI to resume training from previous knowledge or evaluate its performance.
