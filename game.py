from tkinter import Label, Button, DISABLED, Tk
import pickle
import numpy as np
import random

PLAYER_SYMBOL = "0"
AI_SYMBOL = "X"
EMPTY_CELL = 0

# Load the Q-table
with open("qtable_gen_3.pickle", "rb") as file:
    q_table = pickle.load(file)


class TicTacToeGUI(Tk):

    def __init__(self):
        super().__init__()

        self.title("Tic Tac Toe")

        self.team_dict = {"0": 1, "X": 2}

        # Initialize the game board (3x3 grid)
        self.board = [[EMPTY_CELL for _ in range(3)] for _ in range(3)]

        # Create buttons for the grid
        self.buttons = [[Button(self, text=" ", font='Arial 20', width=5, height=2,
                                   command=lambda r=r, c=c: self.player_move(r, c))
                         for c in range(3)] for r in range(3)]

        # Place buttons in the grid
        for r in range(3):
            for c in range(3):
                self.buttons[r][c].grid(row=r, column=c)

        self.status_label = Label(self, text="Your turn!", font='Arial 12')
        self.status_label.grid(row=3, column=0, columnspan=3)
        self.reset_button = Button(self, text="Reset", font='Arial 12', command=self.reset)
        self.reset_button.grid(row=4, column=0, columnspan=3)

        whoPlays = random.choice([1, 2])
        if whoPlays == 2:
            self.status_label.config(text="AI Turn!")
            self.ai_move()
        else:
            self.status_label.config(text="Your Turn!")

        # self.best_move_label = Label(root, text="Counter Move: ", font='Arial 12')
        # self.best_move_label.grid(row=5, column=0, columnspan=3)

    def player_move(self, row, col):
        if self.board[row][col] == EMPTY_CELL:  # Check if the cell is empty
            self.board[row][col] = 1
            self.buttons[row][col].config(text=PLAYER_SYMBOL, state=DISABLED)

            if self.check_winner(1):
                self.status_label.config(text="You win!")
                self.disable_all_buttons()
            elif self.is_draw():
                self.status_label.config(text="It's a draw!")
            else:
                self.ai_move()

    def ai_move(self):

        # best_move_found, row_to_mark, col_to_mark = self.player_best_move(self.board, self.team_dict[AI_SYMBOL])
        #
        # if best_move_found:
        #     self.best_move_label.config(text=f"Counter Move: Done in row: {row_to_mark}, col: {col_to_mark}")
        # else:
        #     self.best_move_label.config(text=f"Counter Move:")

        obs = tuple(tuple(row) for row in self.board)  # Convert to tuple for Q-table lookup
        action = np.argmax(q_table[obs])  # AI chooses the best action

        free_cells = [(r, c) for r in range(3) for c in range(3) if self.board[r][c] == EMPTY_CELL]
        move = free_cells[action]  # Convert action index to board position

        self.board[move[0]][move[1]] = 2
        self.buttons[move[0]][move[1]].config(text=AI_SYMBOL, state=DISABLED)


        if self.check_winner(2):
            self.status_label.config(text="AI wins!")
            self.disable_all_buttons()
        elif self.is_draw():
            self.status_label.config(text="It's a draw!")
        else:
            self.status_label.config(text="Your turn!")

    def check_winner(self, player) -> bool:
        for row in self.board:
            if all([cell == player for cell in row]):
                return True
        for col in range(3):
            if all([self.board[row][col] == player for row in range(3)]):
                return True
        if self.board[0][0] == self.board[1][1] == self.board[2][2] == player:
            return True
        if self.board[2][0] == self.board[1][1] == self.board[0][2] == player:
            return True
        return False

    def is_draw(self):
        return all(cell != EMPTY_CELL for row in self.board for cell in row)

    def disable_all_buttons(self):
        for row in self.buttons:
            for button in row:
                button.config(state=DISABLED)


    def reset(self):
        self.board = [[EMPTY_CELL for _ in range(3)] for _ in range(3)]
        self.buttons = [[Button(self, text=" ", font='Arial 20', width=5, height=2,
                                   command=lambda r=r, c=c: self.player_move(r, c))
                         for c in range(3)] for r in range(3)]

        # Place buttons in the grid
        for r in range(3):
            for c in range(3):
                self.buttons[r][c].grid(row=r, column=c)

        whoPlays = random.choice([1, 2])
        if whoPlays == 2:
            self.status_label.config(text="AI Turn!")
            self.ai_move()
        else:
            self.status_label.config(text="Your Turn!")


    def player_best_move(self, state: list, player_mark: int) -> [bool, int, int]:
        best_move_found = False
        row_to_mark = -1
        col_to_mark = -1

        # Check for horizontal lines
        for row, layers in enumerate(state):
            if layers.count(player_mark) == 2 and layers.count(0) == 1:
                row_to_mark = row
                col_to_mark = layers.index(0)
                best_move_found = True
                return best_move_found, row_to_mark, col_to_mark

        # Check for vertical lines
        for col in range(3):
            current_col = []
            for layers in state:
                current_col.append(layers[col])
            if current_col.count(player_mark) == 2 and current_col.count(0) == 1:
                row_to_mark = current_col.index(0)
                col_to_mark = col
                best_move_found = True
                return best_move_found, row_to_mark, col_to_mark

        # Check for slash lines
        if state[0][0] == state[1][1] == player_mark and state[2][2] == 0:
            row_to_mark = 2
            col_to_mark = 2
            best_move_found = True
        elif state[0][0] == state[2][2] == player_mark and state[1][1] == 0:
            row_to_mark = 1
            col_to_mark = 1
            best_move_found = True
        elif state[1][1] == state[2][2] == player_mark and state[0][0] == 0:
            row_to_mark = 0
            col_to_mark = 0
            best_move_found = True


        elif state[2][0] == state[1][1] == player_mark and state[0][2] == 0:
            row_to_mark = 0
            col_to_mark = 2
            best_move_found = True
        elif state[2][0] == state[0][2] == player_mark and state[1][1] == 0:
            row_to_mark = 1
            col_to_mark = 1
            best_move_found = True
        elif state[1][1] == state[0][2] == player_mark and state[2][0] == 0:
            row_to_mark = 2
            col_to_mark = 0
            best_move_found = True


        return best_move_found, row_to_mark, col_to_mark



if __name__ == "__main__":
    TicTacToeGUI().mainloop()
