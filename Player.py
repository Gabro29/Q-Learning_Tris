from math import fsum

class Player:

    def __init__(self, team: str, table: tuple):
        team_dict = {"0": 1, "X": 2}
        self.player_team_sign = team_dict[team]
        self.current_table = table
        self.index_of_move_done = None
        self.win_game = 0
        self.lose_game = 0
        self.draw_game = 0

    def __str__(self):
        return f"WIN RATE: {round(self.win_game / fsum([self.win_game, self.lose_game]), 2)} - Wins: {self.win_game} - Loss: {self.lose_game} - Draw: {self.draw_game}"

    def init_table(self):
        TABLE = ((0, 0, 0),
                 (0, 0, 0),
                 (0, 0, 0))
        self.current_table = TABLE

    def increment_win_game(self):
        self.win_game += 1

    def increment_lose_game(self):
        self.lose_game += 1

    def increment_draw_game(self):
        self.draw_game += 1

    def set_current_table(self, current_table: tuple):
        self.current_table = current_table

    def observe(self) -> tuple:
        return self.current_table

    @staticmethod
    def index_of_available_cells(current_table: tuple) -> dict:
        available_cells = dict()
        count = 0
        absolute_index = 0
        for layer in current_table:
            for element in layer:
                if element == 0:
                    available_cells[count] = absolute_index
                    count += 1
                absolute_index += 1
        return available_cells

    def action(self, current_table: tuple, q_index: int) -> tuple:
        self.set_current_table(current_table)
        self.index_of_move_done = self.index_of_available_cells(current_table)[q_index]
        new_table = list()
        current_layer = list()
        absolute_index = 0
        for layer in current_table:
            current_layer.clear()
            for element in layer:
                if absolute_index == self.index_of_move_done:
                    current_layer.append(self.player_team_sign)
                else:
                    current_layer.append(element)
                absolute_index += 1
            new_table.append(tuple(current_layer))
        return tuple(new_table)
