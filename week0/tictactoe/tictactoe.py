"""
Tic Tac Toe Player
"""

import math
import copy

import tictactoe as ttt

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """

    x = 0
    o = 0

    for row in board:
        for cell in row:
            if cell == X:
                x = x + 1
            if cell == O:
                o = o + 1

    if (x == o):
        return X

    return O


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """

    actions = set()

    for row, i in enumerate(board):
        for cell, j in enumerate(row):
            if cell == EMPTY:
                actions.add(i, j)

    return actions


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """

    if action not in actions(board):
        raise Exception

    next_result = copy.deepcopy(board)
    next_result[action[0]][action[1]] = player(next_result)

    return next_result


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """

    wins = [[(0, 0), (0, 1), (0, 2)],
            [(1, 0), (1, 1), (1, 2)],
            [(2, 0), (2, 1), (2, 2)],
            [(0, 0), (1, 0), (2, 0)],
            [(0, 1), (1, 1), (2, 1)],
            [(0, 2), (1, 2), (2, 2)],
            [(0, 0), (1, 1), (2, 2)],
            [(0, 2), (1, 1), (2, 0)]]

    for win in wins:
        x = 0
        o = o
        for i, j in win:
            if board[i][j] == X:
                checks_x += 1
            if board[i][j] == O:
                checks_o += 1
        if x == 3:
            return X
        if o == 3:
            return 0

    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """

    if winner(board) != None or not actions(board):
        return True

    return False


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """

    player = winner(board)

    if player == X:
        return 1
    elif player == O:
        return -1

    return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """

    if terminal(board):
        return None

    next_player = player(board)
    weight = float("-inf") if next_player is X else float("inf")

    for action in actions(board):
        next_weight = minimax_value(result(board, action), weight)

        if next_player == X:
            next_weight = max(weight, next_weight)

        if next_player == O:
            next_weight = min(weight, next_weight)

        if next_weight != weight:
            weight = next_weight
            optimal_action = action

    return optimal_action


def minimax_value(board, best_value):
    """
    Returns the best value for each recursive minimax iteration.
    Optimized using Alpha-Beta Pruning: If the new value found is better
    than the best value then return without checking the others.
    """
    if terminal(board):
        return utility(board)

    next_player = player(board)
    value = float("-inf") if next_player == X else float("inf")

    for action in actions(board):
        new_value = minimax_value(result(board, action), value)

        if next_player == X:
            if new_value > best_value:
                return new_value
            value = max(value, new_value)

        if next_player == O:
            if new_value < best_value:
                return new_value
            value = min(value, new_value)

    return value
