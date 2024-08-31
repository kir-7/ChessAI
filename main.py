from mcts import MCTS
from node import Node
import chess


def play_game():
    tree = MCTS()
    board = new_chess_board()
    print(board.visualize_board())
    while True:
        move = input("enter move ")
        
        board = board.make_move(move)
        print(board.visualize_board())
        if board.terminal:
            break
        # You can train as you go, or only at the beginning.
        # Here, we train as we go, doing fifty rollouts each turn.
        for _ in range(50):
            tree.do_rollout(board)

        board = tree.choose(board)
        print(board.visualize_board())
        if board.terminal:
            break

def new_chess_board():
    return Node(chess.Board(), turn=True, winner=None, terminal=False)


if __name__ == "__main__":
    play_game()