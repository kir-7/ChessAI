from mcts import MCTS
from node import Node
import chess
import random
import time
from tqdm import tqdm

## if we train the mcts before we start the process
## assuming an average game lasts for 200 moves 
## we are simulating 50 per move so 50*200 = 10000 simulations before the process should do


def play_game():
    tree = MCTS()
    board = new_chess_board()
    print(board)

    moves = 0

    while True:

        time.sleep(1)

        move = board.board.san(random.choice(list(board.board.legal_moves)))        
        board = board.make_move(move)
        
        print('----------------')
        print(board)
        print('----------------')
        
        if board.terminal:
            break
        # You can train as you go, or only at the beginning.
        # Here, we train as we go, doing fifty rollouts each turn.
        for _ in tqdm(range(50)):  
            tree.do_rollout(board)

        board = tree.choose(board)
        print(board)
        
        if board.terminal:
            break

        moves += 2

    print(f'the entire lasted for {moves} moves (including moves from both players)')

def new_chess_board():
    return Node(chess.Board(), turn=True, winner=None, terminal=False)


if __name__ == "__main__":
    play_game()