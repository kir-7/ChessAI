from mcts import MCTS
from node import Node
import chess
import random
import time
from tqdm import tqdm
from agent import Agent
from ChessEnv import ChessEnv
## if we train the mcts before we start the process
## assuming an average game lasts for 200 moves 
## we are simulating 50 per move so 50*200 = 10000 simulations before the process should do


def play_game():
    tree = MCTS(player=chess.BLACK)
    board = new_chess_board()
    print(board)

    moves = 0

    while True:
        

        move = board.board.san(random.choice(list(board.board.legal_moves)))        
        board = board.make_move(move)
        
        print('----------------')
        print(board)
        print('----------------')
        if board.terminal:
            break
        
        assert tree.player == board.board.turn 

        # You can train as you go, or only at the beginning.
        # Here, we train as we go, doing fifty rollouts each turn.
        tree.run_simulation(50, board)   ## as we expand the tree the possibilities increase which means the time taken 
                                         ## for subsequent moves keeps increasing. so later moves will take longer as tree is larger to explore

        board = tree.choose(board)
        print(board)
        
        if board.terminal:
            break

        moves += 2

    print(f'the entire game lasted for {moves} moves (including moves from both players)')
    print(f"winner:{board.winner}, turn:{board.turn}")
    return board

    ## TODO: currently hard coded that tree is playing as black and i am playing as white so every turn i make a move and make it again 
    #        the tree's turn to play by setting board.turn = True then the board plays as black and the reward and _find_winner functions
    #        are defined to play this way, if later it is decided that tree plays as white then need to change these functions or error 

def new_chess_board():
    return Node(chess.Board(), winner=None, terminal=False)  # we are playing against mcts so turn will be true for tree and if
                                                                        # winner is False then we won else board won


if __name__ == "__main__":
    # final_board = play_game()
    # print(final_board.outcome())
    agent = Agent(chess.WHITE)
    tree = MCTS(agent, agent.player)

    board = new_chess_board()
    # print(ChessEnv.state_to_input(board.board.fen()).shape)
    tree.run_simulation(10, board)

