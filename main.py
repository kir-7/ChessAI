from mcts import MCTS
from node import Node
import chess
import random
import time
from tqdm import tqdm
from agent import Agent
from ChessEnv import ChessEnv
from game import Game
## if we train the mcts before we start the process
## assuming an average game lasts for 200 moves 
## we are simulating 50 per move so 50*200 = 10000 simulations before the process should do

import numpy as np
import config
from trainer import Trainer
from model import RLModel
def play_game():
    
    white = Agent(chess.WHITE)
    black = Agent(chess.BLACK)

    board = new_chess_board()
    print(board)

    moves = 0

    while True:
        
        white.root = board
        white.run_simulations(10)
        board = white.get_best_move()
        
        print('----------------')
        print(board)
        print('----------------')
        if board.terminal:
            break
        
        # You can train as you go, or only at the beginning.
        # Here, we train as we go, doing fifty rollouts each turn.
        black.root = board
        black.run_simulations(10)   ## as we expand the tree the possibilities increase which means the time taken 
        board = black.get_best_move()                                     ## for subsequent moves keeps increasing. so later moves will take longer as tree is larger to explore

        print(board)
        
        if board.terminal:
            break

        moves += 2

    print(f'the entire game lasted for {moves} moves (including moves from both players)')
    print(f"winner:{board.winner}")
    return board

    # TODO: currently loss function, the main updation of model parameters, and the self play are not implemented so for now in mcts we are 
    #       leaving self.P to have values as tensors but ultimately after this we need to decide if we need to store the tensors or just the 
    #       floats are enough  

    
def new_chess_board():
    return Node(chess.Board(), winner=None, terminal=False)  # we are playing against mcts so turn will be true for tree and if
                                                                        # winner is False then we won else board won


if __name__ == "__main__":
    final_board = play_game()
    print(final_board)
    # print(final_board.outcome())
    # white = Agent(chess.WHITE)
    # black = Agent(chess.BLACK)
    # print(ChessEnv.state_to_input(board.board.fen()).shape)
    # agent.run_simulaions(2)
    # moves = agent.get_moves()
    # sum_move_visits = sum(agent.mcts.N[node] for node, action in moves)
    #     # create dictionary of moves and their probabilities
    # search_probabilities = {
    #     action.uci(): agent.mcts.N[node] / sum_move_visits for node, action in moves}
    # print(moves)
    # print()
    # print(sum_move_visits, "\n")
    # print(search_probabilities)
        
    # game = Game(ChessEnv(), white, black)
    # game.play_game(True)
    # model = RLModel(config.INPUT_SHAPE, config.OUTPUT_SHAPE)
    # t = Trainer(model, None)

    # data = np.load(r"D:\ai\chess\memory\game-64dfca9c.npy", allow_pickle=True)
    

    # print(data.shape)

    # x, (y_policy, y_values) = t.split_Xy(data)
    
    # TODO:  UserWarning: Creating a tensor from a list of numpy.ndarrays is extremely slow. Please consider converting the list to a 
    #        single numpy.ndarray with numpy.array() before converting to a tensor. Triggered in trainer.splitXy() at
    #         return torch.Tensor(X), (torch.Tensor(y_probs).reshape(len(y_probs), config.OUTPUT_SHAPE[0]), torch.Tensor(y_value).view(len(y_value), config.OUTPUT_SHAPE[1]))

    # print(y_policy)
    # print(x.shape)
    # print(y_policy.shape)
    # print(y_values.shape)
    # print(t.train_batch(x, y_policy, y_values))