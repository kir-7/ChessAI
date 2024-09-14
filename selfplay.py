import os
import time
from agent import Agent
import random
from ChessEnv import ChessEnv
from game import Game
import config
import numpy as np
import chess



def setup(starting_position: str = chess.STARTING_FEN, stochastic=True) -> Game:
    """
    Setup function to set up a game. 
    This can be used in both the self-play and puzzle solving function
    """
    # set different random seeds for each process
    number = random.randint(0, 12345678)    
    np.random.seed(number)
    print(f"========== > Setup. Test Random number: {np.random.randint(0, 123456789)}")


    # create environment and game
    env = ChessEnv(fen=starting_position)

    # create agents
    model_path = os.path.join(config.MODEL_FOLDER, "model.h5")
    
    if os.path.exists(model_path):
        white = Agent(chess.WHITE, model_path, stochastic=stochastic)
        black = Agent(chess.BLACK, model_path, stochastic=stochastic)
    else:
        white = Agent(chess.WHITE, stochastic=stochastic)
        black = Agent(chess.BLACK, stochastic=stochastic)
    return Game(env=env, white=white, black=black)

def self_play(stochastic=True):
    """
    Continuously play games against itself
    """
    game = setup(stochastic=stochastic)

    # play games continuously
    count = 0
    while True:
        print(f"number of games played:{count}")
        game.play_game(stochastic=stochastic)
        count += 1
        
        if count >config.SELFPLAY_GAMES:
            break 

    print(f"finished self play. total games played:{count}")

if __name__ == "__main__":
    # argparse
    stochastic = True
    self_play(stochastic=stochastic)
    