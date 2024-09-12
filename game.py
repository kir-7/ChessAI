from mcts import MCTS
from ChessEnv import ChessEnv
from agent import Agent
import config
import utils

import numpy as np
import os
import uuid

class Game:
    '''
     the main class that runs the game. 
     the class that is going to be backend for self play and is responsible for creating the data fom neural net training
    '''

    def __init__(self, env:ChessEnv, white:Agent, black:Agent) -> None:
        
        self.env = env
        self.white = white
        self.black = black
        
        self.memory = []

        self.reset()

    def reset(self):
        self.env.reset()
        self.turn = self.env.board.turn

    @staticmethod
    def get_winner(result : str):
        return 1 if result == "1-0" else - 1 if result == "0-1" else 0

    @utils.time_function
    def play_game(self, stochastic:bool =True):
        '''
        Play one game from the starting position, and save it to memory.
        Keep playing moves until either the game is over, or it has reached the move limit.
        If the move limit is reached, the winner is estimated.
        '''
        self.reset()
        self.memory.append([])

        counter, previous_nodes, full_game = 0, (None, None), True
        print("started to play the game.....")

        while not self.env.board.is_game_over():
            # play one move (previous move is used for updating the MCTS tree)
            previous_nodes = self.play_one_move(stochastic=stochastic, previous_nodes=previous_nodes)
            print(f"move:{counter}")
            # end if the game drags on too long
            counter += 1
            if counter > config.MAX_GAME_MOVES or self.env.board.is_repetition(3):
                # estimate the winner based on piece values
                winner = ChessEnv.estimate_winner(self.env.board)
                full_game = False
                break

        if full_game:
            # get the winner based on the result of the game
            winner = Game.get_winner(self.env.board.result())
            
        # save game result to memory for all games
        for index, element in enumerate(self.memory[-1]):
            self.memory[-1][index] = (element[0], element[1], winner)

        # save memory to file
        self.save_game(name="game", full_game=full_game)

        return winner

        # TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO

        ## TODO: finish this Game class as well as see about the self.root in mcts and if the current impl is correct or not
        ##       shifted the self.root from mcts to agent. this will change what i store in previous_moves which will change what 
        #        is stored in the data that the NNet trains on so impl carefully


    def play_one_move(self, stochastic:bool=True, previous_nodes=(None, None), save_moves=True):
        """
        Play one move. If stochastic is True, the move is chosen using a probability distribution.
        Otherwise, the move is chosen based on the highest N (deterministically).
        The previous moves are used to reuse the MCTS tree (if possible): the root node is set to the
        node found after playing the previous moves in the current tree.
        """

        current_player = self.white if self.turn else self.black

        #  since my impl of mcts doesn't use a root node it is belived that current_player's mcts tree can be started from any node
        # this impl differs from referenced impl so need to test this thoroughly

        if previous_nodes[0] is None or previous_nodes[1] is None:
            # create new tree with root node == current board
            current_player.mcts = MCTS(current_player, player=current_player.player, stochastic=stochastic)
        else:
            # change the root node to the node after playing the two previous moves
            try:
                current_player.root = previous_nodes[1]

            except AttributeError:
                current_player.mcts = MCTS(current_player, player=current_player.player, stochastic=stochastic)
        
        current_player.run_simulaions(n=config.SIMULATIONS_PER_MOVE)

        moves = current_player.get_moves()
        
        if save_moves:
            self.save_to_memory(self.env.board.fen(), moves)

        best_move = current_player.get_best_move()
        
        # self.env.step(best_move.action)
        self.env.board = best_move.board  # instead of doing as shown above we will directly change the board of env fro best_move impl difference, check properly

        #switch turn
        self.turn = not self.turn
        return (previous_nodes[1], best_move)
    

    def save_to_memory(self, state, moves) -> None:
        """
        Append the current state and move probabilities to the internal memory.
        """


        current_player = self.white if self.turn else self.black

        sum_move_visits = sum(current_player.mcts.N[node] for node, action in moves)
        # create dictionary of moves and their probabilities
        search_probabilities = {
            action.uci(): current_player.mcts.N[node] / sum_move_visits for node, action in moves}
        
        # winner gets added after game is over
        
        self.memory[-1].append((state, search_probabilities, None))

    def save_game(self, name: str = "game", full_game: bool = False) -> None:
        """
        Save the internal memory to a .npy file.
        """
        # the game id consist of game + datetime
        game_id = f"{name}-{str(uuid.uuid4())[:8]}"
        
        print(game_id)

        if full_game:
            # if the game result was not estimated, save the game id to a seperate file (to look at later)
            with open("full_games.txt", "a") as f:
                f.write(f"{game_id}.npy\n")
        
        print(self.memory[-1])
        print(len(self.memory[-1]))
        # np.save(os.path.join(config.MEMORY_DIR, game_id), self.memory[-1])

        