from mcts import MCTS
from ChessEnv import ChessEnv
from agent import Agent
import config

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

    def play_game(self):
        '''
        Play one game from the starting position, and save it to memory.
        Keep playing moves until either the game is over, or it has reached the move limit.
        If the move limit is reached, the winner is estimated.
        '''
        self.reset()
        self.memory.append([])

        counter, full_game = 0, True

        # TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO

        ## TODO: finish this Game class as well as see about the self.root in mcts and if the current impl is correct or not
        ##       shifted the self.root from mcts to agent. this will change what i store in previous_moves which will change what 
        #        is stored in the data that the NNet trains on so impl carefully


    def play_one_move(self, stochastic:bool=True, previous_moves=(None, None), save_moves=True):
        """
        Play one move. If stochastic is True, the move is chosen using a probability distribution.
        Otherwise, the move is chosen based on the highest N (deterministically).
        The previous moves are used to reuse the MCTS tree (if possible): the root node is set to the
        node found after playing the previous moves in the current tree.
        """

        current_player = self.white if self.turn else self.black

        #  since my impl of mcts doesn't use a root node it is belived that current_player's mcts tree can be started from any node
        # this impl differs from referenced impl so need to test this thoroughly

        if previous_moves[0] is None or previous_moves[1] is None:
            # create new tree with root node == current board
            current_player.mcts = MCTS(current_player, player=current_player.player, stochastic=stochastic)
        else:
            # change the root node to the node after playing the two previous moves
            try:

                # TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO

                node = current_player.mcts.root.get_edge(previous_moves[0].action).output_node
                node = node.get_edge(previous_moves[1].action).output_node
                current_player.root = node

            except AttributeError:
                current_player.mcts = MCTS(current_player, player=current_player.player, stochastic=stochastic)

        current_player.run_simulaions(n=config.SIMULATIONS_PER_MOVE)

        moves = current_player.get_moves()



    def save_to_memry(self):
        pass

    def save_game(self):
        pass

        