
import chess
import torch
import config
from mcts import MCTS
from node import Node

from model import RLModel

class Agent:

    def __init__(self, player=chess.BLACK, model_path=None, compile = False, stochastic:bool=True):
        
        if model_path is not None:
            
            self.model = RLModel(config.INPUT_SHAPE, config.OUTPUT_SHAPE) ## intialize a model
            self.model.load_state_dict(torch.load(model_path)) 
        else:
            self.model = self.build_model()
        if compile:
            
            self.model = torch.compile(self.model)
        
        self.root = Node(chess.Board(), winner=None, terminal=False)
        
        self.player = player

        self.mcts = MCTS(self, player=self.player, exploration_weight=config.EXPLORATION_WEIGHT, stochastic=stochastic)


    def build_model(self):
        """
        Build a new model based on the configuration in config.py
        """
        self.model = RLModel(config.INPUT_SHAPE, config.OUTPUT_SHAPE)
        return self.model
    
    def run_simulaions(self, n :int = 50):
        '''
        Run n simulations of the MCTS algorithm. This function gets called every move.
        '''
        
        print(f"Running {n} simulations...")
        self.mcts.run_simulation(n, self.root)
    
    def get_moves(self):
        return self.mcts.get_possible_moves(self.root)

    def get_best_move(self):
        return self.mcts.choose(self.root)
    

    def save_model(self, path):
        '''
        Save the current model to the specified path
        '''

        torch.save(self.model.state_dict(), path)

    def predict(self, data):
        '''
        Predict the value head an policy head for the given state
        '''

        # TODO: needs to properly set the model.train() and model.eval() as well as check for gradients 

        if not torch.is_tensor(data):
            data = torch.Tensor(data)
            
        (p, v), loss = self.model(data)
        return p, v

