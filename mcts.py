import chess
import chess.pgn
import chess.engine

from node import Node

from collections import defaultdict
import random
import math




class MCTS:
    ''' Monte carlo tree search class
        https://gist.github.com/qpwo/c538c6f73727e254fdc7fab81024f6e1
    '''

    def __init__(self, exploration_weight:int=1):
        self.Q = defaultdict(int)  # collection of rewards for each node
        self.N = defaultdict(int)  # collection of visits for each node
        self.children = dict()     # children of each node
        self.exploration_weight = exploration_weight
    
    def choose(self, node):
        " choose the best successor"

        if node.is_terminal():
            raise RuntimeError(f"choose called on terminal node: {node}")

        if node not in self.children:
            return node.find_random_children()
        
        def score(node):
            if self.N[node] == 0:
                return float('-inf')
            return self.Q[node]/self.N[node]

        return max(self.children[node], key=score)

    def do_rollout(self, node):
        
        "make the tree one layer better"

        path = self._select(node)
        leaf = path[-1]
        self._expand(leaf)
        reward = self._simulate(leaf)
        self._backpropagate(path, reward)

    def _select(self, node):
        "find an unexplored descendent of `node`"
        path = []

        while True:
            path.append(node)

            if node not in self.children or node not in self.children[node]:
                #  node is either terminal or unexplored
                return path 

            unexplored = self.children[node] - self.children.keys()
            if unexplored:
                n = unexplored.pop()
                path.append(n)
                return path
            node = self._uct_select(node)

    def _expand(self, leaf):
        "Update the `children` dict with the children of `node`"
        if leaf in self.children:
            return # not a leaf already explored               
        self.children[leaf] = leaf.find_children()


