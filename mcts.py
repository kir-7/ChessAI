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

    def _simulate(self, node):
        " returns the reward for random simulation (till completion) of `node`"

        invert_reward = 1
        while True:
            if node.is_terminal():
                reward = node.reward()
                return 1-reward if invert_reward else reward
            node = node.find_random_child()

            invert_reward ^= 1
    
    def _backpropogate(self, path, reward):
        "all the ancestors of leaf recieve the reward"

        for node in reversed(path):
            self.N[node] += 1
            self.Q[node] += reward
            
            reward = 1-reward # 1 for `me` is 0 for `enemy`
    
    def _uct_select(self, node):
        "select a child of node balancing exploitation and exploration"

        #  all children of node should already be expanded
        assert all(n in self.children for n in self.children[node])

        log_N_vertex = math.log(self.N[node])

        def uct(n):
            "Upper confidence bound for trees"
            return self.Q[n] / self.N[n] + self.exploration_weight * math.sqrt(
                log_N_vertex / self.N[n]
            )

        return max(self.children[node], key=uct)