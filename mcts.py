import chess
import chess.pgn
import chess.engine
import random
import math

class Node:
    def __init__(self, parent=None):
        self.state = chess.Board()
        self.parent = parent
        self.action = ''
        self.children = set()
        self.v = 0
        self.N = 0
        self.n = 0
        
def ucbl(curr_node):
    return curr_node.v + 2*math.sqrt((math.log(curr_node.N + math.e + 1e-6))/(curr_node.n + 1e-10))

def rollout(curr_node):
    if curr_node.state.is_game_over():
        board = curr_node.state
        if board.result() == '1-0':
            return (1, curr_node)
        elif board.result() == '0-1':
            return (-1, curr_node)
        else :
            return (0.5, curr_node)
    all_moves = [curr_node.state.san(i) for i in curr_node.state.legal_moves]

    for i in all_moves:
        tmp_state = chess.Board(curr_node.state.fen())
        tmp_state.push_san(i)
        child = Node()
        child.state = tmp_state
        child.parent = curr_node
        curr_node.children.add(child)

    a = random.choice(list(curr_node.children))
    return rollout(a)

def expand(curr_node, white):

    if len(curr_node.children) == 0:
        return curr_node

    max_ucb = min_ucb = ucbl(next(iter(curr_node.children)))
    max_child = min_child = next(iter(curr_node.children))
    
    for child in curr_node.children: 
        t = ucbl(child)
        if t > max_ucb:
            max_ucb = t
            max_child = child
        if t < min_ucb:
            min_ucb = t
            min_child = child
    if white:
        return expand(max_child, 0)
    else: 
        return expand(min_child, 1)

def rollback(curr_node, reward):

    curr_node.n += 1
    curr_node.v += reward

    while curr_node.parent != None:
        curr_node.N += 1
        curr_node = curr_node.parent
    
    return curr_node

def mcts(curr_node, over, white, iterations=5):
    if over:
        return -1
    all_moves = [curr_node.state.san(i) for i in curr_node.state.legal_moves]
    
    map_state_move = {}

    for i in all_moves:
        tmp_state = chess.Board(curr_node.state.fen())
        tmp_state.push_san(i)
        child = Node()
        child.state = tmp_state
        child.parent = curr_node
        curr_node.children.add(child)
        map_state_move[child] = i
    
    while iterations >  0:

        max_ucb = min_ucb = ucbl(next(iter(curr_node.children)))
        max_child = min_child = next(iter(curr_node.children))
        
        for child in curr_node.children: 
            t = ucbl(child)
            if t > max_ucb:
                max_ucb = t
                max_child = child
            if t < min_ucb:
                min_ucb = t
                min_child = child
        if white:
            ex_child = expand(max_child, 0)
        else:
            ex_child = expand(min_child, 1)
        reward, state = rollout(ex_child)
        curr_node = rollback(state, reward)

        iterations -= 1
    
    max_ucb = min_ucb = ucbl(next(iter(curr_node.children)))
    max_child = min_child = next(iter(curr_node.children))
    
    for child in curr_node.children: 
        t = ucbl(child)
        if t > max_ucb:
            max_ucb = t
            max_child = child
        if t < min_ucb:
            min_ucb = t
            min_child = child
    
    return map_state_move[max_child] if white else map_state_move[min_child] 
