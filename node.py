import chess

class Node:
    def __init__(self, parent=None):
        self.state = chess.Board()
        self.parent = parent
        self.action = ''
        self.children = set()
        self.v = 0
        self.N = 0
        self.n = 0
