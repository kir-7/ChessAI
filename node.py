import chess

class Node():
    """
    A representation of a single board state.
    MCTS works by constructing a tree of these Nodes.
    Could be e.g. a chess or checkers board state.
    """
    def find_children(self):
        "All possible successors of this board state"
        return set()

    def find_random_child(self):
        "Random successor of this board state (for more efficient simulation)"
        return None

    def is_terminal(self):
        "Returns True if the node has no children"
        return True

    def reward(self):
        "Assumes `self` is terminal node. 1=win, 0=loss, .5=tie, etc"
        return 0

    def __hash__(self):
        "Nodes must be hashable"
        return 123456789

    def __eq__(node1, node2):
        "Nodes must be comparable"
        return True
