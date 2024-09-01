import chess
import random


class Node:
    """
    A representation of a single board state.
    MCTS works by constructing a tree of these Nodes.
    Could be e.g. a chess or checkers board state.
    
    """
    def __init__(self, board, turn, winner, terminal) -> None:
        self.board = board
        self.turn = turn
        self.winner = winner
        self.terminal = terminal

    def find_children(self):
        "All possible successors of this board state"
        
        if self.terminal:
            return set() # cant generate children if game finished
        
        # Otherwise generate all possible next moves
        children = [self.make_move(self.board.san(move)) for move in self.board.legal_moves]
        return set(children)

    def find_random_child(self):
        "Random successor of this board state (for more efficient simulation)"

        if self.terminal :
            return None  # if game is finished no more moves can be made

        return self.make_move(self.board.san(random.choice(list(self.board.legal_moves))))
    

    def is_terminal(self):
        "Returns True if the node has no children"
        return self.terminal
    
    def reward(self):
        "Assumes `self` is terminal node. 1=win, 0=loss, .5=tie, etc"
        if not self.terminal:
            raise RuntimeError(f"reward called on nonterminal board {self}")
        if self.winner and self.turn:
            # It's your turn and you've already won. Should be impossible.
            raise RuntimeError(f"reward called on unreachable board {self}")
        if self.turn and (not self.winner):
            return 0  # Your opponent has just won. Bad.
        if (not self.turn) and self.winner:
            return 1
        if self.winner is None:
            return 0.5 # board is a tie        
        # The winner is neither True, False, nor None
        raise RuntimeError(f"board has unknown winner type {self.winner}")

    def make_move(self, move):

        board = chess.Board(self.board.fen())
        board.push_san(move)
        turn = self.turn ^ 1
        winner = self._find_winner(board)
        is_terminal = board.is_game_over()
        return Node(board, turn, winner, is_terminal)
    
    def _find_winner(self, board):
        "Returns None if no winner, True if we win, else False"
        if board.result()=='0-1':
            return False  ## it's your turn to move and it's a checkmate, you lost
        if board.result() == '1-0':
            return True   ## it's your opponent's turn and its checkmate, you won

        return None  ## it's either a draw or match in progress

    def __repr__(self):
        return str(self.board)

    def __hash__(self):
        "Nodes must be hashable"
        return hash(self.board.board_fen())

    def __eq__(self, other):
        "Nodes must be comparable"
        return self.board.board_fen() == other.board.board_fen()
    