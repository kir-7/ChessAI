import chess
import random
import ChessEnv

class Node:
    """
    A representation of a single board state.
    MCTS works by constructing a tree of these Nodes.
    Could be e.g. a chess or checkers board state.
    
    """
    def __init__(self, board,  winner, terminal) -> None:
        self.board = board
        self.winner = winner
        self.terminal = terminal


    def find_random_child(self):
        "Random successor of this board state (for more efficient simulation)"

        if self.terminal :
            return None  # if game is finished no more moves can be made

        return self.make_move(self.board.san(random.choice(list(self.board.legal_moves))))
    

    def is_terminal(self):
        "Returns True if the node has no children"
        return self.terminal
    
    def reward(self, player):
        "Assumes `self` is terminal node. 1=win, 0=loss, .5=tie, etc"

        if not self.terminal:
            raise RuntimeError(f"reward called on nonterminal board {self}")
        
        if self.winner == player:
            return 1 ## you won the match
        elif self.winner != player:
            return 0  ## you lost the match
        else:
            return 0.5 ## it's a draw
        
        # The winner is neither True, False, nor None
        raise RuntimeError(f"board has unknown winner type {self.winner}")

    def make_move(self, move):

        board = chess.Board(self.board.fen())
        board.push_san(move)
        winner = self._find_winner(board)
        is_terminal = board.is_game_over()
        return Node(board, winner, is_terminal)
    
    def _find_winner(self, board):
        "Returns None if no winner, True if we win, else False"
    
        outcome = board.outcome()
        if outcome == None:
            return None
        else:
            if outcome.winner == None:
                return None
            return outcome.winner

    def __repr__(self):
        return str(self.board)

    def __hash__(self):
        "Nodes must be hashable"
        return hash(self.board.board_fen())

    def __eq__(self, other):
        "Nodes must be comparable"
        return self.board.board_fen() == other.board.board_fen()
    