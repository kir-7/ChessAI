from mcts import mcts, Node
import chess



if __name__ == '__main__':
    board = chess.Board()   
    white = 1
    over = 0 

    game = chess.pgn.Game()
    evaluation = []
    pgn = []
    sm = 0
    moves = 0

    while not board.is_game_over():
        all_moves = [board.san(i) for i in board.legal_moves]
        root = Node()
        root.state = board
        result = mcts(root, board.is_game_over(), white)
        print(result)
        board.push_san(result)
        pgn.append(result)
        white ^= 1
        moves += 1
    
    print(board)
    print(''.join(pgn))
    print()
    print(board.result)
