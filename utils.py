import torch
from torch import nn

import chess
from mapper import Mapping
import numpy as np
from chess import Move

import time
import tqdm

def get_params(model):
    total_param = 0
    for param in model.parameters():
        total_param += np.prod(list(param.data.size()))
    return total_param

def get_size(model):
    size_model = 0
    for param in model.parameters():
        if param.data.is_floating_point():
            size_model += param.numel() * torch.finfo(param.data.dtype).bits
        else:
            size_model += param.numel() * torch.iinfo(param.data.dtype).bits
    return size_model, (size_model / 8e6)

def move_to_plane_index(move: str, board: chess.Board):
    """"
    Convert a move to a plane index and the row and column on the board
    """
    move: Move = Move.from_uci(move)
    # get start and end position
    from_square = move.from_square
    to_square = move.to_square
    # get piece
    piece: chess.Piece = board.piece_at(from_square)
    

    if piece is None:
            raise Exception(f"No piece at {from_square}")

    plane_index: int = None

    if move.promotion and move.promotion != chess.QUEEN:
        piece_type, direction = Mapping.get_underpromotion_move(
            move.promotion, from_square, to_square
        )
        plane_index = Mapping.mapper[piece_type][1 - direction]
    else:
        if piece.piece_type == chess.KNIGHT:
            # get direction
                direction = Mapping.get_knight_move(from_square, to_square)
                plane_index = Mapping.mapper[direction]
        else:
            # get direction of queen-type move
            direction, distance = Mapping.get_queenlike_move(
                from_square, to_square)
            plane_index = Mapping.mapper[direction][np.abs(distance)-1]
    row = from_square % 8
    col = 7 - (from_square // 8)
    return (plane_index, row, col)

def moves_to_output_vector(moves: dict, board: chess.Board) -> np.ndarray:
    """
    Convert a dictionary of moves to a vector of probabilities
    """
    vector = np.zeros((73, 8, 8), dtype=np.float32)
    
    # print(moves)

    for move, prob in moves.items():
        plane_index, row, col = move_to_plane_index(move, board)
        vector[plane_index, row, col] = prob

    return np.asarray(vector)

def time_function(func):
    """
    Decorator to time a function
    """
    def wrap_func(*args, **kwargs):
        t1 = time.time()
        result = func(*args, **kwargs)
        t2 = time.time()
        print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s')
        return result
    return wrap_func

    
def eval(model, data, device='cpu'):

    model.eval()
    error = 0

    for d in data:
        data = data.to(device)

        with torch.no_grad():
            logits, loss = model(data)
            # mean absolute error
            error += loss.item()

    return error / len(data)


def run_experiment(model, model_name, n_epochs, loss_function, optimizer, scheduler, train_loader, val_loader, test_loader):

    print(f"Running experiment for {model_name}, training on {len(train_loader.dataset)} samples for {n_epochs} epochs.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("\nModel architecture:")
    print(model)
    print(f'Total parameters: {get_params(model)}')
    model = model.to(device)

    print("\nStart training:")
    best_val_error = None
    perf_per_epoch = [] # Track Test/Val MAE vs. epoch (for plotting)


    t = time.time()

    for epoch in tqdm(range(1, n_epochs+1)):
        # Call LR scheduler at start of each epoch
        lr = scheduler.optimizer.param_groups[0]['lr']
        # Train model for one epoch, return avg. training loss
        # loss = train(model, train_loader, loss_function, optimizer, device)

        # Evaluate model on validation set
        val_error = eval(model, val_loader, loss_function, device)

        scheduler.step(val_error)


        if best_val_error is None or val_error <= best_val_error:
            # Evaluate model on test set if validation metric improves
            test_error = eval(model, test_loader, loss_function, device)
            best_val_error = val_error

        # if epoch % 10 == 0:
            # Print and track stats every 10 epochs
            # print(f'Epoch: {epoch:03d}, LR: {lr:5f}, Loss: {loss:.7f}, '
                #   f'Val Loss: {val_error:.7f}, Test Loss: {test_error:.7f}')

        # perf_per_epoch.append((loss, test_error, val_error, epoch, model_name))

    t = time.time() - t
    train_time = t/60
    print(f"\nDone! Training took {train_time:.2f} mins. Best validation Loss: {best_val_error:.7f}, corresponding test Loss: {test_error:.7f}.")

    return best_val_error, test_error, train_time, perf_per_epoch