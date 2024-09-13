import torch
from torch import nn

import numpy as np
import chess
from tqdm import tqdm
import matplotlib as plt
from datetime import datetime
import pandas as pd
import os

from ChessEnv import ChessEnv
import config
import utils

## https://github.com/zjeffer/chess-deep-rl

class Trainer:
    def __init__(self, model: nn.Module, optimizer, device='cpu'):
        self.model = model
        self.batch_size = config.BATCH_SIZE
        self.lr = config.LEARNING_RATE

        if optimizer is None:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), self.lr)
        else:
            self.optimizer = optimizer

        self.device = device

    def sample_batch(self, data):
        if self.batch_size > len(data):
            return data
        else:
            np.random.shuffle(data)
            return data[:self.batch_size]

    def split_Xy(self, data):
        # board to input format (19x8x8)
        X = np.array([ChessEnv.state_to_input(i[0])[0] for i in data])
        # moves to output format (73x8x8)
        y_probs = []
        # values = winner
        y_value = []
        for position in data:
            # for every position in the batch, get the output probablity vector and value of the state
            board = chess.Board(position[0])
            moves = utils.moves_to_output_vector(position[1], board)

            y_probs.append(moves)
            y_value.append(position[2])
        
        X = np.array(X)

        return torch.Tensor(X), (torch.Tensor(y_probs).reshape(len(y_probs), config.OUTPUT_SHAPE[0]), torch.Tensor(y_value).view(len(y_value), config.OUTPUT_SHAPE[1]))

    def train_batch(self, X, y_probs, y_value):
        
        self.model.train()
        
        
        X = X.to(self.device)
        y_probs = y_probs.to(self.device)
        y_value = y_value.to(self.device)

        self.optimizer.zero_grad()
        logits, loss, policy_loss, value_loss = self.model(X, y=(y_probs, y_value))
        loss.backward()
        self.optimizer.step()

        return {"loss":loss, 'policy_head_loss':policy_loss, 'value_head_loss':value_loss}
    
    def train_all_data(self, data):
        """
        Train the model on all given data.
        """
        history = []
        np.random.shuffle(data)
        print("Splitting data into labels and target...")
        X, y = self.split_Xy(data)
        print("Training batches...")
        history = self.train_batch(X, y[0], y[1])

        return history

    def train_random_batches(self, data):
        """
        Train the model on batches of data

        X = the state of the board (a fen string)
        y = the search probs by MCTS (array of dicts), and the winner (-1, 0, 1)
        """
        history = []
        X, (y_probs, y_value) = self.split_Xy(data)
        for _ in tqdm(range(2*max(5, len(data) // self.batch_size))):
            indexes = np.random.choice(len(data), size=self.batch_size, replace=True)
            # only select X values with these indexes
            X_batch = X[indexes]
            y_probs_batch = y_probs[indexes]
            y_value_batch = y_value[indexes]
            
            losses = self.train_batch(X_batch, y_probs_batch, y_value_batch)
            history.append(losses)
        return history

    def plot_loss(self, history):
        df = pd.DataFrame(history)
        df[['loss', 'policy_head_loss', 'value_head_loss']] = df[['loss', 'policy_head_loss', 'value_head_loss']].apply(pd.to_numeric, errors='coerce')
        total_loss = df[['loss']].values
        policy_loss = df[['policy_head_loss']].values
        value_loss = df[['value_head_loss']].values
        plt.plot(total_loss, label='loss')
        plt.plot(policy_loss, label='policy_head_loss')
        plt.plot(value_loss, label='value_head_loss')
        plt.legend()
        plt.title(f"Loss over time\nLearning rate: {config.LEARNING_RATE}")
        plt.savefig(f"{config.LOSS_PLOTS_FOLDER}/loss-{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.png")

    def save_model(self):
        os.makedirs(config.MODEL_FOLDER, exist_ok=True)
        path = f"{config.MODEL_FOLDER}/model-{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.h5"
        torch.save(self.model, path)
        print(f"Model trained. Saved model to {path}")