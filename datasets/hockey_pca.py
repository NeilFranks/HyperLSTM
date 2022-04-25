import random

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

from .hockey import *

from sklearn.decomposition import PCA

class PCAHockeyDataset(Dataset):
    def __init__(self, file_name, features, pad_length=20, n_components=32):
        self.pca = PCA(n_components=n_components)
        self.pad_length=pad_length

        self.X_COLUMNS = features
        self.Y_COLUMN = "Home_Won"

        self.data = pd.read_csv(file_name)
        self.wins = self.data[self.Y_COLUMN]
        print(self.data)
        self.data.drop(columns=[self.Y_COLUMN], inplace=True)
        self.enc  = pd.get_dummies(self.data)
        self.as_numpy = self.enc.to_numpy(copy=False) # DON'T copy :)
        self.pca.fit(self.as_numpy)

        print('Fit PCA to hockey dataset, resulting in metrics:')
        print(self.pca.singular_values_)
        print(self.pca.explained_variance_ratio_)

        self.NUM_OF_TEAMS = 57  # 57 teams have played in the NHL

        # store the indices of games played by each team
        # effectively, we will have 57 very long sequences, each corresponding to a team's entire game history
        self.games_played_by_team = {
            team_ID: self.data.index[
                (self.data["Home_ID"] == team_ID) |
                (self.data["Away_ID"] == team_ID)
            ]
            for team_ID in range(self.NUM_OF_TEAMS)
        }

        # split these 57 sequences into shorter sub-sequences
        SUB_SEQUENCE_MIN_LENGTH = 10
        SUB_SEQUENCE_MAX_LENGTH = 20
        self.sub_sequences = []
        for team_ID in range(self.NUM_OF_TEAMS):
            sequence = self.games_played_by_team[team_ID]

            start_index = 0
            end_index = start_index + random.randrange(
                SUB_SEQUENCE_MIN_LENGTH, SUB_SEQUENCE_MAX_LENGTH
            )

            while end_index < len(sequence):
                self.sub_sequences.append(sequence[start_index:end_index])
                start_index = end_index
                end_index = start_index + random.randrange(
                    SUB_SEQUENCE_MIN_LENGTH, SUB_SEQUENCE_MAX_LENGTH
                )

            # whatever's left, call it a subsequence. May be smaller than you specified by SUB_SEQUENCE_MIN_LENGTH, but that'll just make a better dataset, right? :)
            self.sub_sequences.append(sequence[start_index:])

    def __len__(self):
        return len(self.sub_sequences)

    def __getitem__(self, index):
        # get the sub-sequence at the given index
        sub_sequence = self.sub_sequences[index]

        # turn every game in the sub-sequence into vectors
        x_sequence = None
        y_sequence = None

        for game_index in sub_sequence:
            # get the series at index
            series = self.enc.iloc[game_index]
            win    = self.wins.iloc[game_index]

            # separate the row into "features" (x) and "outcome" (y)
            x = torch.tensor([])
            y = torch.tensor([win], dtype=torch.float64)

            for column_name in series.axes[0]:
                x = torch.cat((
                    x,
                    torch.tensor(
                        [series[column_name]],
                        dtype=torch.float64
                    )
                ))

            # update x and y sequences
            if x_sequence is None:
                x_sequence = torch.unsqueeze(x, dim=0)
            else:
                x_sequence = torch.vstack(
                    (x_sequence, x)
                )

            if y_sequence is None:
                y_sequence = torch.unsqueeze(y, dim=0)
            else:
                y_sequence = torch.vstack(
                    (y_sequence, y)
                )

        x_sequence = torch.tensor(self.pca.transform(x_sequence))

        # Important! Pad sequences to a common length (20 in our case)
        x_sequence = pad_and_add_channel(
             x_sequence, self.pad_length, axis=0, channel_axis=1
             )

        y_sequence = pad_and_add_channel(
            y_sequence, self.pad_length, axis=0, channel_axis=None # Don't add channel here
            )
        return x_sequence, y_sequence
