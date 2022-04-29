import random

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


def pad_and_add_channel(t, length, axis=0, channel_axis=1, index=None):
    assert len(t.shape) > 1, 'Cannot pad a 1D tensor'
    assert axis != channel_axis
    # First, edit the tensor's shape to reflect padding
    edit_shape = list(t.shape)
    original = edit_shape[axis]
    edit_shape[axis] = length
    # Then, add a channel to the OTHER axis for masking
    if channel_axis is not None:
        edit_shape[channel_axis] += 1
    # Create the background tensor from edited shape
    background = torch.zeros(tuple(edit_shape))
    # Calculate where to place our tensor
    if index is None:
        index = (0,) * len(t.shape)  # Starting points for embed
    slices = [np.s_[i:i+j] for i, j in zip(index, t.shape)]
    background[slices] = t
    if channel_axis is not None:
        background[:original, -1] = torch.ones(original)
    return background


class HockeyDataset(Dataset):
    def __init__(self, file_name, features, sequence_length=10, pad_length=20, restrict_to_years: list = None):
        self.pad_length = pad_length

        self.X_COLUMNS = features
        self.Y_COLUMN = "Home_Won"

        self.data = pd.read_csv(file_name)

        if restrict_to_years:
            # restrict the data games which occured during the specified year
            self.data = self.data.loc[
                self.data['Year'].isin(restrict_to_years)
            ].reset_index()

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

        # algorithm to get sequences where NO output game EVER appears in input
        self.sequences = []
        output_game_indices = set([])

        for potential_output_game_index in range(len(self.data)):
            potential_output_game = self.data.iloc[potential_output_game_index]
            home_ID = potential_output_game["Home_ID"]
            away_ID = potential_output_game["Away_ID"]

            # check that each team has a valid length sequence preceding this game
            home_team_preceding_games = self.games_played_by_team[home_ID][
                :list(
                    self.games_played_by_team[home_ID]
                ).index(potential_output_game_index)
            ]
            away_team_preceding_games = self.games_played_by_team[away_ID][
                :list(
                    self.games_played_by_team[away_ID]
                ).index(potential_output_game_index)
            ]

            home_sequence = list(
                home_team_preceding_games[-(sequence_length-1):]
            )
            away_sequence = list(
                away_team_preceding_games[-(sequence_length-1):]
            )

            if len(home_sequence) >= sequence_length-1 and len(away_sequence) >= sequence_length-1:
                # can make a sequence, but check if any of the preceding games were already used as output games
                if not (
                    any(
                        input_game_index in output_game_indices for input_game_index in home_sequence
                    ) or any(
                        input_game_index in output_game_indices for input_game_index in away_sequence
                    )
                ):
                    # add these sequences
                    home_sequence.append(potential_output_game_index)
                    away_sequence.append(potential_output_game_index)

                    self.sequences.append(home_sequence)
                    self.sequences.append(away_sequence)

                    output_game_indices.add(potential_output_game_index)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        # get the sequence at the given index
        sequence = self.sequences[index]

        # turn every game in the sub-sequence into vectors
        x_sequence = None

        for game_index in sequence:
            # get the series at index
            series = self.data.iloc[game_index]

            # separate the row into "features" (x) and "outcome" (y)
            x = torch.tensor([])
            y = None

            for column_name in series.axes[0]:
                if column_name in self.X_COLUMNS:
                    if column_name == "Home_ID" or column_name == "Away_ID":
                        # turn the team IDs into one-hot vectors
                        x = torch.cat((
                            x,
                            torch.eye(self.NUM_OF_TEAMS)[
                                int(series[column_name])]
                        ))

                    elif column_name == self.Y_COLUMN:
                        # add this as a feature for every game EXCEPT THE LAST ONE
                        # predicting the outcome of the last game is all we care about
                        if game_index != sequence[-1]:
                            x = torch.cat((
                                x,
                                torch.tensor(
                                    [series[column_name]],
                                    dtype=torch.float64
                                )
                            ))
                        else:
                            # last game in the series; THIS is the y you want to predict
                            x = torch.cat((
                                x,
                                torch.tensor(
                                    # where either a 0 or 1 would go, just put a neutral 0.5
                                    [0.5],
                                    dtype=torch.float64
                                )
                            ))

                            y = torch.tensor(
                                # flip: 0 now means Home Team won. 1 means Home Team lost.
                                [1-series[column_name]],
                                dtype=torch.float64
                            )

                    else:
                        x = torch.cat((
                            x,
                            torch.tensor(
                                [series[column_name]],
                                dtype=torch.float64
                            )
                        ))

            # update x sequence
            if x_sequence is None:
                x_sequence = torch.unsqueeze(x, dim=0)
            else:
                x_sequence = torch.vstack(
                    (x_sequence, x)
                )

        return x_sequence, y
