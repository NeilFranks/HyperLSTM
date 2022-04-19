import random

import pandas as pd
import torch
from torch.utils.data import Dataset

random.seed(505)


class HockeyDataset(Dataset):
    def __init__(self, file_name):
        self.X_COLUMNS = [
            "Year", "Month", "Day",
            "Home_ID", "Home_wins_last10", "Home_wins_VERSUS_last2",
            "Home_goals_lastGame", "Home_assists_lastGame",
            "Home_GA_startingGoalie", "Home_SA_startingGoalie",
            "Home_GA_allGoalies", "Home_SA_allGoalies",
            "Away_ID", "Away_wins_last10", "Away_wins_VERSUS_last2",
            "Away_goals_lastGame", "Away_assists_lastGame",
            "Away_GA_startingGoalie", "Away_SA_startingGoalie",
            "Away_GA_allGoalies", "Away_SA_allGoalies"
        ]

        self.Y_COLUMN = "Home_Won"

        self.data = pd.read_csv(file_name)

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

                    else:
                        x = torch.cat((
                            x,
                            torch.tensor(
                                [series[column_name]],
                                dtype=torch.float64
                            )
                        ))

                elif column_name == self.Y_COLUMN:
                    y = torch.tensor([series[column_name]],
                                     dtype=torch.float64)

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

        return x_sequence, y_sequence
