from datasets.hockey         import HockeyDataset
from plot import *

import torch

features = [
    "Year", "Month", "Day",
    "Home_ID",
    "Home_wins_last10",
    "Home_wins_VERSUS_last2",
    "Home_goals_lastGame", "Home_assists_lastGame",
    "Home_GA_startingGoalie", "Home_SA_startingGoalie",
    "Home_GA_allGoalies", "Home_SA_allGoalies",
    "Away_ID",
    "Away_wins_last10",
    "Away_wins_VERSUS_last2",
    "Away_goals_lastGame", "Away_assists_lastGame",
    "Away_GA_startingGoalie", "Away_SA_startingGoalie",
    "Away_GA_allGoalies", "Away_SA_allGoalies"
]


def main(*args):
    full_dataset = HockeyDataset(
        "data/standardized_data.csv",
        features,
        pad_length=20,
        # only get games which occured from 1950 to 1960
        # restrict_to_years=[e-1918 for e in range(2010, 2023)]
        restrict_to_years=[e-1918 for e in range(1918, 2023)] # all years?
    )
    correlations = full_dataset.enc.corr()
    def formatter(fig):
        vals  = list(range(len(full_dataset.enc.columns)))
        names = full_dataset.enc.columns
        fig.update_layout(
            yaxis=dict(tickmode='array',
                       tickvals=vals,
                       ticktext=names,),
            xaxis=dict(tickmode='array',
                       tickvals=vals,
                       ticktext=names,)
            )
    heatmap(correlations.to_numpy(),
            title=f'correlations', horizontal=True, zmin=None, zmax=None,
            formatter=formatter)
    # full_dataset = MinimalHockeyDataset("data/standardized_data.csv")

    # l = len(full_dataset)
    # print('Investigating hockey dataset..')
    # print(l)
    # for i, (x, y) in enumerate(full_dataset):
    #     if i > 10:
    #         break
    #     print(i, x.shape, y.shape)
    1/0


if __name__ == '__main__':
    main(sys.argv[1:])
