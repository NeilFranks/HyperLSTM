from datasets.hockey         import HockeyDataset
from plot import *
import numpy as np
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
    year_corr = []
    years = []
    for year in range(1918, 2023):
        full_dataset = HockeyDataset(
            "data/standardized_data.csv",
            features,
            pad_length=20,
            restrict_to_years=[year-1918-1, year-1918+1]
        )
        relevant = [col for col in full_dataset.enc.columns if 'name' not in col]
        relevant_data = full_dataset.enc[relevant]
        correlations  = relevant_data.corr()
        predict_win   = correlations['Home_Won']
        print(predict_win.to_numpy().shape)
        year_corr.append(np.expand_dims(predict_win.to_numpy(), -1))
        print(relevant_data['Year'])
        print(year)
        print(predict_win)
        years.append(year)

    all_year_corr = np.concatenate(year_corr, axis=1)

    def formatter(fig):
        vals  = list(range(len(relevant)))
        fig.update_layout(
            yaxis=dict(tickmode='array',
                       tickvals=vals,
                       ticktext=relevant,),
            xaxis=dict(tickmode='array',
                       tickvals=[y-1918 for y in years],
                       ticktext=[str(y) for y in years],)
            )
    heatmap(all_year_corr.T,
            title=f'by-year correlations', horizontal=True, zmin=None, zmax=None,
            formatter=formatter)


if __name__ == '__main__':
    main(sys.argv[1:])
