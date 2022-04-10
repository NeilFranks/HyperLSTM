from calendar import c
import os

import pandas as pd

DATA_DIR = "data/seasons"
STANDARDIZE_DATA_CSV = "data/standardized_data.csv"


def run():
    # combine all CSVs into one massive dataframe
    combined_df = None

    for filename in os.listdir(DATA_DIR):
        df = pd.read_csv(
            os.path.join(DATA_DIR, filename),
            usecols=[
                "Year", "Month", "Day", "Home_ID", "Home_name",
                "Home_wins_last10", "Home_wins_VERSUS_last2",
                "Home_goals_lastGame", "Home_assists_lastGame",
                "Home_GA_startingGoalie", "Home_SA_startingGoalie",
                "Home_GA_allGoalies", "Home_SA_allGoalies", "Away_ID",
                "Away_name", "Away_wins_last10", "Away_wins_VERSUS_last2",
                "Away_goals_lastGame", "Away_assists_lastGame",
                "Away_GA_startingGoalie", "Away_SA_startingGoalie",
                "Away_GA_allGoalies", "Away_SA_allGoalies", "Home_Won"
            ]
        )

        if combined_df:
            combined_df.append(df)
        else:
            combined_df = df

    # standardize the data in each appropriate column
    COLUMNS_TO_NORMALIZE = [
        "Year", "Month", "Day",
        "Home_wins_last10", "Home_wins_VERSUS_last2",
        "Home_goals_lastGame", "Home_assists_lastGame",
        "Home_GA_startingGoalie", "Home_SA_startingGoalie",
        "Home_GA_allGoalies", "Home_SA_allGoalies",
        "Away_wins_last10", "Away_wins_VERSUS_last2",
        "Away_goals_lastGame", "Away_assists_lastGame",
        "Away_GA_startingGoalie", "Away_SA_startingGoalie",
        "Away_GA_allGoalies", "Away_SA_allGoalies"
    ]

    df_z_scaled = combined_df.copy()

    for column in df_z_scaled .columns:
        if column in COLUMNS_TO_NORMALIZE:
            df_z_scaled[column] = (
                df_z_scaled[column] - df_z_scaled[column].mean()
            ) / df_z_scaled[column].std()

    # save to CSV
    df_z_scaled.to_csv(STANDARDIZE_DATA_CSV)

if __name__ == "__main__":
    run()
