"""
Problem: starting and restarting the script that webscraped the data (or having it blowup)
caused the TEAMS list to be wiped clean (because I was dumb and forgot to write that to a file).

As such, I need to go back through every CSV, give each team a unique ID, and edit the column in each CSV.
"""
import json
import os

import pandas as pd

DATA_DIR = "data/seasons"


def run():
    # combine all CSVs into one massive dataframe
    combined_df = pd.DataFrame()

    for filename in os.listdir(DATA_DIR):
        filepath = os.path.join(DATA_DIR, filename)
        df = pd.read_csv(
            os.path.join(DATA_DIR, filename)
        )

        if not combined_df.empty:
            combined_df = pd.concat([combined_df, df])
        else:
            combined_df = df

    # get all the team names (can be cheeky and just look in the HOME column)
    team_names = combined_df[" Home_name"].copy()
    team_names = list(team_names.drop_duplicates())

    # might as well fix whitespace errors
    team_names = [
        team_name.strip()
        for team_name in team_names
    ]

    # NOW, write the teams to a json file
    # note that the "ID" is actually the order they appeared in the NHL
    with open("data/team_names.json", 'w') as file:
        file.write(
            json.dumps(team_names)
        )

    # use team_names to set all the IDs straight
    for filename in os.listdir(DATA_DIR):
        filepath = os.path.join(DATA_DIR, filename)
        df = pd.read_csv(filepath)

        # while we're at it, fix whitespace errors
        df = df.apply(
            lambda x: x.str.strip() if x.dtype == "object" else x
        ).rename(
            columns=lambda x: x.strip()
        )

        # Now, fix team IDs
        df["Home_ID"] = pd.DataFrame(
            [
                team_names.index(team_name)
                for team_name in df["Home_name"]
            ]
        )

        df["Away_ID"] = pd.DataFrame(
            [
                team_names.index(team_name)
                for team_name in df["Away_name"]
            ]
        )

        # save CSV
        df.to_csv(filepath, index=False)


if __name__ == "__main__":
    run()
