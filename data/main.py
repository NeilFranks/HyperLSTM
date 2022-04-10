import os

from bs4 import BeautifulSoup
import requests

from utils import *


URL_BASE = "https://www.hockey-reference.com"

global outcome_dict
global outcome_by_team_dict
global players_dict
global goalies_dict
global goalies_on_team_dict


def get_data_for_year(year, CSV_NAME=None):
    global outcome_dict
    global outcome_by_team_dict
    global players_dict
    global goalies_dict
    global goalies_on_team_dict

    page = requests.get(
        f"{URL_BASE}/leagues/NHL_{year}_games.html"
    )

    soup = BeautifulSoup(page.content, 'html.parser')

    # TODO: playoff games as well
    games_table = soup.find("table", {"id": "games"})
    games_table_data = games_table.tbody.find_all("tr")

    # write to CSV
    if CSV_NAME:
        with open(CSV_NAME, 'w') as file:
            file.write("Year, Month, Day,")
            file.write("Home_ID, Home_name, Home_wins_last10, Home_wins_VERSUS_last10, Home_goals_lastGame, Home_assists_lastGame, Home_GA_startingGoalie, Home_SA_startingGoalie, Home_GA_allGoalies, Home_SA_allGoalies,")
            file.write("Away_ID, Away_name, Away_wins_last10, Away_wins_VERSUS_last10, Away_goals_lastGame, Away_assists_lastGame, Away_GA_startingGoalie, Away_SA_startingGoalie, Away_GA_allGoalies, Away_SA_allGoalies,")
            file.write("Home_Won")
            file.write("\n")

    for row in games_table_data:
        """
        CREATE A NEW ROW OF DATA, INCLUDING FEATURES AND OUTPUT

        Features:
            Year
            Month
            Day

            Home team ID
            Home team name
            Home team # of wins over the last 10 games
            Home team # of wins over the away team in their last 10 games against each other
            Home team # of goals for each player in the lineup during their most recent game
            Home team # of assists for each player in the lineup during their most recent game
            Home team # of goals allowed by the starting goalie during their most recent game
            Home team # of shots faced by the starting goalie during their most recent game
            Home team # of goals allowed by all goalies during their most recent game (account for any backup goalies)
            Home team # of shots faced by all goalies during their most recent game (account for any backup goalies)

            Away team ID
            Away team name
            Away team # of wins over the last 10 games
            Away team # of wins over the home team in their last 10 games against each other (not always inferrible, since they could've tied!)
            Away team # of goals for each player in the lineup during their most recent game
            Away team # of assists for each player in the lineup during their most recent game
            Away team # of goals allowed by the starting goalie during their most recent game
            Away team # of shots faced by the starting goalie during their most recent game
            Away team # of goals allowed by all goalies during their most recent game (account for any backup goalies)
            Away team # of shots faced by all goalies during their most recent game (account for any backup goalies)

        Output:
            Outcome (+1 for home team wins. 0 for home team ties or loses)
        """
        heading = row.find_all("th")

        # get boxscore
        boxscore_soup = BeautifulSoup(
            requests.get(
                URL_BASE+heading[0].next.attrs['href']
            ).content,
            'html.parser')

        skaters_tables = boxscore_soup.find_all(
            "table", {"id": lambda L: L and L.endswith('skaters')}
        )

        # When a game was forfeited, there is no boxscore. Skip it as if the game never happened
        if skaters_tables:

            # get date information
            date = heading[0].text.split("-")
            assert len(date) == 3
            year = int(date[0])
            month = int(date[1])
            day = int(date[2])

            # get name of teams and the goals the scored
            home_team_name = get_data_from_row(row, "home_team_name")
            away_team_name = get_data_from_row(row, "visitor_team_name")
            home_goals = int(get_data_from_row(row, "home_goals"))
            away_goals = int(get_data_from_row(row, "visitor_goals"))

            # get team IDs
            home_team_id = get_team_ID(home_team_name)
            away_team_id = get_team_ID(away_team_name)

            # get number of wins over the last 10 games
            home_wins_in_last_10 = sum(
                outcome_dict[home_team_id][-10:]
            ) if home_team_id in outcome_dict else 0
            away_wins_in_last_10 = sum(
                outcome_dict[away_team_id][-10:]
            ) if away_team_id in outcome_dict else 0

            # get number of wins versus each other over last 10 meetings
            home_wins_VERSUS_last_10 = sum(
                outcome_by_team_dict[home_team_id][away_team_id][-10:]
            ) if home_team_id in outcome_by_team_dict and away_team_id in outcome_by_team_dict[home_team_id] else 0
            away_wins_VERSUS_last_10 = sum(
                outcome_by_team_dict[away_team_id][home_team_id][-10:]
            ) if away_team_id in outcome_by_team_dict and home_team_id in outcome_by_team_dict[away_team_id] else 0

            goalies_tables = boxscore_soup.find_all(
                "table", {"id": lambda L: L and L.endswith('goalies')}
            )

            # get players who skated during the game
            home_players = get_player_ids_from_table(skaters_tables[1])
            away_players = get_player_ids_from_table(skaters_tables[0])

            # update players dict if it needs updating
            for player_id in home_players:
                players_dict = player_exists(players_dict, player_id)
            for player_id in away_players:
                players_dict = player_exists(players_dict, player_id)

            # get starting goalies for the game
            home_starting_goalie = get_player_ids_from_table(goalies_tables[1])[
                0]
            away_starting_goalie = get_player_ids_from_table(goalies_tables[0])[
                0]

            # update goalies on team
            goalies_on_team_dict = goalie_is_on_team(
                goalies_on_team_dict, home_team_id, home_starting_goalie
            )
            goalies_on_team_dict = goalie_is_on_team(
                goalies_on_team_dict, away_team_id, away_starting_goalie
            )

            # get all goalies on the team
            home_goalies = goalies_on_team_dict[home_team_id].split(";")
            away_goalies = goalies_on_team_dict[away_team_id].split(";")

            # update goalies dict if it needs updating
            for player_id in home_goalies:
                goalies_dict = goalie_exists(goalies_dict, player_id)
            for player_id in away_goalies:
                goalies_dict = goalie_exists(goalies_dict, player_id)

            # get number of goals scored for each player in the lineup during their most recent game
            home_players_goals_last_game = sum(
                [players_dict[player][-1]["goals"] for player in home_players]
            )
            away_players_goals_last_game = sum(
                [players_dict[player][-1]["goals"] for player in away_players]
            )

            # get number of assists scored for each player in the lineup during their most recent game
            home_players_assists_last_game = sum(
                [players_dict[player][-1]["assists"]
                    for player in home_players]
            )
            away_players_assists_last_game = sum(
                [players_dict[player][-1]["assists"]
                    for player in away_players]
            )

            # get number of goals allowed by starting goalie during their most recent game
            home_starting_goalie_goals_against_last_game = goalies_dict[
                home_starting_goalie][-1]["goals_against"]
            away_starting_goalie_goals_against_last_game = goalies_dict[
                away_starting_goalie][-1]["goals_against"]

            # get shots faced by starting goalie during most recent game
            home_starting_goalie_shots_against_last_game = goalies_dict[
                home_starting_goalie][-1]["shots_against"]
            away_starting_goalie_shots_against_last_game = goalies_dict[
                away_starting_goalie][-1]["shots_against"]

            # get number of goals allowed by all goalies during their most recent game
            home_all_goalies_goals_against_last_game = sum(
                [
                    goalies_dict[goalie][-1]["goals_against"]
                    for goalie in home_goalies
                ]
            )
            away_all_goalies_goals_against_last_game = sum(
                [
                    goalies_dict[goalie][-1]["goals_against"]
                    for goalie in away_goalies
                ]
            )

            # get shots faced by all goalies during most recent game
            home_all_goalies_shots_against_last_game = sum(
                [
                    goalies_dict[goalie][-1]["shots_against"]
                    for goalie in home_goalies
                ]
            )
            away_all_goalies_shots_against_last_game = sum(
                [
                    goalies_dict[goalie][-1]["shots_against"]
                    for goalie in away_goalies
                ]
            )

            # compute outcome: +1 is home team wins, 0 is home team loses or ties
            outcome = home_goals > away_goals

            # Write data to CSV
            if CSV_NAME:
                write_to_CSV(
                    CSV_NAME, year, month, day,
                    home_team_id, home_team_name, home_wins_in_last_10, home_wins_VERSUS_last_10, home_players_goals_last_game, home_players_assists_last_game, home_starting_goalie_goals_against_last_game, home_starting_goalie_shots_against_last_game, home_all_goalies_goals_against_last_game, home_all_goalies_shots_against_last_game,
                    away_team_id, away_team_name, away_wins_in_last_10, away_wins_VERSUS_last_10, away_players_goals_last_game, away_players_assists_last_game, away_starting_goalie_goals_against_last_game, away_starting_goalie_shots_against_last_game, away_all_goalies_goals_against_last_game, away_all_goalies_shots_against_last_game,
                    outcome
                )

            # Record everything in dicts
            outcome_dict, outcome_by_team_dict, players_dict, goalies_dict, goalies_on_team_dict = record(
                # dicts to update
                outcome_dict, outcome_by_team_dict, players_dict, goalies_dict, goalies_on_team_dict,
                # new information
                outcome, home_team_id, away_team_id, skaters_tables, goalies_tables
            )


def write_to_CSV(CSV_NAME, year, month, day, home_id, home_name, home_wins_last10, home_wins_versus_last10, home_goals_lastGame, home_assists_lastGame, home_GA_startingGoalie, home_SA_startingGoalie, home_GA_allGoalies, home_SA_allGoalies, away_id, away_name, away_wins_last10, away_wins_versus_last10, away_goals_lastGame, away_assists_lastGame, away_GA_startingGoalie, away_SA_startingGoalie, away_GA_allGoalies, away_SA_allGoalies, outcome):
    with open(CSV_NAME, 'a') as file:
        file.write(f"{year}, {month}, {day},")
        file.write(f"{home_id}, {home_name}, {home_wins_last10}, {home_wins_versus_last10}, {home_goals_lastGame}, {home_assists_lastGame}, {home_GA_startingGoalie}, {home_SA_startingGoalie}, {home_GA_allGoalies}, {home_SA_allGoalies},")
        file.write(f"{away_id}, {away_name}, {away_wins_last10}, {away_wins_versus_last10}, {away_goals_lastGame}, {away_assists_lastGame}, {away_GA_startingGoalie}, {away_SA_startingGoalie}, {away_GA_allGoalies}, {away_SA_allGoalies},")
        file.write(f"{outcome}")
        file.write("\n")


def record(outcome_dict, outcome_by_team_dict, players_dict, goalies_dict, goalies_on_team_dict, outcome, home_team_id, away_team_id, skaters_tables, goalies_tables):
    # update outcome dicts
    outcome_dict = list_in_dict(outcome_dict, home_team_id)
    outcome_dict = list_in_dict(outcome_dict, away_team_id)

    outcome_by_team_dict = dict_in_dict(outcome_by_team_dict, home_team_id)
    outcome_by_team_dict[home_team_id] = list_in_dict(
        outcome_by_team_dict[home_team_id],
        away_team_id
    )

    outcome_by_team_dict = dict_in_dict(outcome_by_team_dict, away_team_id)
    outcome_by_team_dict[away_team_id] = list_in_dict(
        outcome_by_team_dict[away_team_id],
        home_team_id
    )

    outcome_dict[home_team_id].append(outcome)
    outcome_dict[away_team_id].append(not outcome)
    outcome_by_team_dict[home_team_id][away_team_id].append(outcome)
    outcome_by_team_dict[away_team_id][home_team_id].append(not outcome)

    # get skaters and their points
    for skaters_table in skaters_tables:
        skaters_table_data = skaters_table.tbody.find_all("tr")
        for row in skaters_table_data:
            # player ID is taken from their URL
            player_id = get_player_id_from_row(row)

            # get points
            goals = int(get_data_from_row(row, "goals"))
            assists = int(get_data_from_row(row, "assists"))

            # update player history
            players_dict = list_in_dict(players_dict, player_id)
            players_dict[player_id].append(
                {"goals": goals, "assists": assists}
            )

    # get goalies and their stats
    for i in range(len(goalies_tables)):
        goalies_table = goalies_tables[i]
        goalies_table_data = goalies_table.tbody.find_all("tr")
        for row in goalies_table_data:
            # player ID is taken from their URL
            player_id = get_player_id_from_row(row)

            # get stats
            goals_against = int(get_data_from_row(row, "goals_against"))
            shots_against = get_data_from_row(row, "shots_against")
            if shots_against:
                shots_against = int(shots_against)
            else:
                # they didn't keep track of this stat back in the day; 0 it is!
                shots_against = 0

            # update player history
            goalies_dict = list_in_dict(goalies_dict, player_id)
            goalies_dict[player_id].append(
                {"goals_against": goals_against, "shots_against": shots_against}
            )

            # note that this goalie is on this team
            team_id = away_team_id if i == 0 else home_team_id
            goalies_on_team_dict = goalie_is_on_team(
                goalies_on_team_dict, team_id, player_id)

    return outcome_dict, outcome_by_team_dict, players_dict, goalies_dict, goalies_on_team_dict


if __name__ == "__main__":
    """
    Need to be clever about how to do season-after-season data without blowing up memory;

    Hacky solution will be to generate one season in-advance to act at the starting point for next season;
    re-initialize all dicts after, say, 4 seasons?
    """
    DIR = "data/seasons"
    if not os.path.isdir(DIR):
        os.mkdir(DIR)

    for start_year in range(1918, 2022, 4):
        # keep track of game outcomes
        # key: team, value: list of binary win conditions (win: +1; tie/loss: 0)
        outcome_dict = {}
        outcome_by_team_dict = {}

        # keep track of players who played this season.
        # Will tally up the number of goals and assists they scored during each game
        players_dict = {}

        # keep track of goalies who played this season.
        # Will tally up the number their goals allowed and shots faced
        goalies_dict = {}
        goalies_on_team_dict = {}  # key: team id, value: goalies separated by semicolon

        get_data_for_year(
            # won't write to a CSV; this is just to re-fill the dicts with games from this season
            start_year
        )
        for year in range(start_year+1, start_year+5):
            get_data_for_year(year, CSV_NAME=f"{DIR}/{year}.csv")
