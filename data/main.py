import requests

from bs4 import BeautifulSoup

from utils import get_data_from_row, get_team_ID, dict_in_dict, list_in_dict, get_players_for_game


URL = "https://www.hockey-reference.com"


def get_data_for_year(year):
    page = requests.get(
        f"https://www.hockey-reference.com/leagues/NHL_{year}_games.html"
    )

    soup = BeautifulSoup(page.content, 'html.parser')

    games_table = soup.find("table", {"id": "games"})
    games_table_data = games_table.tbody.find_all("tr")

    # keep track of game outcomes
    # key: team, value: list of binary win conditions (win: +1; tie/loss: 0)
    outcome_dict = {}
    outcome_by_team_dict = {}

    # keep track of players who played this season.
    # Will tally up the number of goals and assists they scored during each game
    players = {}

    # keep track of goalies who played this season.
    # Will tally up the number their goals allowed and shots faced
    goalies = {}

    data = {}
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
            Home team # of points for each player in the lineup during their most recent game
            Home team # of goals allowed by the starting goalie during their most recent game
            Home team # of shots faced by the starting goalie during their most recent game
            Home team # of goals allowed by all goalies during their most recent game (account for any backup goalies)
            Home team # of shots faced by all goalies during their most recent game (account for any backup goalies)

            Away team ID
            Away team name
            Away team # of wins over the last 10 games
            Away team # of wins over the home team in their last 10 games against each other (not always inferrible, since they could've tied!)
            Away team # of points for each player in the lineup during their most recent game
            Away team # of goals allowed by the starting goalie during their most recent game
            Away team # of shots faced by the starting goalie during their most recent game
            Away team # of goals allowed by all goalies during their most recent game (account for any backup goalies)
            Away team # of shots faced by all goalies during their most recent game (account for any backup goalies)

        Output:
            Outcome (+1 for home team wins. 0 for home team ties or loses)
        """
        heading = row.find_all("th")

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
        home_wins_VERSUS = sum(
            outcome_by_team_dict[home_team_id][away_team_id][-10:]
        ) if home_team_id in outcome_by_team_dict and away_team_id in outcome_by_team_dict[home_team_id] else 0
        away_wins_VERSUS = sum(
            outcome_by_team_dict[away_team_id][home_team_id][-10:]
        ) if away_team_id in outcome_by_team_dict and home_team_id in outcome_by_team_dict[away_team_id] else 0

        # get number of points scored for each player in the lineup during their most recent game

        # get starting goalie

        boxscore_url = heading[0].next

        # compute outcome: +1 is home team wins, 0 is home team loses or ties
        outcome = home_goals > away_goals

        # Record everything
        record(outcome, home_team_id, away_team_id)


def record(outcome, home_team_id, away_team_id):
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


if __name__ == "__main__":
    get_data_for_year(1918)
