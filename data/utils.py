TEAMS = []


def get_data_from_row(row, data_stat: str):
    return row.find_all(
        "td", {"data-stat": data_stat}
    )[0].text


def get_team_ID(team_name: str):
    if team_name not in TEAMS:
        TEAMS.append(team_name)
    return TEAMS.index(team_name)


def dict_in_dict(d, k):
    if k not in d:
        d[k] = {}
    return d


def list_in_dict(d, k):
    if k not in d:
        d[k] = []
    return d


def get_players_for_game():
    pass
