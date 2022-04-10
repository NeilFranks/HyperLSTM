TEAMS = []


# general stuff


def dict_in_dict(d, k):
    if k not in d:
        d[k] = {}
    return d


def list_in_dict(d, k):
    if k not in d:
        d[k] = []
    return d

# HTML stuff


def get_data_from_row(row, data_stat: str):
    return row.find_all(
        "td", {"data-stat": data_stat}
    )[0].text


def get_player_id_from_row(row):
    player = row.find(
        "td", {"data-stat": "player"}
    )
    try:
        return player.next.attrs['href'].split("/")[-1].split(".")[0]
    except:
        if player.text == "Empty Net":
            return None
        else:
            print("interesting.... %s" % player.text)
            return player.text


def get_player_ids_from_table(table):
    rows = table.tbody.find_all("tr")
    player_ids = []
    for row in rows:
        player_id = get_player_id_from_row(row)
        # "Empty Net" is sometimes listed as a goalie, which returns an id of 'None'
        if player_id:
            player_ids.append(player_id)

    return player_ids

# team stuff


def get_team_ID(team_name: str):
    if team_name not in TEAMS:
        TEAMS.append(team_name)
    return TEAMS.index(team_name)


def player_exists(players_dict, player_id):
    if player_id not in players_dict:
        # initialize with one "null" game
        players_dict[player_id] = [{"goals": 0, "assists": 0}]
    return players_dict


def goalie_exists(goalies_dict, player_id):
    if player_id not in goalies_dict:
        # initialize with one "null" game
        goalies_dict[player_id] = [{"goals_against": 0, "shots_against": 0}]
    return goalies_dict


def goalie_is_on_team(goalies_on_team_dict, team_id, player_id):
    if team_id not in goalies_on_team_dict:
        goalies_on_team_dict[team_id] = player_id
    else:
        goalies_on_team_list = goalies_on_team_dict[team_id].split(";")
        if player_id not in goalies_on_team_list:
            goalies_on_team_list.append(player_id)
            goalies_on_team_dict[team_id] = ";".join(goalies_on_team_list)
    return goalies_on_team_dict
