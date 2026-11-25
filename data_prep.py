"""
Programmer: Felix Cohen, Connor Whyte
Class: CPSC 322, Fall 2026
Final Project
Date: 11/24/2026

Description: Cleans the raw FPL Core Insights data (olbauday/FPL-Core-Insights,
    snapshot in Inputs/data) into five tidy CSV files that are ready for EDA:

        teams.csv              one row per team per season
        matches.csv            one row per finished Premier League match
        player_matches.csv     one row per player per match he actually played
        player_seasons.csv     one row per outfield player per season (position classifier)
        match_features.csv     one row per match, pre-match form only (home-win classifier)

    Every function is written to be run in order (see run_all() at the bottom,
    or data_cleaning.ipynb, which walks through each step). 

    We use our own MyPyTable class and plain Python (no pandas) so the cleaned
    data comes out in the same format our classifiers from class expect.
"""
import os
from datetime import datetime

from mysklearn.mypytable import MyPyTable
from mysklearn import myutils

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
RAW_DATA_DIR = os.path.join("Inputs", "data")
CLEAN_DATA_DIR = "clean_data"
SEASONS = ["2024-2025", "2025-2026", "2026-2027"]
NUM_GAMEWEEKS = 38
OTHER_COMPETITIONS = ["Champions League", "Europa League", "Conference League", "EFL Cup"]

# 2024-25 uses an older folder layout: one file per table for the whole season,
# instead of one file per gameweek. This maps each file name to its folder.
OLD_LAYOUT_FOLDERS = {"matches.csv": "matches", "playermatchstats.csv": "playermatchstats",
                      "players.csv": "players", "teams.csv": "teams"}

# Team-level match statistics we keep (each one exists as home_<stat> and away_<stat>).
# We skip columns that are blank for every match (e.g. distance covered) and
# percentage columns, which can be recomputed from counts if needed.
MATCH_STATS = ["possession", "expected_goals_xg", "non_penalty_xg", "xg_on_target_xgot",
               "total_shots", "shots_on_target", "big_chances", "touches_in_opposition_box",
               "accurate_passes", "accurate_long_balls", "accurate_crosses", "corners",
               "tackles_won", "interceptions", "blocks", "clearances", "keeper_saves",
               "duels_won", "aerial_duels_won", "successful_dribbles", "fouls_committed",
               "yellow_cards", "red_cards", "offsides"]

# Player-match columns that are identifiers or that we deliberately drop.
#   * "_percent" columns: a 0% from 0 attempts looks the same as 0% from 10 attempts,
#     so they are misleading. Counts (e.g. accurate_crosses) are kept instead.
#   * start_min / finish_min: the data source's README warns these are unreliable.
#     minutes_played gives actual minutes.
PLAYER_ID_COLUMNS = ["player_id", "match_id", "minutes_played", "file_gameweek"]
PLAYER_DROPPED_COLUMNS = ["start_min", "finish_min"]

# Stats that only make sense for goalkeepers (or describe the team, not the
# player). They stay in player_matches.csv but are left out of the position table.
GOALKEEPER_OR_TEAM_STATS = ["saves", "goals_conceded", "xgot_faced", "goals_prevented",
                            "sweeper_actions", "gk_accurate_passes", "gk_accurate_long_balls",
                            "high_claim", "saves_inside_box", "team_goals_conceded"]

# Physical tracking stats only exist for 2026-27, so they can't be compared
# across seasons. Kept in player_matches.csv, left out of the position table.
TRACKING_ONLY_STATS = ["top_speed", "distance_covered", "walking_distance", "running_distance",
                       "sprinting_distance", "number_of_sprints"]

# Seasons where the source writes blanks instead of zeros (see fill_blank_zeros).
BLANK_MEANS_ZERO_SEASONS = ["2026-2027"]

# FPL and the lineup files use different words for the same positions.
POSITION_LABELS = {"Goalkeeper": "GK", "Defender": "DEF", "Midfielder": "MID", "Forward": "FWD",
                   "G": "GK", "D": "DEF", "M": "MID", "F": "FWD"}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def load_csv(path):
    """Loads a CSV into a MyPyTable and marks blank cells as "NA".

    MyPyTable (from class) uses the string "NA" for missing values, but the
    raw files use empty cells, so we convert them here once.
    """
    table = MyPyTable().load_from_file(path)
    for row in table.data:
        for j, value in enumerate(row):
            if value == "":
                row[j] = "NA"
    return table


def stack_tables(tables):
    """Stacks tables on top of each other, lining columns up by name.

    Different seasons (and sometimes different gameweeks) have slightly
    different columns. Any column missing from one table is filled with "NA"
    for that table's rows, so nothing is lost and nothing is misaligned.
    """
    column_names = []
    for table in tables:
        for name in table.column_names:
            if name not in column_names:
                column_names.append(name)
    data = []
    for table in tables:
        positions = [table.column_names.index(name) if name in table.column_names else None
                     for name in column_names]
        for row in table.data:
            data.append([row[p] if p is not None else "NA" for p in positions])
    return MyPyTable(column_names, data)


def load_season_file(season, filename, competition="Premier League"):
    """Loads one kind of file (e.g. "matches.csv") for a whole season.

    For 2025-26 onward the data is split into one folder per gameweek, so we
    load every gameweek's copy and stack them, recording which gameweek folder
    each row came from in a "file_gameweek" column.
    """
    if season == "2024-2025":
        if competition != "Premier League":
            return MyPyTable([], [])  # 2024-25 has no cup or European data
        return load_csv(os.path.join(RAW_DATA_DIR, season, OLD_LAYOUT_FOLDERS[filename], filename))

    competition_dir = os.path.join(RAW_DATA_DIR, season, "By Tournament", competition)
    if not os.path.isdir(competition_dir):
        return MyPyTable([], [])
    tables = []
    for folder in sorted(os.listdir(competition_dir)):
        path = os.path.join(competition_dir, folder, filename)
        if folder.startswith("GW") and os.path.exists(path):
            table = load_csv(path)
            gameweek = int(folder[2:])
            table.column_names.append("file_gameweek")
            for row in table.data:
                row.append(gameweek)
            tables.append(table)
    return stack_tables(tables) if tables else MyPyTable([], [])


def row_as_dict(table, row):
    """Pairs a row's values with its table's column names (makes code easier to read)."""
    return dict(zip(table.column_names, row))


def is_true(value):
    """True/False columns are written as "True" in some seasons and "TRUE" in others."""
    return str(value).lower() == "true"


def to_int(value):
    """Converts IDs like 36.0 (how the raw files store them) to 36. Keeps "NA" as is."""
    if value == "NA":
        return "NA"
    return int(float(value))


def normalize_kickoff(text):
    """Puts every kickoff time into one format: "YYYY-MM-DD HH:MM".

    The raw files mix three styles ("2024-10-19 16:30:00",
    "2025-11-29T15:00:00+00:00" and "2026-11-21T15:00:00"). We only use kickoff
    times to put matches in order and count days between them, so the small
    time-zone differences between styles don't matter.
    """
    text = str(text).replace("T", " ").split("+", maxsplit=1)[0].rstrip("Z")
    return datetime.strptime(text[:16], "%Y-%m-%d %H:%M").strftime("%Y-%m-%d %H:%M")


def days_between(earlier_kickoff, later_kickoff):
    """Number of days (as a decimal) between two normalized kickoff times."""
    fmt = "%Y-%m-%d %H:%M"
    difference = datetime.strptime(later_kickoff, fmt) - datetime.strptime(earlier_kickoff, fmt)
    return difference.total_seconds() / 86400


def save_table(table, filename):
    """Saves a cleaned table into the clean_data folder."""
    os.makedirs(CLEAN_DATA_DIR, exist_ok=True)
    table.save_to_file(os.path.join(CLEAN_DATA_DIR, filename))


# ---------------------------------------------------------------------------
# Step 1: teams
# ---------------------------------------------------------------------------
def build_teams():
    """One row per team per season: season, team_code, team_name, short_name.

    We keep only names and codes. The raw teams files also have "strength" and
    "elo" columns, but those are a snapshot from the day the data was
    downloaded (i.e. they already know how the season went), so using them as
    features would leak future information.
    """
    data = []
    for season in SEASONS:
        if season == "2024-2025":
            path = os.path.join(RAW_DATA_DIR, season, "teams", "teams.csv")
        else:
            path = os.path.join(RAW_DATA_DIR, season, "teams.csv")
        raw = load_csv(path)
        for row in raw.data:
            team = row_as_dict(raw, row)
            data.append([season, to_int(team["code"]), team["name"], team["short_name"]])
    return MyPyTable(["season", "team_code", "team_name", "short_name"], data)


# ---------------------------------------------------------------------------
# Step 2: matches
# ---------------------------------------------------------------------------
def build_matches(teams):
    """One row per finished Premier League match, sorted by kickoff time.

    Cleaning steps:
      1. Load only the "Premier League" folders. The "By Gameweek" folders also
         contain cup and European matches, which would mix competitions.
      2. Keep only finished matches (2026-27 also lists future fixtures).
      3. Remove duplicate match_ids (a postponed match could appear in two
         gameweek folders). We check with MyPyTable.find_duplicates from class.
      4. Turn team codes like 36.0 into whole numbers and add team names.
      5. Add the labels: result (H/D/A) and home_win (yes/no).
      6. Add our own pre-match Elo ratings (see compute_elo_ratings).
    """
    team_names = {(row[0], row[1]): row[2] for row in teams.data}
    columns = ["season", "gameweek", "match_id", "kickoff", "home_code", "home_name",
               "away_code", "away_name", "home_goals", "away_goals", "result", "home_win",
               "clubelo_home", "clubelo_away"]
    for stat in MATCH_STATS:
        columns += ["home_" + stat, "away_" + stat]

    data = []
    for season in SEASONS:
        raw = load_season_file(season, "matches.csv")
        for row in raw.data:
            match = row_as_dict(raw, row)
            if not is_true(match["finished"]):
                continue
            home_code, away_code = to_int(match["home_team"]), to_int(match["away_team"])
            home_goals, away_goals = to_int(match["home_score"]), to_int(match["away_score"])
            if home_goals > away_goals:
                result = "H"
            elif home_goals < away_goals:
                result = "A"
            else:
                result = "D"
            new_row = [season, to_int(match["gameweek"]), match["match_id"],
                       normalize_kickoff(match["kickoff_time"]),
                       home_code, team_names.get((season, home_code), "NA"),
                       away_code, team_names.get((season, away_code), "NA"),
                       home_goals, away_goals, result, "yes" if result == "H" else "no",
                       match.get("home_team_elo", "NA"), match.get("away_team_elo", "NA")]
            for stat in MATCH_STATS:
                new_row += [match.get("home_" + stat, "NA"), match.get("away_" + stat, "NA")]
            data.append(new_row)

    matches = MyPyTable(columns, data)
    matches.drop_rows(matches.find_duplicates(["match_id"]))
    matches.data.sort(key=lambda r: r[columns.index("kickoff")])
    compute_elo_ratings(matches)
    return matches


def goal_difference_multiplier(goal_difference):
    """How much more a big win should move the ratings than a narrow one.

    Uses the World Football Elo rule: a 1-goal game counts once, a 2-goal win
    counts 1.5 times, and wins by 3+ count (11 + margin) / 8 times.
    """
    margin = abs(goal_difference)
    if margin <= 1:
        return 1.0
    if margin == 2:
        return 1.5
    return (11 + margin) / 8


def compute_elo_ratings(matches, k_factor=20, home_advantage=65):
    """Adds elo_home and elo_away: each team's strength rating BEFORE kickoff.

    Why we compute our own: the raw data has ClubElo ratings (we checked they are
    pre-match, so they don't leak the result), but they are blank for 93 of the
    110 finished 2026-27 matches and for all of 2024-25 gameweek 38. A feature
    that is missing for most of a season is not usable, so we rebuild the
    ratings ourselves from match results using the standard Elo method:

      * Every team starts at the first ClubElo value the data has for it, so our
        ratings start on the same scale as ClubElo. (Promoted teams therefore
        start at a rating that reflects their season in the Championship.)
      * Before each match we work out how likely the home team was to win
        (expected score), giving the home team a bonus of `home_advantage` points.
      * After the match, both teams' ratings move by
        k_factor * goal_difference_multiplier * (actual score - expected score),
        where actual score is 1 for a win, 0.5 for a draw and 0 for a loss.
        The winner gains exactly what the loser drops.

    We store each rating before it is updated, so a match's own result never
    affects its own row. Unlike ClubElo, our ratings only use Premier League
    matches, so cup and European results don't move them.
    """
    index = {name: matches.column_names.index(name) for name in
             ["home_code", "away_code", "home_goals", "away_goals", "clubelo_home", "clubelo_away"]}

    # Starting rating for each team: the first ClubElo value we see for it.
    starting_ratings = {}
    for row in matches.data:
        for code_col, elo_col in [("home_code", "clubelo_home"), ("away_code", "clubelo_away")]:
            code, clubelo = row[index[code_col]], row[index[elo_col]]
            if code not in starting_ratings and clubelo != "NA":
                starting_ratings[code] = clubelo
    fallback_rating = sum(starting_ratings.values()) / len(starting_ratings)

    ratings = {}
    matches.column_names += ["elo_home", "elo_away"]
    for row in matches.data:
        home, away = row[index["home_code"]], row[index["away_code"]]
        home_rating = ratings.get(home, starting_ratings.get(home, fallback_rating))
        away_rating = ratings.get(away, starting_ratings.get(away, fallback_rating))
        row += [round(home_rating, 1), round(away_rating, 1)]

        expected_home = 1 / (1 + 10 ** ((away_rating - home_rating - home_advantage) / 400))
        goal_difference = row[index["home_goals"]] - row[index["away_goals"]]
        if goal_difference > 0:
            actual_home = 1.0
        elif goal_difference < 0:
            actual_home = 0.0
        else:
            actual_home = 0.5
        change = k_factor * goal_difference_multiplier(goal_difference) * (actual_home - expected_home)
        ratings[home] = home_rating + change
        ratings[away] = away_rating - change


# ---------------------------------------------------------------------------
# Step 3: player matches
# ---------------------------------------------------------------------------
def build_player_matches(matches, fill_zeros=True):
    """One row per player per Premier League match in which he actually played.

    Cleaning steps:
      1. Stack every season's playermatchstats files and keep only rows whose
         match is in our cleaned matches table (finished Premier League games).
      2. Drop rows with 0 minutes. These are unused substitutes (1,262 rows in
         2025-26 and 995 so far in 2026-27). They didn't play, so all their
         stats are empty and would only drag averages toward zero.
      3. Remove duplicate (player, match) rows, using find_duplicates from class.
      4. Attach who the player is and which team he played for (see
         attach_player_info). The raw stats file has no team column at all.
      5. Use player_code, not player_id, as the player's identity. We checked:
         of 534 players in both 2024-25 and 2025-26, only 1 kept the same
         player_id, because FPL renumbers players every season. player_code is
         the same across seasons, which we need to keep a player's seasons
         together when splitting into training and test sets.
      6. Fix stats where a blank really means zero (see fill_blank_zeros).
    """
    match_lookup = {}
    for row in matches.data:
        match = row_as_dict(matches, row)
        match_lookup[match["match_id"]] = match

    tables = []
    for season in SEASONS:
        raw = load_season_file(season, "playermatchstats.csv")
        tables.append(attach_player_info(season, raw, match_lookup))
    player_matches = stack_tables(tables)
    player_matches.drop_rows(player_matches.find_duplicates(["player_code", "match_id"]))
    resolve_unmatched_teams(player_matches)
    if fill_zeros:
        fill_blank_zeros(player_matches)
    drop_empty_columns(player_matches)
    player_matches.data.sort(key=lambda r: (r[3], r[2]))  # by kickoff, then match_id
    return player_matches


def attach_player_info(season, raw, match_lookup):  # pylint: disable=too-many-branches
    """Adds identity, team, opponent and position columns to one season's stats.

    Where the player's team comes from (best source first):
      * 2025-26: lineups.csv, which says exactly which side each player was on.
      * 2025-26 and 2026-27: the players.csv inside each gameweek folder. It is a
        snapshot of every player's club AT THAT GAMEWEEK, so a player who
        transferred in January is credited to the right club before and after.
        We checked it agrees with lineups.csv on every 2025-26 row.
      * 2024-25: only one players.csv for the whole season, which shows each
        player's LAST club. For the 82 rows where that club wasn't even in the
        match (players who moved mid-season), resolve_unmatched_teams fixes it.

    Positions:
      * fpl_position: FPL's official label (our class label for the position task).
      * lineup_position: where the player was listed in that day's lineup
        (2025-26 only). This is NOT a feature; we use it afterwards to look for
        players FPL labels as one position who actually play another.
    """
    season_players = {}
    gameweek_players = {}
    if season == "2024-2025":
        players = load_season_file(season, "players.csv")
        for row in players.data:
            player = row_as_dict(players, row)
            season_players[to_int(player["player_id"])] = player
    else:
        players = load_season_file(season, "players.csv")
        for row in players.data:
            player = row_as_dict(players, row)
            gameweek_players[(to_int(player["player_id"]), player["file_gameweek"])] = player

    lineups = {}
    if season != "2024-2025":
        lineup_table = load_season_file(season, "lineups.csv")
        for row in lineup_table.data:
            entry = row_as_dict(lineup_table, row)
            lineups[(to_int(entry["player_id"]), entry["match_id"])] = entry

    stat_columns = [name for name in raw.column_names if name not in PLAYER_ID_COLUMNS
                    and name not in PLAYER_DROPPED_COLUMNS and not name.endswith("_percent")]
    columns = ["season", "gameweek", "match_id", "kickoff", "player_code", "player_id", "player_name",
               "team_code", "opponent_code", "is_home", "fpl_position", "lineup_position", "started",
               "minutes"] + stat_columns

    data = []
    for row in raw.data:
        stats = row_as_dict(raw, row)
        match = match_lookup.get(stats["match_id"])
        if match is None or stats["minutes_played"] == "NA" or stats["minutes_played"] <= 0:
            continue
        player_id = to_int(stats["player_id"])
        if season == "2024-2025":
            player = season_players.get(player_id)
        else:
            player = gameweek_players.get((player_id, match["gameweek"]))
        if player is None:
            continue  # never happened in our checks, but don't crash if it does
        lineup = lineups.get((player_id, stats["match_id"]))
        team_code = to_int(lineup["team_code"]) if lineup else to_int(player["team_code"])

        if team_code == match["home_code"]:
            opponent_code, is_home = match["away_code"], "yes"
        elif team_code == match["away_code"]:
            opponent_code, is_home = match["home_code"], "no"
        else:
            opponent_code, is_home = "NA", "NA"  # fixed later by resolve_unmatched_teams

        data.append([season, match["gameweek"], stats["match_id"], match["kickoff"],
                     to_int(player["player_code"]), player_id, player["web_name"],
                     team_code, opponent_code, is_home,
                     POSITION_LABELS.get(player["position"], "Unknown"),
                     POSITION_LABELS.get(lineup["position"], "NA") if lineup else "NA",
                     ("yes" if is_true(lineup["is_starting"]) else "no") if lineup else "NA",
                     stats["minutes_played"]] + [stats[name] for name in stat_columns])
    return MyPyTable(columns, data)


def drop_empty_columns(table):
    """Removes columns that are "NA" in every row (e.g. corners, which the source
    lists for 2026-27 but never fills in). An empty column can't help any analysis.

    Returns:
        list of str: names of the removed columns
    """
    empty = [j for j in range(len(table.column_names))
             if all(row[j] == "NA" for row in table.data)]
    removed = [table.column_names[j] for j in empty]
    table.column_names = [name for j, name in enumerate(table.column_names) if j not in empty]
    table.data = [[value for j, value in enumerate(row) if j not in empty] for row in table.data]
    return removed


def resolve_unmatched_teams(player_matches):
    """Fixes rows where the player's listed club didn't play in that match.

    This only happens in 2024-25, where the players file shows each player's
    last club of the season. If a player moved in January, his early-season
    matches are for a club the file doesn't mention.

    Our fix: each of those matches was between two clubs, and one of them was
    his old club. His old club shows up in every one of those matches, while
    his opponents change from week to week. So we count how often each club
    appears across his unmatched matches and pick the most common one. If
    there's a tie (e.g. only one unmatched match), we can't tell, so we drop
    those rows.
    """
    col = {name: player_matches.column_names.index(name) for name in
           ["player_code", "season", "match_id", "team_code", "opponent_code", "is_home"]}
    match_teams = {}
    for row in player_matches.data:
        if row[col["is_home"]] != "NA":
            # In a matched row we know both sides of that match.
            key = row[col["match_id"]]
            home = row[col["team_code"]] if row[col["is_home"]] == "yes" else row[col["opponent_code"]]
            away = row[col["opponent_code"]] if row[col["is_home"]] == "yes" else row[col["team_code"]]
            match_teams[key] = (home, away)

    unmatched = {}
    for i, row in enumerate(player_matches.data):
        if row[col["is_home"]] == "NA":
            unmatched.setdefault((row[col["player_code"]], row[col["season"]]), []).append(i)

    rows_to_drop = []
    for indexes in unmatched.values():
        club_counts = {}
        for i in indexes:
            for club in match_teams[player_matches.data[i][col["match_id"]]]:
                club_counts[club] = club_counts.get(club, 0) + 1
        best = max(club_counts.values())
        best_clubs = [club for club, count in club_counts.items() if count == best]
        if len(best_clubs) != 1:
            rows_to_drop.extend(indexes)
            continue
        for i in indexes:
            row = player_matches.data[i]
            home, away = match_teams[row[col["match_id"]]]
            row[col["team_code"]] = best_clubs[0]
            row[col["is_home"]] = "yes" if best_clubs[0] == home else "no"
            row[col["opponent_code"]] = away if best_clubs[0] == home else home
    player_matches.drop_rows(rows_to_drop)
    return len(rows_to_drop)


def fill_blank_zeros(player_matches):
    """Turns blanks into 0 in the season where the source stopped writing zeros.

    In 2026-27, the data source leaves a cell blank instead of writing 0. We
    checked this against the data:
      * total_shots is blank for 1,536 outfield players who played, and the
        column has no explicit 0 anywhere. Those players simply had no shots.
      * shots_on_target is blank in exactly those same 1,536 rows (never when
        total_shots is above 0). A player with no shots has 0 shots on target.
      * accurate_crosses has real zeros (crossed, but none landed) AND blanks
        (didn't cross at all). Both mean 0 accurate crosses.
    So for 2026-27 outfield players, a blank in a stat column means 0.

    Exceptions, which stay "NA":
      * Columns that are blank for the WHOLE season (e.g. tackles,
        defensive_contributions in 2026-27). Those stats weren't recorded at all,
        and filling them with 0 would make every player look like he never tackled.
      * Goalkeepers in 2026-27. Their passing is stored in the gk_ columns and
        their outfield columns are left blank, so a blank there doesn't mean 0.
        (Goalkeepers aren't in either of our classification tasks anyway.)
      * Earlier seasons. There, blanks are rare and appear in rows where a whole
        group of detailed stats is missing (e.g. 293 rows from late 2025-26), so
        they really are unknown.

    Returns:
        list of str: the columns that were filled
    """
    col = player_matches.column_names.index
    first_stat_index = col("minutes") + 1
    rows = [row for row in player_matches.data
            if row[col("season")] in BLANK_MEANS_ZERO_SEASONS and row[col("fpl_position")] != "GK"]
    filled_columns = []
    for j in range(first_stat_index, len(player_matches.column_names)):
        values = [row[j] for row in rows]
        if "NA" in values and not all(value == "NA" for value in values):
            for row in rows:
                if row[j] == "NA":
                    row[j] = 0
            filled_columns.append(player_matches.column_names[j])
    return filled_columns


def check_blank_zero_evidence(player_matches, season="2026-2027"):
    """Counts how shots_on_target blanks line up with total_shots (backs up fill_blank_zeros).

    Must be run on the player-match table BEFORE fill_blank_zeros.

    Returns:
        dict: counts of blank shots_on_target where total_shots is blank vs. where it is above 0
    """
    col = player_matches.column_names.index
    counts = {"total_shots blank": 0, "total_shots above 0": 0, "explicit 0 in total_shots": 0}
    for row in player_matches.data:
        if row[col("season")] != season or row[col("fpl_position")] == "GK":
            continue
        if row[col("total_shots")] == 0:
            counts["explicit 0 in total_shots"] += 1
        if row[col("shots_on_target")] == "NA":
            key = "total_shots blank" if row[col("total_shots")] == "NA" else "total_shots above 0"
            counts[key] += 1
    return counts


# ---------------------------------------------------------------------------
# Step 4: player seasons (position classifier)
# ---------------------------------------------------------------------------
def build_player_seasons(player_matches, min_minutes=450):
    """One row per outfield player per season, with per-90-minute stats.

    Cleaning steps:
      1. Leave out goalkeepers (not one of our classes) and "Unknown" positions.
      2. Combine all of a player's matches in a season into one row. One row per
         player-season (instead of per match) smooths out single odd games, such
         as a defender who happened to score twice.
      3. Convert totals into "per 90 minutes" rates, so a player who played
         2,000 minutes can be compared fairly with one who played 700.
      4. Each per-90 rate only uses the minutes from matches where that stat was
         recorded. Otherwise a stat that is missing for part of a season would
         look artificially low.
      5. Drop player-seasons under `min_minutes` (default 450, i.e. five full
         matches). Below that, rates are too noisy: one shot in 20 minutes looks
         like 4.5 shots per 90.
      6. Keep the most common lineup position (2025-26 only) and how often he
         was listed there. These are for the out-of-position analysis, NOT
         features for the classifier.
    """
    col = {name: i for i, name in enumerate(player_matches.column_names)}
    first_stat_index = col["minutes"] + 1
    stat_names = [name for name in player_matches.column_names[first_stat_index:]
                  if name not in GOALKEEPER_OR_TEAM_STATS and name not in TRACKING_ONLY_STATS]

    groups = {}
    for row in player_matches.data:
        if row[col["fpl_position"]] in ("GK", "Unknown"):
            continue
        groups.setdefault((row[col["player_code"]], row[col["season"]]), []).append(row)

    columns = ["player_code", "season", "player_name", "team_code", "fpl_position",
               "lineup_position", "lineup_position_share", "matches", "minutes"] + \
              [name + "_per90" for name in stat_names]
    data = []
    for (player_code, season), rows in groups.items():
        minutes = sum(row[col["minutes"]] for row in rows)
        if minutes < min_minutes:
            continue
        # Club = the one he played the most minutes for (matters for January transfers).
        minutes_by_team = {}
        for row in rows:
            minutes_by_team[row[col["team_code"]]] = minutes_by_team.get(row[col["team_code"]], 0) + row[col["minutes"]]
        team_code = max(minutes_by_team, key=minutes_by_team.get)

        lineup_positions = [row[col["lineup_position"]] for row in rows if row[col["lineup_position"]] != "NA"]
        if lineup_positions:
            lineup_position = myutils.get_most_frequent(lineup_positions)
            lineup_share = round(lineup_positions.count(lineup_position) / len(lineup_positions), 3)
        else:
            lineup_position, lineup_share = "NA", "NA"

        rates = []
        for name in stat_names:
            j = col[name]
            recorded = [row for row in rows if row[j] != "NA"]
            recorded_minutes = sum(row[col["minutes"]] for row in recorded)
            if recorded_minutes == 0:
                rates.append("NA")
            else:
                rates.append(round(sum(row[j] for row in recorded) / recorded_minutes * 90, 3))

        data.append([player_code, season, rows[-1][col["player_name"]], team_code,
                     myutils.get_most_frequent([row[col["fpl_position"]] for row in rows]),
                     lineup_position, lineup_share, len(rows), minutes] + rates)
    data.sort(key=lambda r: (r[1], r[2]))
    return MyPyTable(columns, data)


# ---------------------------------------------------------------------------
# Step 5: match features (home-win classifier)
# ---------------------------------------------------------------------------
FORM_STATS = {  # feature name -> (what to read from the team's side, what to read from the opponent's side)
    "points": None,
    "goals_for": ("goals", None), "goals_against": (None, "goals"),
    "xg_for": ("expected_goals_xg", None), "xg_against": (None, "expected_goals_xg"),
    "shots_for": ("total_shots", None), "shots_against": (None, "total_shots"),
    "shots_on_target_for": ("shots_on_target", None),
    "big_chances_for": ("big_chances", None), "big_chances_against": (None, "big_chances"),
    "box_touches_for": ("touches_in_opposition_box", None),
    "possession": ("possession", None),
}


def build_team_log(matches):
    """Rewrites each match as two rows, one from each team's point of view.

    Rolling form is about a team's own recent games, whether at home or away,
    so it's easier to compute from a list of "team X played, scored this many,
    conceded this many..." entries in date order.
    """
    log = {}
    for row in matches.data:
        match = row_as_dict(matches, row)
        for side, other in [("home", "away"), ("away", "home")]:
            goals_for, goals_against = match[side + "_goals"], match[other + "_goals"]
            entry = {"season": match["season"], "kickoff": match["kickoff"], "match_id": match["match_id"],
                     "points": 3 if goals_for > goals_against else (1 if goals_for == goals_against else 0)}
            for feature, (own_stat, opponent_stat) in [(f, s) for f, s in FORM_STATS.items() if s]:
                if own_stat == "goals" or opponent_stat == "goals":
                    entry[feature] = goals_for if own_stat else goals_against
                elif own_stat:
                    entry[feature] = match[side + "_" + own_stat]
                else:
                    entry[feature] = match[other + "_" + opponent_stat]
            log.setdefault(match[side + "_code"], []).append(entry)
    return log


def load_other_competition_kickoffs():
    """Kickoff times of each Premier League club's cup and European matches.

    Used for a fixture-congestion feature: did the team play a midweek match
    just before this one? Only available from 2025-26 (2024-25 has no cup or
    European data). Foreign clubs have no team code, so they are skipped.
    """
    kickoffs = {}
    for season in SEASONS[1:]:
        for competition in OTHER_COMPETITIONS:
            raw = load_season_file(season, "matches.csv", competition)
            for row in raw.data:
                match = row_as_dict(raw, row)
                if not is_true(match.get("finished")) or match["kickoff_time"] == "NA":
                    continue  # unplayed, or no kickoff time recorded (can't place it in time)
                for side in ("home_team", "away_team"):
                    if match[side] != "NA":
                        kickoffs.setdefault(to_int(match[side]), []).append(normalize_kickoff(match["kickoff_time"]))
    return kickoffs


def build_match_features(matches, window=5, min_history=3, congestion_days=4):
    """One row per match, using ONLY information available before kickoff.

    This is the most important rule for the home-win task: every feature must
    be something a bettor could have known before the match. So for each match
    we look back at each team's previous `window` league matches and average
    their stats. The match's own stats (its xG, shots, etc.) are never used.

    Other decisions:
      * Form can carry over from the end of last season (a team's last few games
        in May still say something about August). It never reaches back more
        than one season. Promoted teams (and every team in early 2024-25, the
        first season we have) start with no history.
      * If a team has fewer than `min_history` previous matches, its form
        features are "NA" rather than an average of 1-2 games. The columns
        home_history / away_history let EDA decide whether to drop those rows.
      * rest_days counts days since the team's previous LEAGUE match in the same
        season. It is "NA" for a team's first match of a season, because the
        summer break (around 90 days) isn't the kind of rest we're measuring.
      * played_midweek ("yes"/"no") says whether the team played a cup or
        European match in the `congestion_days` days before kickoff. It is "NA"
        for 2024-25, where we have no cup data.
      * Labels: home_win (yes/no) and result (H/D/A). Elo ratings come from the
        cleaned matches table and are already pre-match.
    """
    team_log = build_team_log(matches)
    other_kickoffs = load_other_competition_kickoffs()
    features = list(FORM_STATS)

    columns = ["season", "gameweek", "match_id", "kickoff", "home_code", "home_name", "away_code",
               "away_name", "elo_home", "elo_away", "elo_diff"]
    for side in ("home", "away"):
        columns += [side + "_history", side + "_rest_days", side + "_played_midweek"]
        columns += [side + "_avg_" + name for name in features]
    columns += ["result", "home_win"]

    data = []
    for row in matches.data:
        match = row_as_dict(matches, row)
        season_number = SEASONS.index(match["season"])
        new_row = [match[name] for name in columns[:10]] + [round(match["elo_home"] - match["elo_away"], 1)]
        for side in ("home", "away"):
            team = match[side + "_code"]
            earlier = [entry for entry in team_log[team]
                       if entry["kickoff"] < match["kickoff"]
                       and season_number - SEASONS.index(entry["season"]) <= 1]
            recent = earlier[-window:]
            if earlier and earlier[-1]["season"] == match["season"]:
                rest_days = round(days_between(earlier[-1]["kickoff"], match["kickoff"]), 1)
            else:
                rest_days = "NA"  # first match of the season: the "rest" is the summer break
            if match["season"] == "2024-2025":
                played_midweek = "NA"
            else:
                played_midweek = "yes" if any(0 < days_between(k, match["kickoff"]) <= congestion_days
                                              for k in other_kickoffs.get(team, [])) else "no"
            new_row += [len(recent), rest_days, played_midweek]
            for name in features:
                values = [entry[name] for entry in recent if entry[name] != "NA"]
                if len(recent) < min_history or not values:
                    new_row.append("NA")
                else:
                    new_row.append(round(sum(values) / len(values), 3))
        new_row += [match["result"], match["home_win"]]
        data.append(new_row)
    return MyPyTable(columns, data)


# ---------------------------------------------------------------------------
# Data checks (used in the notebook to back up the decisions above)
# ---------------------------------------------------------------------------
def check_clubelo_timing(matches):
    """Tests whether the raw ClubElo ratings are recorded before or after each match.

    For every team we line up its matches in date order and look at how its
    rating changes. If the ratings are PRE-match, a match's result shows up in
    the change from that row to the NEXT row. If they are POST-match, it shows
    up in the change from the PREVIOUS row to this one. We correlate each
    kind of change with the match's goal difference; whichever correlates
    strongly tells us which it is.

    Returns:
        tuple of (float, float): correlation with the change to the next row,
            correlation with the change from the previous row
    """
    sequences = {}
    for row in matches.data:
        match = row_as_dict(matches, row)
        goal_difference = match["home_goals"] - match["away_goals"]
        sequences.setdefault(match["home_code"], []).append((match["clubelo_home"], goal_difference))
        sequences.setdefault(match["away_code"], []).append((match["clubelo_away"], -goal_difference))
    gd_next, change_next, gd_prev, change_prev = [], [], [], []
    for sequence in sequences.values():
        for i, (elo, goal_difference) in enumerate(sequence):
            if elo == "NA":
                continue
            if i + 1 < len(sequence) and sequence[i + 1][0] != "NA":
                gd_next.append(goal_difference)
                change_next.append(sequence[i + 1][0] - elo)
            if i > 0 and sequence[i - 1][0] != "NA":
                gd_prev.append(goal_difference)
                change_prev.append(elo - sequence[i - 1][0])
    return (myutils.compute_correlation_coefficient(gd_next, change_next),
            myutils.compute_correlation_coefficient(gd_prev, change_prev))


def compare_elo_to_clubelo(matches):
    """Correlation between our Elo and ClubElo on rows where ClubElo exists."""
    ours, theirs = [], []
    for row in matches.data:
        match = row_as_dict(matches, row)
        for side in ("home", "away"):
            if match["clubelo_" + side] != "NA":
                ours.append(match["elo_" + side])
                theirs.append(match["clubelo_" + side])
    return myutils.compute_correlation_coefficient(ours, theirs), len(ours)


# ---------------------------------------------------------------------------
# Run everything
# ---------------------------------------------------------------------------
def run_all():
    """Builds and saves all five cleaned tables. Returns them in a dict."""
    teams = build_teams()
    matches = build_matches(teams)
    player_matches = build_player_matches(matches)
    player_seasons = build_player_seasons(player_matches)
    match_features = build_match_features(matches)
    tables = {"teams.csv": teams, "matches.csv": matches, "player_matches.csv": player_matches,
              "player_seasons.csv": player_seasons, "match_features.csv": match_features}
    for filename, table in tables.items():
        save_table(table, filename)
    return tables


if __name__ == "__main__":
    for table_name, built in run_all().items():
        print(f"{table_name:22s} {built.get_shape()}")
