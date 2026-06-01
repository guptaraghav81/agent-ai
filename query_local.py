"""
query_local.py — Query AskAI parquet files directly
Usage: python query_local.py
From: C:\SD\SF360\sportsfan360-ai-agent-main
"""

import pandas as pd
import numpy as np
import os

DATA_DIR = "AskAI_Data"

def load():
    data = {}
    files = {
        "players":      "players.parquet",
        "registry":     "player_registry.parquet",
        "matchup":      "matchup/summary.parquet",
        "season_bat":   "season_batting.parquet",
        "season_bowl":  "season_bowling.parquet",
        "venue_bat":    "venue_batting.parquet",
        "venue_bowl":   "venue_bowling.parquet",
        "tvt":          "team_vs_team.parquet",
    }
    for key, fname in files.items():
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            data[key] = pd.read_parquet(path)
            print(f"✓ {fname}")
        else:
            print(f"✗ {fname} — not found")
    return data


def matchup(data, batter, bowler, competition="IPL", phase="ALL"):
    df = data["matchup"]
    rows = df[
        df["batter"].str.contains(batter, case=False, na=False) &
        df["bowler"].str.contains(bowler, case=False, na=False) &
        (df["competition"] == competition) &
        (df["phase"] == phase)
    ]
    print(f"\n=== {batter} vs {bowler} | {competition} | {phase} ===")
    if rows.empty:
        print("No data found")
    else:
        print(rows[["batter","bowler","competition","phase","balls","runs","dismissed","sr","dot_pct","dismiss_rate"]].to_string(index=False))


def matchup_all_phases(data, batter, bowler, competition="IPL"):
    df = data["matchup"]
    rows = df[
        df["batter"].str.contains(batter, case=False, na=False) &
        df["bowler"].str.contains(bowler, case=False, na=False) &
        (df["competition"] == competition)
    ]
    print(f"\n=== {batter} vs {bowler} | {competition} | ALL PHASES ===")
    if rows.empty:
        print("No data found")
    else:
        print(rows[["batter","bowler","phase","balls","runs","dismissed","sr","dot_pct"]].to_string(index=False))


def season_runs(data, season, competition="IPL", n=5):
    df = data["season_bat"]
    rows = df[
        (df["season"] == season) &
        (df["competition"] == competition) &
        (df["phase"] == "ALL")
    ].nlargest(n, "runs")
    print(f"\n=== Most runs | {competition} {season} | Top {n} ===")
    print(rows[["player","innings","runs","avg","sr","fours","sixes"]].to_string(index=False))


def season_wickets(data, season, competition="IPL", n=5):
    df = data["season_bowl"]
    rows = df[
        (df["season"] == season) &
        (df["competition"] == competition) &
        (df["phase"] == "ALL")
    ].nlargest(n, "wickets")
    print(f"\n=== Most wickets | {competition} {season} | Top {n} ===")
    print(rows[["player","bowl_innings","wickets","economy","bowl_avg"]].to_string(index=False))


def player_season(data, player, season, competition="IPL", phase="ALL"):
    bat = data["season_bat"]
    bowl = data["season_bowl"]

    b = bat[
        bat["player"].str.contains(player, case=False, na=False) &
        (bat["season"] == season) &
        (bat["competition"] == competition) &
        (bat["phase"] == phase)
    ]
    bw = bowl[
        bowl["player"].str.contains(player, case=False, na=False) &
        (bowl["season"] == season) &
        (bowl["competition"] == competition) &
        (bowl["phase"] == phase)
    ]

    print(f"\n=== {player} | {competition} {season} | {phase} ===")
    if not b.empty:
        print("BATTING:")
        print(b[["player","innings","runs","avg","sr","fours","sixes"]].to_string(index=False))
    if not bw.empty:
        print("BOWLING:")
        print(bw[["player","bowl_innings","wickets","economy","bowl_avg"]].to_string(index=False))
    if b.empty and bw.empty:
        print("No data found")


def career_leaderboard(data, stat, prefix="IPL", n=10):
    df = data["players"]
    col = f"{stat}_{prefix}"
    if col not in df.columns:
        print(f"Column {col} not found")
        return
    ascending = stat in {"Econ","Bowling_Avg","Bowling_SR"}
    rows = df.dropna(subset=[col]).sort_values(col, ascending=ascending).head(n)
    print(f"\n=== {stat} | {prefix} | Top {n} ===")
    print(rows[["unique_name", col]].to_string(index=False))


def venue(data, player, venue_str):
    vb = data["venue_bat"]
    vbow = data["venue_bowl"]

    b = vb[
        vb["player"].str.contains(player, case=False, na=False) &
        vb["venue"].str.contains(venue_str, case=False, na=False)
    ]
    bw = vbow[
        vbow["player"].str.contains(player, case=False, na=False) &
        vbow["venue"].str.contains(venue_str, case=False, na=False)
    ]

    print(f"\n=== {player} at {venue_str} ===")
    if not b.empty:
        print("BATTING:")
        print(b[["player","venue","innings","runs","avg","sr"]].to_string(index=False))
    if not bw.empty:
        print("BOWLING:")
        print(bw[["player","venue","bowl_innings","wickets","economy"]].to_string(index=False))


def team_h2h(data, team1, team2, competition="IPL"):
    df = data["tvt"]
    ta, tb = sorted([team1, team2])
    rows = df[
        (df["team_a"].str.contains(ta, case=False, na=False)) &
        (df["team_b"].str.contains(tb, case=False, na=False)) &
        (df["competition"] == competition)
    ]
    print(f"\n=== {team1} vs {team2} | {competition} ===")
    if rows.empty:
        print("No data found")
    else:
        print(rows[["team_a","team_b","matches","team_a_wins","team_b_wins","no_result"]].to_string(index=False))


# ── Run queries ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading parquet files...")
    data = load()

    # ── Edit these to test whatever you want ──────────────────────────────────

    matchup_all_phases(data, "Kohli", "Bumrah", "IPL")
    season_runs(data, 2010, "IPL", n=10)
    season_runs(data, 2008, "IPL", n=5)
    player_season(data, "Kohli", 2016, "IPL")
    player_season(data, "Bumrah", 2020, "IPL", phase="DEATH")
    career_leaderboard(data, "Runs", "IPL", n=5)
    career_leaderboard(data, "Wickets", "IPL", n=5)
    venue(data, "Kohli", "Wankhede")
    team_h2h(data, "Mumbai", "Chennai", "IPL")
