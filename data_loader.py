import os
import json
import re
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
_BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR    = os.path.join(_BASE_DIR, "AskAI_Data")
_MATCHUP_DIR = os.path.join(_DATA_DIR, "matchup")

_PLAYERS_PATH           = os.path.join(_DATA_DIR, "players.parquet")
_REGISTRY_PATH          = os.path.join(_DATA_DIR, "player_registry.parquet")
_TEAM_RECORDS_PATH      = os.path.join(_DATA_DIR, "team_match_records.parquet")
_MATCHUP_SUMMARY_PATH   = os.path.join(_MATCHUP_DIR, "summary.parquet")
_MATCHUP_BY_BATTER      = os.path.join(_MATCHUP_DIR, "by_batter")
_PLAYER_INDEX_PATH      = os.path.join(_DATA_DIR, "player_index.json")
_SEASON_BATTING_PATH    = os.path.join(_DATA_DIR, "season_batting.parquet")
_SEASON_BOWLING_PATH    = os.path.join(_DATA_DIR, "season_bowling.parquet")
_VENUE_BATTING_PATH     = os.path.join(_DATA_DIR, "venue_batting.parquet")
_VENUE_BOWLING_PATH     = os.path.join(_DATA_DIR, "venue_bowling.parquet")
_TEAM_VS_TEAM_PATH      = os.path.join(_DATA_DIR, "team_vs_team.parquet")
_TEAM_VS_TEAM_S_PATH    = os.path.join(_DATA_DIR, "team_vs_team_season.parquet")

# ── Module-level state ─────────────────────────────────────────────────────────
_loaded = False

players_df        = None
registry_df       = None
team_records_df   = None
matchup_df        = None
season_batting_df = None
season_bowling_df = None
venue_batting_df  = None
venue_bowling_df  = None
team_vs_team_df   = None
team_vs_team_s_df = None

_player_index = {}
_display_map  = {}
_unique_map   = {}
_player_names = []

# ── IPL titles (hardcoded) ─────────────────────────────────────────────────────
IPL_TITLES = {
    "Mumbai Indians":              5,
    "Chennai Super Kings":         5,
    "Kolkata Knight Riders":       3,
    "Rajasthan Royals":            1,
    "Sunrisers Hyderabad":         1,
    "Gujarat Titans":              1,
    "Royal Challengers Bengaluru": 1,
    "Delhi Capitals":              0,
    "Punjab Kings":                0,
    "Lucknow Super Giants":        0,
}

VALID_PREFIXES = {"Overall", "IPL", "T20I", "2025", "IPL26"}

# ── Helpers ────────────────────────────────────────────────────────────────────

def _safe(val):
    if val is None:
        return None
    try:
        if np.isnan(val):
            return None
    except (TypeError, ValueError):
        pass
    return val

def _col(prefix, stat):
    return f"{stat}_{prefix}"

def _load_optional(path, name):
    if os.path.exists(path):
        df = pd.read_parquet(path)
        print(f"{name} loaded — {len(df):,} rows")
        return df
    print(f"WARNING: {name} not found at {path}")
    return None

# ── Loader ─────────────────────────────────────────────────────────────────────

def load():
    global _loaded
    global players_df, registry_df, team_records_df
    global season_batting_df, season_bowling_df
    global venue_batting_df, venue_bowling_df
    global team_vs_team_df, team_vs_team_s_df
    global _player_index, _display_map, _unique_map, _player_names

    if _loaded:
        return

    players_df      = pd.read_parquet(_PLAYERS_PATH)
    registry_df     = pd.read_parquet(_REGISTRY_PATH)
    team_records_df = pd.read_parquet(_TEAM_RECORDS_PATH)
    print(f"players.parquet loaded — {len(players_df):,} players")

    season_batting_df = _load_optional(_SEASON_BATTING_PATH, "season_batting.parquet")
    season_bowling_df = _load_optional(_SEASON_BOWLING_PATH, "season_bowling.parquet")
    venue_batting_df  = _load_optional(_VENUE_BATTING_PATH,  "venue_batting.parquet")
    venue_bowling_df  = _load_optional(_VENUE_BOWLING_PATH,  "venue_bowling.parquet")
    team_vs_team_df   = _load_optional(_TEAM_VS_TEAM_PATH,   "team_vs_team.parquet")
    team_vs_team_s_df = _load_optional(_TEAM_VS_TEAM_S_PATH, "team_vs_team_season.parquet")

    # Build player index
    cid_to_unique = {}
    for _, row in registry_df.iterrows():
        cid = row.get("cricinfo_id")
        if cid is not None and not (isinstance(cid, float) and np.isnan(cid)):
            cid_to_unique[int(cid)] = row["unique_name"]

    with open(_PLAYER_INDEX_PATH, "r", encoding="utf-8") as f:
        raw_index = json.load(f)

    for alias, cid in raw_index.items():
        if cid is not None:
            uname = cid_to_unique.get(int(cid))
            if uname:
                _player_index[alias] = uname

    for _, row in registry_df.iterrows():
        uname = row["unique_name"]
        dname = str(row.get("display_name", uname))
        _display_map[uname]          = dname
        _unique_map[dname.lower()]   = uname
        _unique_map[uname.lower()]   = uname
        _player_index[uname.lower()] = uname
        _player_index[dname.lower()] = uname
        parts = dname.lower().split()
        if len(parts) >= 2:
            _player_index.setdefault(parts[-1], uname)
            _player_index.setdefault(parts[0] + " " + parts[-1], uname)

    _player_names = sorted(
        set(_display_map[u] for u in players_df["unique_name"].tolist()
            if u in _display_map)
    )

    # ── Manual aliases for full names Groq commonly generates ─────────────────
    _MANUAL_ALIASES = {
        "sunil narine": "SP Narine", "sunil philip narine": "SP Narine",
        "jasprit bumrah": "JJ Bumrah", "bumrah": "JJ Bumrah",
        "virat kohli": "V Kohli", "virat": "V Kohli",
        "rohit sharma": "RG Sharma", "rohit": "RG Sharma", "rg sharma": "RG Sharma",
        "ms dhoni": "MS Dhoni", "dhoni": "MS Dhoni",
        "mahendra singh dhoni": "MS Dhoni",
        "kl rahul": "KL Rahul",
        "shubman gill": "S Gill", "gill": "S Gill",
        "hardik pandya": "HH Pandya",
        "ravindra jadeja": "RA Jadeja",
        "yuzvendra chahal": "YS Chahal", "chahal": "YS Chahal",
        "bhuvneshwar kumar": "B Kumar",
        "mohammed shami": "Mohammed Shami",
        "mohammed siraj": "Mohammed Siraj",
        "ravichandran ashwin": "R Ashwin",
        "shreyas iyer": "SS Iyer",
        "rishabh pant": "RR Pant",
        "david warner": "DA Warner",
        "kane williamson": "KS Williamson",
        "jos buttler": "JC Buttler",
        "rashid khan": "Rashid Khan",
        "chris gayle": "CH Gayle",
        "ab de villiers": "AB de Villiers", "ab devilliers": "AB de Villiers",
        "faf du plessis": "F du Plessis",
        "quinton de kock": "Q de Kock",
        "andre russell": "AD Russell",
        "kieron pollard": "KA Pollard",
        "josh hazlewood": "JR Hazlewood", "hazlewood": "JR Hazlewood",
        "trent boult": "TA Boult",
        "pat cummins": "PJ Cummins",
        "mitchell starc": "MA Starc",
        "sam curran": "SM Curran",
        "nicholas pooran": "N Pooran",
        "ishan kishan": "IK Kishan",
        "sanju samson": "SV Samson",
        "axar patel": "AR Patel",
        "washington sundar": "W Sundar",
        "abhishek sharma": "Abhishek Sharma",
        "prithvi shaw": "PP Shaw",
        "yashasvi jaiswal": "YBK Jaiswal", "jaiswal": "YBK Jaiswal",
        "ruturaj gaikwad": "RD Gaikwad", "gaikwad": "RD Gaikwad",
        "tilak varma": "T Varma",
        "suryakumar yadav": "SA Yadav", "surya": "SA Yadav",
        "karun nair": "KK Nair",
        "nitish reddy": "N Kumar Reddy",
        "narine": "SP Narine",
    }
    for alias, short_name in _MANUAL_ALIASES.items():
        match = registry_df[registry_df["unique_name"].str.contains(
            short_name.replace(" ", ".*"), case=False, na=False, regex=True
        )]
        if not match.empty:
            _player_index[alias] = match.iloc[0]["unique_name"]
        else:
            existing = _player_index.get(short_name.lower())
            if existing:
                _player_index[alias] = existing

    _loaded = True
    print(f"Data layer ready — {len(_player_names):,} players indexed")


# ── Entity resolution ──────────────────────────────────────────────────────────

def resolve_player(name: str) -> str | None:
    load()
    key = name.lower().strip()
    if key in _player_index:
        return _player_index[key]
    for dname_lower, uname in _unique_map.items():
        if key in dname_lower:
            return uname
    return None

def resolve_display(unique_name: str) -> str:
    return _display_map.get(unique_name, unique_name)

def all_player_names() -> list[str]:
    load()
    return _player_names


# ── Player stats (players.parquet prefixes) ────────────────────────────────────

def _player_row(unique_name: str):
    load()
    rows = players_df[players_df["unique_name"] == unique_name]
    return rows.iloc[0] if not rows.empty else None

def get_player_stats(name: str, prefix: str = "IPL") -> dict | None:
    load()
    if prefix not in VALID_PREFIXES:
        prefix = "IPL"
    unique_name = resolve_player(name)
    if not unique_name:
        return None
    row = _player_row(unique_name)
    if row is None:
        return None

    def g(stat):
        return _safe(row.get(_col(prefix, stat)))

    reg_rows = registry_df[registry_df["unique_name"] == unique_name]
    nation   = reg_rows.iloc[0]["nation"] if not reg_rows.empty else ""

    return {
        "name":         resolve_display(unique_name),
        "unique_name":  unique_name,
        "nation":       nation,
        "ipl_ever":     bool(reg_rows.iloc[0]["ipl_ever"]) if not reg_rows.empty else False,
        "innings":      g("Innings"),
        "runs":         g("Runs"),
        "balls_faced":  g("Balls_Faced"),
        "dismissed":    g("Dismissed"),
        "avg":          g("Batting_Avg"),
        "sr":           g("Batting_SR"),
        "fours":        g("Fours"),
        "sixes":        g("Sixes"),
        "dot_pct":      g("Dot_Pct"),
        "bowl_innings": g("Bowl_Innings"),
        "wickets":      g("Wickets"),
        "balls_bowled": g("Balls_Bowled"),
        "runs_conceded":g("Runs_Conceded"),
        "economy":      g("Econ"),
        "bowl_avg":     g("Bowling_Avg"),
        "bowl_sr":      g("Bowling_SR"),
    }


# ── Season stats ───────────────────────────────────────────────────────────────

def get_season_player_stats(player_name: str, season: int,
                             competition: str = "IPL",
                             phase: str = "ALL") -> dict | None:
    """Full stats for a player in a specific season/competition/phase."""
    load()
    unique_name = resolve_player(player_name)
    if not unique_name:
        return None

    result = {"name": resolve_display(unique_name), "season": season,
              "competition": competition, "phase": phase}

    if season_batting_df is not None:
        bat = season_batting_df[
            (season_batting_df["player"] == unique_name) &
            (season_batting_df["season"] == season) &
            (season_batting_df["competition"] == competition) &
            (season_batting_df["phase"] == phase)
        ]
        if not bat.empty:
            r = bat.iloc[0]
            result.update({
                "innings":     int(r.get("innings", 0) or 0),
                "runs":        int(r.get("runs", 0) or 0),
                "balls_faced": int(r.get("balls_faced", 0) or 0),
                "dismissed":   int(r.get("dismissed", 0) or 0),
                "avg":         _safe(r.get("avg")),
                "sr":          _safe(r.get("sr")),
                "fours":       int(r.get("fours", 0) or 0),
                "sixes":       int(r.get("sixes", 0) or 0),
                "dot_pct":     _safe(r.get("dot_pct")),
            })

    if season_bowling_df is not None:
        bowl = season_bowling_df[
            (season_bowling_df["player"] == unique_name) &
            (season_bowling_df["season"] == season) &
            (season_bowling_df["competition"] == competition) &
            (season_bowling_df["phase"] == phase)
        ]
        if not bowl.empty:
            r = bowl.iloc[0]
            result.update({
                "bowl_innings":  int(r.get("bowl_innings", 0) or 0),
                "wickets":       int(r.get("wickets", 0) or 0),
                "balls_bowled":  int(r.get("balls_bowled", 0) or 0),
                "runs_conceded": int(r.get("runs_conceded", 0) or 0),
                "economy":       _safe(r.get("economy")),
                "bowl_avg":      _safe(r.get("bowl_avg")),
                "bowl_sr":       _safe(r.get("bowl_sr")),
                "bowl_dot_pct":  _safe(r.get("dot_pct")),
            })

    return result if len(result) > 4 else None


def get_season_leaderboard(stat: str, season: int,
                            competition: str = "IPL",
                            phase: str = "ALL",
                            n: int = 5) -> list[dict]:
    """Top N players by any stat in a specific season/competition/phase."""
    load()

    bat_stats  = {"runs", "avg", "sr", "sixes", "fours", "dot_pct", "balls_faced", "innings"}
    bowl_stats = {"wickets", "economy", "bowl_avg", "bowl_sr"}
    ascending  = stat in {"economy", "bowl_avg", "bowl_sr"}

    if stat in bat_stats and season_batting_df is not None:
        df = season_batting_df
        col = stat
        min_col = "innings" if stat in {"avg", "sr"} else None
        min_val = 3
    elif stat in bowl_stats and season_bowling_df is not None:
        df = season_bowling_df
        col = stat
        min_col = "wickets" if stat in {"bowl_avg", "bowl_sr"} else None
        min_val = 3
    else:
        return []

    filtered = df[
        (df["season"] == season) &
        (df["competition"] == competition) &
        (df["phase"] == phase)
    ].dropna(subset=[col])

    if min_col:
        filtered = filtered[filtered[min_col] >= min_val]

    if filtered.empty:
        return []

    top = filtered.nlargest(n, col) if not ascending else filtered.nsmallest(n, col)

    return [
        {"rank": i+1, "player": resolve_display(r["player"]) or r["player"],
         "value": round(float(r[col]), 2), "stat": stat}
        for i, (_, r) in enumerate(top.iterrows())
    ]


# ── Career leaderboards ────────────────────────────────────────────────────────

def get_career_leaderboard(stat: str, prefix: str = "IPL",
                            n: int = 5, min_qualifier: int = None) -> list[dict]:
    """Top N players by stat across career prefix."""
    load()
    if prefix not in VALID_PREFIXES:
        prefix = "IPL"

    stat_col_map = {
        "runs": "Runs", "wickets": "Wickets", "sixes": "Sixes", "fours": "Fours",
        "avg": "Batting_Avg", "sr": "Batting_SR", "economy": "Econ",
        "bowl_avg": "Bowling_Avg", "bowl_sr": "Bowling_SR",
        "dot_pct": "Dot_Pct", "balls_faced": "Balls_Faced",
    }
    ascending = stat in {"economy", "bowl_avg", "bowl_sr"}
    col = _col(prefix, stat_col_map.get(stat, stat))
    df  = players_df.dropna(subset=[col])

    if min_qualifier is None:
        defaults = {"avg": 20, "sr": 20, "economy": 20, "bowl_avg": 30, "bowl_sr": 30}
        min_qualifier = defaults.get(stat, 0)

    if min_qualifier > 0:
        qual_col_map = {
            "avg": "Innings", "sr": "Innings",
            "economy": "Bowl_Innings", "bowl_avg": "Wickets", "bowl_sr": "Wickets",
        }
        qual_col = _col(prefix, qual_col_map.get(stat, "Innings"))
        if qual_col in df.columns:
            df = df[df[qual_col] >= min_qualifier]

    df = df.sort_values(col, ascending=ascending).head(n)

    return [
        {"rank": i+1, "player": resolve_display(r["unique_name"]),
         "value": round(float(r[col]), 2), "stat": stat, "prefix": prefix}
        for i, (_, r) in enumerate(df.iterrows())
    ]


# ── Matchup ────────────────────────────────────────────────────────────────────

def _load_matchup():
    global matchup_df
    if matchup_df is None:
        matchup_df = pd.read_parquet(_MATCHUP_SUMMARY_PATH)
    return matchup_df

def get_matchup(batter_name: str, bowler_name: str,
                competition: str = "IPL",
                phase: str = "ALL") -> dict | None:
    load()
    b_unique  = resolve_player(batter_name)
    bw_unique = resolve_player(bowler_name)
    if not b_unique or not bw_unique:
        return None

    safe_name   = b_unique.replace(" ", "_").replace("/", "-")
    batter_file = os.path.join(_MATCHUP_BY_BATTER, f"{safe_name}.parquet")

    if os.path.exists(batter_file):
        df = pd.read_parquet(batter_file)
    else:
        df = _load_matchup()
        df = df[df["batter"] == b_unique]

    rows = df[
        (df["bowler"]      == bw_unique) &
        (df["competition"] == competition) &
        (df["phase"]       == phase)
    ]
    if rows.empty:
        return None

    r = rows.iloc[0]
    return {
        "batter":       resolve_display(b_unique),
        "bowler":       resolve_display(bw_unique),
        "competition":  competition,
        "phase":        phase,
        "balls":        int(r["balls"]),
        "runs":         int(r["runs"]),
        "dismissed":    int(r["dismissed"]),
        "fours":        int(r["fours"]),
        "sixes":        int(r["sixes"]),
        "sr":           _safe(r.get("sr")),
        "dot_pct":      _safe(r.get("dot_pct")),
        "dismiss_rate": _safe(r.get("dismiss_rate")),
    }

def get_batter_vs_all_bowlers(batter_name: str,
                               competition: str = "IPL",
                               phase: str = "ALL",
                               min_balls: int = 12,
                               n: int = 5) -> dict:
    load()
    b_unique = resolve_player(batter_name)
    if not b_unique:
        return {"error": f"Player not found: {batter_name}"}

    safe_name   = b_unique.replace(" ", "_").replace("/", "-")
    batter_file = os.path.join(_MATCHUP_BY_BATTER, f"{safe_name}.parquet")

    if os.path.exists(batter_file):
        df = pd.read_parquet(batter_file)
    else:
        df = _load_matchup()
        df = df[df["batter"] == b_unique]

    df = df[
        (df["competition"] == competition) &
        (df["phase"]       == phase) &
        (df["balls"]       >= min_balls)
    ]
    if df.empty:
        return {"error": f"No matchup data for {resolve_display(b_unique)} in {competition}"}

    df = df.copy()
    df["bowler_display"] = df["bowler"].apply(resolve_display)

    weak_against = (df.sort_values("dismiss_rate", ascending=False)
                      .head(n)[["bowler_display","balls","runs","dismissed","sr","dismiss_rate"]]
                      .to_dict("records"))
    dominates    = (df.sort_values("sr", ascending=False)
                      .head(n)[["bowler_display","balls","runs","dismissed","sr","dismiss_rate"]]
                      .to_dict("records"))

    return {
        "batter":       resolve_display(b_unique),
        "competition":  competition,
        "phase":        phase,
        "weak_against": weak_against,
        "dominates":    dominates,
    }


# ── Venue stats ────────────────────────────────────────────────────────────────

def get_venue_stats(player_name: str, venue_query: str,
                    role: str = "both") -> dict | None:
    load()
    unique_name = resolve_player(player_name)
    if not unique_name:
        return None

    result = {"name": resolve_display(unique_name), "venue_query": venue_query}

    if role in ("batting", "both") and venue_batting_df is not None:
        bat = venue_batting_df[
            venue_batting_df["venue"].str.contains(venue_query, case=False, na=False) &
            (venue_batting_df["player"] == unique_name)
        ]
        if not bat.empty:
            agg = bat.agg({"innings":"sum","runs":"sum","balls_faced":"sum",
                           "dismissed":"sum","fours":"sum","sixes":"sum"})
            result["batting"] = {
                "innings":     int(agg["innings"]),
                "runs":        int(agg["runs"]),
                "balls_faced": int(agg["balls_faced"]),
                "dismissed":   int(agg["dismissed"]),
                "fours":       int(agg["fours"]),
                "sixes":       int(agg["sixes"]),
                "avg":  round(agg["runs"]/agg["dismissed"], 2) if agg["dismissed"] else None,
                "sr":   round(agg["runs"]/agg["balls_faced"]*100, 2) if agg["balls_faced"] else None,
                "venues_matched": bat["venue"].unique().tolist(),
            }

    if role in ("bowling", "both") and venue_bowling_df is not None:
        bowl = venue_bowling_df[
            venue_bowling_df["venue"].str.contains(venue_query, case=False, na=False) &
            (venue_bowling_df["player"] == unique_name)
        ]
        if not bowl.empty:
            agg = bowl.agg({"bowl_innings":"sum","balls_bowled":"sum",
                            "runs_conceded":"sum","wickets":"sum"})
            result["bowling"] = {
                "bowl_innings":  int(agg["bowl_innings"]),
                "balls_bowled":  int(agg["balls_bowled"]),
                "runs_conceded": int(agg["runs_conceded"]),
                "wickets":       int(agg["wickets"]),
                "economy":  round(agg["runs_conceded"]/agg["balls_bowled"]*6, 2) if agg["balls_bowled"] else None,
                "bowl_avg": round(agg["runs_conceded"]/agg["wickets"], 2) if agg["wickets"] else None,
            }

    return result if len(result) > 2 else None


def get_venue_leaderboard(venue_query: str, stat: str = "runs",
                           n: int = 5) -> list[dict]:
    load()
    ascending = stat in {"economy", "bowl_avg"}

    if stat in {"runs","avg","sr","sixes","fours"} and venue_batting_df is not None:
        df = venue_batting_df[
            venue_batting_df["venue"].str.contains(venue_query, case=False, na=False)
        ].copy()
        if df.empty:
            return []
        grp = df.groupby("player").agg(
            innings=("innings","sum"), runs=("runs","sum"),
            balls_faced=("balls_faced","sum"), dismissed=("dismissed","sum"),
            fours=("fours","sum"), sixes=("sixes","sum")
        ).reset_index()
        grp["avg"] = (grp["runs"] / grp["dismissed"].replace(0, np.nan)).round(2)
        grp["sr"]  = (grp["runs"] / grp["balls_faced"] * 100).round(2)
        if stat == "avg": grp = grp[grp["innings"] >= 5]
        grp = grp.dropna(subset=[stat])
        top = grp.nlargest(n, stat) if not ascending else grp.nsmallest(n, stat)
        return [{"rank": i+1, "player": resolve_display(r["player"]) or r["player"],
                 "value": round(float(r[stat]), 2), "stat": stat}
                for i, (_, r) in enumerate(top.iterrows())]

    elif stat in {"wickets","economy","bowl_avg"} and venue_bowling_df is not None:
        df = venue_bowling_df[
            venue_bowling_df["venue"].str.contains(venue_query, case=False, na=False)
        ].copy()
        if df.empty:
            return []
        grp = df.groupby("player").agg(
            bowl_innings=("bowl_innings","sum"), balls_bowled=("balls_bowled","sum"),
            runs_conceded=("runs_conceded","sum"), wickets=("wickets","sum")
        ).reset_index()
        grp["economy"]  = (grp["runs_conceded"] / grp["balls_bowled"] * 6).round(2)
        grp["bowl_avg"] = (grp["runs_conceded"] / grp["wickets"].replace(0, np.nan)).round(2)
        if stat == "bowl_avg": grp = grp[grp["wickets"] >= 5]
        grp = grp.dropna(subset=[stat])
        top = grp.nlargest(n, stat) if not ascending else grp.nsmallest(n, stat)
        return [{"rank": i+1, "player": resolve_display(r["player"]) or r["player"],
                 "value": round(float(r[stat]), 2), "stat": stat}
                for i, (_, r) in enumerate(top.iterrows())]

    return []


# ── Team vs Team ───────────────────────────────────────────────────────────────

def _normalize_team(name: str) -> str:
    aliases = {
        "mi": "Mumbai Indians", "mumbai": "Mumbai Indians",
        "csk": "Chennai Super Kings", "chennai": "Chennai Super Kings",
        "rcb": "Royal Challengers Bengaluru", "bangalore": "Royal Challengers Bengaluru",
        "bengaluru": "Royal Challengers Bengaluru",
        "kkr": "Kolkata Knight Riders", "kolkata": "Kolkata Knight Riders",
        "srh": "Sunrisers Hyderabad", "sunrisers": "Sunrisers Hyderabad", "hyderabad": "Sunrisers Hyderabad",
        "rr": "Rajasthan Royals", "rajasthan": "Rajasthan Royals",
        "pbks": "Punjab Kings", "punjab": "Punjab Kings", "kxip": "Punjab Kings",
        "dc": "Delhi Capitals", "delhi": "Delhi Capitals", "dd": "Delhi Capitals",
        "gt": "Gujarat Titans", "gujarat": "Gujarat Titans",
        "lsg": "Lucknow Super Giants", "lucknow": "Lucknow Super Giants",
    }
    key = name.lower().strip()
    if key in aliases:
        return aliases[key]
    for k, v in aliases.items():
        if k in key or key in v.lower():
            return v
    return name

def _extract_team_from_question(question: str) -> str | None:
    """Extract a single team name from a question using the alias map."""
    q = question.lower()
    aliases = {
        "mi": "Mumbai Indians", "mumbai": "Mumbai Indians",
        "csk": "Chennai Super Kings", "chennai": "Chennai Super Kings",
        "rcb": "Royal Challengers Bengaluru", "bangalore": "Royal Challengers Bengaluru",
        "bengaluru": "Royal Challengers Bengaluru", "royal challengers": "Royal Challengers Bengaluru",
        "kkr": "Kolkata Knight Riders", "kolkata": "Kolkata Knight Riders",
        "srh": "Sunrisers Hyderabad", "sunrisers": "Sunrisers Hyderabad", "hyderabad": "Sunrisers Hyderabad",
        "rr": "Rajasthan Royals", "rajasthan": "Rajasthan Royals",
        "pbks": "Punjab Kings", "punjab": "Punjab Kings", "kxip": "Punjab Kings",
        "dc": "Delhi Capitals", "delhi": "Delhi Capitals",
        "gt": "Gujarat Titans", "gujarat": "Gujarat Titans",
        "lsg": "Lucknow Super Giants", "lucknow": "Lucknow Super Giants",
    }
    for alias, full_name in aliases.items():
        if alias in q.split() or alias in q:
            return full_name
    return None


def get_team_vs_team(team1: str, team2: str,
                     competition: str = "IPL",
                     season: int = None) -> dict | None:
    load()
    t1 = _normalize_team(team1)
    t2 = _normalize_team(team2)
    ta, tb = sorted([t1, t2])

    if season is not None and team_vs_team_s_df is not None:
        df = team_vs_team_s_df[
            (team_vs_team_s_df["team_a"] == ta) &
            (team_vs_team_s_df["team_b"] == tb) &
            (team_vs_team_s_df["competition"] == competition) &
            (team_vs_team_s_df["season"] == season)
        ]
    elif team_vs_team_df is not None:
        df = team_vs_team_df[
            (team_vs_team_df["team_a"] == ta) &
            (team_vs_team_df["team_b"] == tb) &
            (team_vs_team_df["competition"] == competition)
        ]
    else:
        return None

    if df.empty:
        return None

    r = df.iloc[0]
    ta_wins = int(r["team_a_wins"])
    tb_wins = int(r["team_b_wins"])

    return {
        "team1":       t1,
        "team2":       t2,
        "competition": competition,
        "season":      season,
        "matches":     int(r["matches"]),
        "team1_wins":  ta_wins if t1 == ta else tb_wins,
        "team2_wins":  tb_wins if t1 == ta else ta_wins,
        "no_result":   int(r.get("no_result", 0)),
    }


# ── Standings + Titles ─────────────────────────────────────────────────────────

IPL26_STANDINGS = [
    {"position": 1, "team": "Royal Challengers Bengaluru", "played": 14, "won": 9, "lost": 5, "nr": 0, "points": 18, "nrr": 0.408},
    {"position": 2, "team": "Mumbai Indians",              "played": 14, "won": 9, "lost": 5, "nr": 0, "points": 18, "nrr": 0.368},
    {"position": 3, "team": "Sunrisers Hyderabad",         "played": 14, "won": 8, "lost": 6, "nr": 0, "points": 16, "nrr": 0.215},
    {"position": 4, "team": "Punjab Kings",                "played": 14, "won": 8, "lost": 6, "nr": 0, "points": 16, "nrr": 0.098},
    {"position": 5, "team": "Kolkata Knight Riders",       "played": 14, "won": 7, "lost": 7, "nr": 0, "points": 14, "nrr": 0.130},
    {"position": 6, "team": "Delhi Capitals",              "played": 14, "won": 7, "lost": 7, "nr": 0, "points": 14, "nrr": -0.050},
    {"position": 7, "team": "Chennai Super Kings",         "played": 14, "won": 6, "lost": 8, "nr": 0, "points": 12, "nrr": -0.182},
    {"position": 8, "team": "Gujarat Titans",              "played": 14, "won": 6, "lost": 8, "nr": 0, "points": 12, "nrr": -0.223},
    {"position": 9, "team": "Rajasthan Royals",            "played": 14, "won": 5, "lost": 9, "nr": 0, "points": 10, "nrr": -0.331},
    {"position":10, "team": "Lucknow Super Giants",        "played": 14, "won": 5, "lost": 9, "nr": 0, "points": 10, "nrr": -0.441},
]

def get_team_win_rate(team: str = None, competition: str = "IPL") -> list[dict]:
    """
    Overall win rate for one team or all teams in a competition.
    team: optional partial name e.g. 'Punjab', 'MI', 'CSK'
    Returns list of dicts with team, matches, wins, win_rate
    """
    load()
    df = team_records_df[team_records_df["competition"] == competition].copy()
    if df.empty:
        return []

    # 'won' column is unreliable for defunct teams (always False).
    # Derive wins by checking if opponent lost (opponent's won=False in same match).
    df = df.copy()
    opp = team_records_df[["match_id","team","won"]].rename(
        columns={"team": "opponent", "won": "opp_won"}
    )
    df = df.merge(opp, on=["match_id","opponent"], how="left")
    # A team won if either own 'won'=True OR opponent 'won'=False
    df["win"] = df["won"].fillna(False) | (~df["opp_won"].fillna(True))
    stats = df.groupby("team").agg(
        matches=("match_id", "nunique"),
        wins=("win", "sum")
    ).reset_index()
    stats["win_rate"] = (stats["wins"] / stats["matches"] * 100).round(1)
    stats = stats[stats["matches"] >= 10]  # exclude tiny sample teams
    stats = stats.sort_values("win_rate", ascending=False).reset_index(drop=True)

    if team:
        # Normalize and filter to single team
        t_norm = _normalize_team(team)
        # Try exact match first, then partial
        exact = stats[stats["team"] == t_norm]
        if not exact.empty:
            stats = exact
        else:
            stats = stats[stats["team"].str.contains(team, case=False, na=False)]

    return [
        {
            "rank":     i + 1,
            "team":     r["team"],
            "matches":  int(r["matches"]),
            "wins":     int(r["wins"]),
            "win_rate": float(r["win_rate"]),
        }
        for i, r in stats.iterrows()
    ]


def get_standings() -> list[dict]:
    return IPL26_STANDINGS

def ipl_titles_table() -> list[dict]:
    return sorted(
        [{"team": t, "titles": c} for t, c in IPL_TITLES.items()],
        key=lambda x: x["titles"], reverse=True,
    )


# ── Legacy helpers ─────────────────────────────────────────────────────────────

def top_run_scorers(n=5, prefix="IPL"):
    rows = get_career_leaderboard("runs", prefix, n)
    # Add legacy keys
    for r in rows:
        r["runs"] = r["value"]
    return rows

def top_wicket_takers(n=5, prefix="IPL"):
    rows = get_career_leaderboard("wickets", prefix, n)
    for r in rows:
        r["wickets"] = r["value"]
        r["economy"] = None
        r["avg"]     = None
    return rows

def top_six_hitters(n=5, prefix="IPL"):
    rows = get_career_leaderboard("sixes", prefix, n)
    for r in rows:
        r["sixes"] = r["value"]
        r["sr"]    = None
    return rows

def top_run_scorers_ipl26(n=5):
    return top_run_scorers(n, "IPL26")

def top_wicket_takers_ipl26(n=5):
    return top_wicket_takers(n, "IPL26")

def top_form_batters(n=5):
    return top_run_scorers(n, "2025")

def compare_players(name1: str, name2: str, prefix: str = "IPL") -> dict:
    load()
    s1 = get_player_stats(name1, prefix)
    s2 = get_player_stats(name2, prefix)
    if not s1:
        return {"error": f"Could not find player: {name1}"}
    if not s2:
        return {"error": f"Could not find player: {name2}"}

    def verdict(a, b, higher_is_better=True):
        if a is None or b is None:
            return "n/a"
        return s1["name"] if (a > b) == higher_is_better else s2["name"]

    return {
        "player1": s1, "player2": s2,
        "edges": {
            "more_runs":    verdict(s1["runs"],    s2["runs"]),
            "better_avg":   verdict(s1["avg"],     s2["avg"]),
            "better_sr":    verdict(s1["sr"],      s2["sr"]),
            "more_wickets": verdict(s1["wickets"], s2["wickets"]),
            "better_econ":  verdict(s1["economy"], s2["economy"], higher_is_better=False),
            "more_sixes":   verdict(s1["sixes"],   s2["sixes"]),
        },
    }

def extract_compare_names(question: str):
    load()
    q = question.lower()
    for sep in [" vs ", " versus ", " and ", " with ", " against "]:
        if sep in q:
            q_clean = re.sub(r"^(compare|who is better|battle|fight)\s*", "", q).strip()
            parts   = q_clean.split(sep, 1)
            if len(parts) == 2:
                n1 = resolve_player(parts[0].strip())
                n2 = resolve_player(parts[1].strip())
                if n1 and n2:
                    return n1, n2
    found = []
    for dname in _player_names:
        if dname.lower() in question.lower():
            uname = resolve_player(dname)
            if uname and uname not in found:
                found.append(uname)
        if len(found) == 2:
            break
    return (found[0], found[1]) if len(found) == 2 else (None, None)

def top_season_run_scorers(season: int, n: int = 5) -> list[dict]:
    rows = get_season_leaderboard("runs", season, "IPL", "ALL", n)
    for r in rows:
        r["runs"] = r["value"]
    return rows

def top_season_wicket_takers(season: int, n: int = 5) -> list[dict]:
    rows = get_season_leaderboard("wickets", season, "IPL", "ALL", n)
    for r in rows:
        r["wickets"] = r["value"]
    return rows
