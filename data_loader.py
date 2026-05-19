import os
import json
import re
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
_BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR    = os.path.join(_BASE_DIR, "AskAI_Data")
_MATCHUP_DIR = os.path.join(_DATA_DIR, "matchup")

_PLAYERS_PATH         = os.path.join(_DATA_DIR, "players.parquet")
_REGISTRY_PATH        = os.path.join(_DATA_DIR, "player_registry.parquet")
_TEAM_RECORDS_PATH    = os.path.join(_DATA_DIR, "team_match_records.parquet")
_MATCHUP_SUMMARY_PATH  = os.path.join(_MATCHUP_DIR, "summary.parquet")
_MATCHUP_BY_BATTER     = os.path.join(_MATCHUP_DIR, "by_batter")
_PLAYER_INDEX_PATH     = os.path.join(_DATA_DIR, "player_index.json")
_SEASON_BATTING_PATH   = os.path.join(_DATA_DIR, "season_batting.parquet")
_SEASON_BOWLING_PATH   = os.path.join(_DATA_DIR, "season_bowling.parquet")

# ── Module-level state ─────────────────────────────────────────────────────────
_loaded = False

players_df        = None   # players.parquet  — 2,516 players, all prefixes
registry_df       = None   # player_registry.parquet
team_records_df   = None   # team_match_records.parquet
matchup_df        = None   # matchup/summary.parquet (loaded lazily)
season_batting_df = None   # season_batting.parquet — runs per player per IPL season
season_bowling_df = None   # season_bowling.parquet — wickets per player per IPL season

_player_index   = {}     # lowercase name/alias → unique_name (for resolution)
_display_map    = {}     # unique_name → display_name
_unique_map     = {}     # lowercase display_name → unique_name
_player_names   = []     # sorted display names for autocomplete

# ── IPL titles (stable history — hardcoded) ────────────────────────────────────
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

# ── Valid prefixes ─────────────────────────────────────────────────────────────
VALID_PREFIXES = {"Overall", "IPL", "T20I", "2025", "IPL26"}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _safe(val):
    """Return None for NaN/missing, else the value."""
    if val is None:
        return None
    try:
        if np.isnan(val):
            return None
    except (TypeError, ValueError):
        pass
    return val


def _col(prefix, stat):
    """Return column name e.g. _col('IPL', 'Runs') → 'Runs_IPL'"""
    return f"{stat}_{prefix}"


# ── Loader ─────────────────────────────────────────────────────────────────────

def load():
    """Load all parquet files once at startup. Safe to call multiple times."""
    global _loaded
    global players_df, registry_df, team_records_df
    global season_batting_df, season_bowling_df
    global _player_index, _display_map, _unique_map, _player_names

    if _loaded:
        return

    # ── players.parquet ───────────────────────────────────────────────────────
    players_df = pd.read_parquet(_PLAYERS_PATH)
    print(f"players.parquet loaded — {len(players_df):,} players")

    # ── player_registry.parquet ───────────────────────────────────────────────
    registry_df = pd.read_parquet(_REGISTRY_PATH)

    # ── team_match_records.parquet ────────────────────────────────────────────
    team_records_df = pd.read_parquet(_TEAM_RECORDS_PATH)

    # ── season_batting/bowling.parquet — pre-computed season aggregates ───────
    if os.path.exists(_SEASON_BATTING_PATH):
        season_batting_df = pd.read_parquet(_SEASON_BATTING_PATH)
        print(f"season_batting.parquet loaded — {len(season_batting_df):,} rows")
    else:
        print("WARNING: season_batting.parquet not found — season cap queries disabled")

    if os.path.exists(_SEASON_BOWLING_PATH):
        season_bowling_df = pd.read_parquet(_SEASON_BOWLING_PATH)
        print(f"season_bowling.parquet loaded — {len(season_bowling_df):,} rows")
    else:
        print("WARNING: season_bowling.parquet not found — season cap queries disabled")

    # ── player_index.json — name/alias → unique_name ─────────────────────────
    # The JSON maps lowercase name → cricinfo_id.
    # We need: lowercase name → unique_name (for lookups in players.parquet).
    # Build a cricinfo_id → unique_name reverse map from registry.
    cid_to_unique = {}
    for _, row in registry_df.iterrows():
        cid = row.get("cricinfo_id")
        if cid is not None and not (isinstance(cid, float) and np.isnan(cid)):
            cid_to_unique[int(cid)] = row["unique_name"]

    with open(_PLAYER_INDEX_PATH, "r", encoding="utf-8") as f:
        raw_index = json.load(f)  # lowercase alias → cricinfo_id (int or None)

    # Map: lowercase alias → unique_name
    for alias, cid in raw_index.items():
        if cid is not None:
            uname = cid_to_unique.get(int(cid))
            if uname:
                _player_index[alias] = uname
        # Also store alias → alias directly (covers cases where alias IS unique_name)
        # Will be overwritten by cid-based lookup if both exist — that's fine.

    # Also index directly by unique_name and display_name from registry
    for _, row in registry_df.iterrows():
        uname = row["unique_name"]
        dname = str(row.get("display_name", uname))
        _display_map[uname]            = dname
        _unique_map[dname.lower()]     = uname
        _unique_map[uname.lower()]     = uname
        _player_index[uname.lower()]   = uname
        _player_index[dname.lower()]   = uname
        # Last name shortcut
        parts = dname.lower().split()
        if len(parts) >= 2:
            _player_index.setdefault(parts[-1], uname)
            _player_index.setdefault(parts[0] + " " + parts[-1], uname)

    # ── Player name list for autocomplete ─────────────────────────────────────
    _player_names = sorted(
        set(_display_map[u] for u in players_df["unique_name"].tolist()
            if u in _display_map)
    )

    _loaded = True
    print(f"Data layer ready — {len(_player_names):,} players indexed")


# ── Entity resolution ──────────────────────────────────────────────────────────

def resolve_player(name: str) -> str | None:
    """
    Resolve any name variant → unique_name used in players.parquet.
    Handles: display names, Cricsheet unique names, last names, initials.
    Returns unique_name string or None.
    """
    load()
    key = name.lower().strip()

    # Direct index lookup
    if key in _player_index:
        return _player_index[key]

    # Partial substring match on display names (last resort)
    for dname_lower, uname in _unique_map.items():
        if key in dname_lower:
            return uname

    return None


def resolve_display(unique_name: str) -> str:
    """Return display name for a unique_name, fallback to unique_name itself."""
    return _display_map.get(unique_name, unique_name)


def all_player_names() -> list[str]:
    """Return sorted display names for all players in the dataset."""
    load()
    return _player_names


# ── Player stats ───────────────────────────────────────────────────────────────

def _player_row(unique_name: str):
    """Return the players.parquet row for a unique_name."""
    load()
    rows = players_df[players_df["unique_name"] == unique_name]
    if rows.empty:
        return None
    return rows.iloc[0]


def get_player_stats(name: str, prefix: str = "IPL") -> dict | None:
    """
    Return a clean stats dict for a player.
    prefix = "IPL" | "IPL26" | "T20I" | "2025" | "Overall"
    """
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

    # Get team from registry
    reg_rows = registry_df[registry_df["unique_name"] == unique_name]
    nation   = reg_rows.iloc[0]["nation"] if not reg_rows.empty else ""

    return {
        "name":         resolve_display(unique_name),
        "unique_name":  unique_name,
        "nation":       nation,
        "ipl_ever":     bool(reg_rows.iloc[0]["ipl_ever"]) if not reg_rows.empty else False,

        # batting
        "innings":      g("Innings"),
        "runs":         g("Runs"),
        "balls_faced":  g("Balls_Faced"),
        "dismissed":    g("Dismissed"),
        "avg":          g("Batting_Avg"),
        "sr":           g("Batting_SR"),
        "fours":        g("Fours"),
        "sixes":        g("Sixes"),
        "dot_pct":      g("Dot_Pct"),

        # bowling
        "bowl_innings": g("Bowl_Innings"),
        "wickets":      g("Wickets"),
        "balls_bowled": g("Balls_Bowled"),
        "runs_conceded":g("Runs_Conceded"),
        "economy":      g("Econ"),
        "bowl_avg":     g("Bowling_Avg"),
        "bowl_sr":      g("Bowling_SR"),
    }


# ── Leaderboard queries ────────────────────────────────────────────────────────

def top_run_scorers(n: int = 5, prefix: str = "IPL") -> list[dict]:
    load()
    col = _col(prefix, "Runs")
    df  = players_df.dropna(subset=[col]).sort_values(col, ascending=False).head(n)
    return [
        {
            "rank":    i + 1,
            "player":  resolve_display(r["unique_name"]),
            "runs":    int(r[col]),
            "avg":     _safe(r.get(_col(prefix, "Batting_Avg"))),
            "sr":      _safe(r.get(_col(prefix, "Batting_SR"))),
            "innings": _safe(r.get(_col(prefix, "Innings"))),
        }
        for i, (_, r) in enumerate(df.iterrows())
    ]


def top_wicket_takers(n: int = 5, prefix: str = "IPL") -> list[dict]:
    load()
    col = _col(prefix, "Wickets")
    df  = players_df.dropna(subset=[col]).sort_values(col, ascending=False).head(n)
    return [
        {
            "rank":    i + 1,
            "player":  resolve_display(r["unique_name"]),
            "wickets": int(r[col]),
            "economy": _safe(r.get(_col(prefix, "Econ"))),
            "avg":     _safe(r.get(_col(prefix, "Bowling_Avg"))),
            "sr":      _safe(r.get(_col(prefix, "Bowling_SR"))),
        }
        for i, (_, r) in enumerate(df.iterrows())
    ]


def top_six_hitters(n: int = 5, prefix: str = "IPL") -> list[dict]:
    load()
    col = _col(prefix, "Sixes")
    df  = players_df.dropna(subset=[col]).sort_values(col, ascending=False).head(n)
    return [
        {
            "rank":   i + 1,
            "player": resolve_display(r["unique_name"]),
            "sixes":  int(r[col]),
            "sr":     _safe(r.get(_col(prefix, "Batting_SR"))),
        }
        for i, (_, r) in enumerate(df.iterrows())
    ]


def top_run_scorers_ipl26(n: int = 5) -> list[dict]:
    """Top run scorers in IPL 2026 specifically."""
    return top_run_scorers(n=n, prefix="IPL26")


def top_wicket_takers_ipl26(n: int = 5) -> list[dict]:
    """Top wicket takers in IPL 2026 specifically."""
    return top_wicket_takers(n=n, prefix="IPL26")


def top_form_batters(n: int = 5) -> list[dict]:
    """Top run scorers in 2025 form window."""
    return top_run_scorers(n=n, prefix="2025")


def highest_individual_score(prefix: str = "IPL") -> dict | None:
    """Highest individual score — not available in aggregated parquet.
    Falls back to top run scorer as proxy."""
    load()
    rows = top_run_scorers(n=1, prefix=prefix)
    return rows[0] if rows else None


def ipl_titles_table() -> list[dict]:
    return sorted(
        [{"team": t, "titles": c} for t, c in IPL_TITLES.items()],
        key=lambda x: x["titles"],
        reverse=True,
    )


# ── Points table (hardcoded IPL 2026 — update after each match) ───────────────
# TODO: replace with live scrape or parquet-derived standings when available

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

def get_standings() -> list[dict]:
    return IPL26_STANDINGS


# ── Player comparison ──────────────────────────────────────────────────────────

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
        "player1": s1,
        "player2": s2,
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
    """
    Parse 'compare Kohli vs Rohit' style questions.
    Returns (unique_name1, unique_name2) or (None, None).
    """
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

    # Fallback: find any two known players mentioned
    found = []
    q_lower = question.lower()
    for dname in _player_names:
        if dname.lower() in q_lower and dname not in found:
            uname = resolve_player(dname)
            if uname:
                found.append(uname)
        if len(found) == 2:
            break

    if len(found) == 2:
        return found[0], found[1]

    return None, None


# ── Matchup queries ────────────────────────────────────────────────────────────

def _load_matchup():
    """Load matchup summary lazily (only when needed)."""
    global matchup_df
    if matchup_df is None:
        matchup_df = pd.read_parquet(_MATCHUP_SUMMARY_PATH)
    return matchup_df


def get_matchup(batter_name: str, bowler_name: str,
                competition: str = "Career", phase: str = "ALL") -> dict | None:
    """
    Return head-to-head stats for a batter vs bowler.
    competition: "IPL" | "T20I" | "Career" | "BBL" etc.
    phase: "ALL" | "PP" | "MID" | "DEATH"
    """
    load()
    b_unique  = resolve_player(batter_name)
    bw_unique = resolve_player(bowler_name)
    if not b_unique or not bw_unique:
        return None

    # Try fast per-batter file first
    safe_name  = b_unique.replace(" ", "_").replace("/", "-")
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
        "batter":        resolve_display(b_unique),
        "bowler":        resolve_display(bw_unique),
        "competition":   competition,
        "phase":         phase,
        "balls":         int(r["balls"]),
        "runs":          int(r["runs"]),
        "dismissed":     int(r["dismissed"]),
        "fours":         int(r["fours"]),
        "sixes":         int(r["sixes"]),
        "sr":            _safe(r.get("sr")),
        "dot_pct":       _safe(r.get("dot_pct")),
        "dismiss_rate":  _safe(r.get("dismiss_rate")),
    }


def top_season_run_scorers(season: int, n: int = 5) -> list[dict]:
    """Top run scorers in a specific IPL season — from pre-computed season_batting.parquet."""
    load()
    if season_batting_df is None:
        return []
    try:
        df = season_batting_df[season_batting_df["season"] == season]
        if df.empty:
            return []
        top = df.nlargest(n, "runs")
        return [
            {"rank": i+1, "player": resolve_display(r["player"]) or r["player"], "runs": int(r["runs"])}
            for i, r in enumerate(top.itertuples(index=False))
        ]
    except Exception as e:
        print(f"Season run scorers error: {e}")
        return []


def top_season_wicket_takers(season: int, n: int = 5) -> list[dict]:
    """Top wicket takers in a specific IPL season — from pre-computed season_bowling.parquet."""
    load()
    if season_bowling_df is None:
        return []
    try:
        df = season_bowling_df[season_bowling_df["season"] == season]
        if df.empty:
            return []
        top = df.nlargest(n, "wickets")
        return [
            {"rank": i+1, "player": resolve_display(r["player"]) or r["player"], "wickets": int(r["wickets"])}
            for i, r in enumerate(top.itertuples(index=False))
        ]
    except Exception as e:
        print(f"Season wicket takers error: {e}")
        return []


# ── New leaderboard helpers ────────────────────────────────────────────────────

def _top_by_col(col: str, n: int, ascending: bool = False,
                min_col: str = None, min_val: int = 0,
                extra_cols: list = None) -> list[dict]:
    """Generic leaderboard helper over players_df."""
    load()
    df = players_df.dropna(subset=[col])
    if min_col:
        df = df[df[min_col] >= min_val]
    df = df.sort_values(col, ascending=ascending).head(n)
    rows = []
    for i, (_, r) in enumerate(df.iterrows()):
        row = {"rank": i+1, "player": resolve_display(r["unique_name"]), "value": _safe(r[col])}
        if extra_cols:
            for ec in extra_cols:
                row[ec] = _safe(r.get(ec))
        rows.append(row)
    return rows


# ── Overall (all T20s) leaderboards ───────────────────────────────────────────

def top_overall_run_scorers(n: int = 5) -> list[dict]:
    return top_run_scorers(n=n, prefix="Overall")

def top_overall_wicket_takers(n: int = 5) -> list[dict]:
    return top_wicket_takers(n=n, prefix="Overall")

def top_overall_six_hitters(n: int = 5) -> list[dict]:
    return top_six_hitters(n=n, prefix="Overall")


# ── IPL 2026 quality leaderboards ─────────────────────────────────────────────

def top_ipl26_six_hitters(n: int = 5) -> list[dict]:
    return top_six_hitters(n=n, prefix="IPL26")

def top_ipl26_avg(n: int = 5, min_innings: int = 5) -> list[dict]:
    return _top_by_col("Batting_Avg_IPL26", n, ascending=False,
                       min_col="Innings_IPL26", min_val=min_innings,
                       extra_cols=["Innings_IPL26", "Runs_IPL26"])

def top_ipl26_sr(n: int = 5, min_innings: int = 5) -> list[dict]:
    return _top_by_col("Batting_SR_IPL26", n, ascending=False,
                       min_col="Innings_IPL26", min_val=min_innings,
                       extra_cols=["Innings_IPL26", "Runs_IPL26"])

def top_ipl26_economy(n: int = 5, min_innings: int = 5) -> list[dict]:
    return _top_by_col("Econ_IPL26", n, ascending=True,
                       min_col="Bowl_Innings_IPL26", min_val=min_innings,
                       extra_cols=["Bowl_Innings_IPL26", "Wickets_IPL26"])

def top_ipl26_bowl_avg(n: int = 5, min_wickets: int = 5) -> list[dict]:
    return _top_by_col("Bowling_Avg_IPL26", n, ascending=True,
                       min_col="Wickets_IPL26", min_val=min_wickets,
                       extra_cols=["Wickets_IPL26", "Bowl_Innings_IPL26"])

def top_ipl26_bowl_sr(n: int = 5, min_wickets: int = 5) -> list[dict]:
    return _top_by_col("Bowling_SR_IPL26", n, ascending=True,
                       min_col="Wickets_IPL26", min_val=min_wickets,
                       extra_cols=["Wickets_IPL26"])


# ── Form (2025) bowling leaderboard ───────────────────────────────────────────

def top_form_wicket_takers(n: int = 5) -> list[dict]:
    return top_wicket_takers(n=n, prefix="2025")


# ── T20I quality leaderboards ─────────────────────────────────────────────────

def top_t20i_avg(n: int = 5, min_innings: int = 20) -> list[dict]:
    return _top_by_col("Batting_Avg_T20I", n, ascending=False,
                       min_col="Innings_T20I", min_val=min_innings,
                       extra_cols=["Innings_T20I", "Runs_T20I"])

def top_t20i_sr(n: int = 5, min_innings: int = 20) -> list[dict]:
    return _top_by_col("Batting_SR_T20I", n, ascending=False,
                       min_col="Innings_T20I", min_val=min_innings,
                       extra_cols=["Innings_T20I", "Runs_T20I"])

def top_t20i_economy(n: int = 5, min_innings: int = 20) -> list[dict]:
    return _top_by_col("Econ_T20I", n, ascending=True,
                       min_col="Bowl_Innings_T20I", min_val=min_innings,
                       extra_cols=["Bowl_Innings_T20I", "Wickets_T20I"])

def top_t20i_six_hitters(n: int = 5) -> list[dict]:
    return top_six_hitters(n=n, prefix="T20I")


# ── IPL career misc leaderboards ──────────────────────────────────────────────

def top_dot_pct_bowlers(n: int = 5, min_innings: int = 20) -> list[dict]:
    """Bowlers with highest dot ball percentage in IPL."""
    return _top_by_col("Dot_Pct_IPL", n, ascending=False,
                       min_col="Bowl_Innings_IPL", min_val=min_innings,
                       extra_cols=["Bowl_Innings_IPL", "Wickets_IPL", "Econ_IPL"])

def top_balls_faced(n: int = 5) -> list[dict]:
    """Batters who have faced most balls in IPL."""
    return _top_by_col("Balls_Faced_IPL", n, ascending=False,
                       extra_cols=["Runs_IPL", "Batting_SR_IPL", "Innings_IPL"])


def get_batter_vs_all_bowlers(batter_name: str,
                               competition: str = "IPL",
                               phase: str = "ALL",
                               min_balls: int = 12,
                               n: int = 5) -> dict:
    """
    Return bowlers who trouble a batter most (highest dismiss rate) and
    bowlers a batter dominates (highest SR), for a given competition/phase.
    """
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
