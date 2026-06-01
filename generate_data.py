"""
generate_data.py — Run once locally to generate all pre-computed parquet files.
Usage: python generate_data.py
From: C:\SD\SF360\sportsfan360-ai-agent-main

Generates:
  AskAI_Data/season_batting.parquet   — per player/season/competition/phase batting stats
  AskAI_Data/season_bowling.parquet   — per player/season/competition/phase bowling stats
  AskAI_Data/venue_batting.parquet    — per player/venue batting stats
  AskAI_Data/venue_bowling.parquet    — per player/venue bowling stats
  AskAI_Data/team_vs_team.parquet     — head to head records per team pair/competition
"""

import pandas as pd
import numpy as np
import os

DATA_DIR = "AskAI_Data"

# ── Load bbb_base ──────────────────────────────────────────────────────────────
print("Loading bbb_base.parquet (~1.78M rows)...")
bbb = pd.read_parquet(os.path.join(DATA_DIR, "bbb_base.parquet"))
print(f"  Loaded: {len(bbb):,} rows")

base = bbb[~bbb["is_super_over"]].copy()

# ── Fix Cricsheet season tagging for early IPL seasons ────────────────────────
# Problem 1: IPL 2007/08 is tagged season=2007 → remap to 2008
#            (58 matches, all played Apr-Jun 2008)
# Problem 2: IPL 2009 AND IPL 2010 are BOTH tagged season=2009 → split by date
#            2009 season: Apr-May 2009 (South Africa)
#            2010 season: Mar-Apr 2010 (India) — tagged as 2009 by Cricsheet

base["match_date"] = pd.to_datetime(base["match_date"])

# Fix 1: 2007 → 2008
mask_2007 = (base["competition"] == "IPL") & (base["season"] == 2007)
base.loc[mask_2007, "season"] = 2008
print(f"  Season fix: {mask_2007.sum():,} deliveries remapped 2007→2008")

# Fix 2: Split 2009 by date — matches before Oct 2009 = season 2009, after = season 2010
mask_2009 = (base["competition"] == "IPL") & (base["season"] == 2009)
mask_2009_to_2010 = mask_2009 & (base["match_date"] >= "2009-10-01")
base.loc[mask_2009_to_2010, "season"] = 2010
print(f"  Season fix: {mask_2009_to_2010.sum():,} deliveries remapped 2009→2010 (IPL 2010 season)")
print(f"  Remaining season=2009: {(mask_2009 & ~mask_2009_to_2010).sum():,} deliveries (IPL 2009 SA)")

# Extend with phase='ALL' rows for each delivery
print("Extending with ALL phase...")
base_all = base.copy()
base_all["phase"] = "ALL"
extended = pd.concat([base, base_all], ignore_index=True)
legal = extended[~extended["is_wide"]]
# Pre-filter non-super-over for bowling (avoid re-filtering extended)
extended_bowl = extended  # is_super_over already excluded in base

# ── SEASON BATTING ─────────────────────────────────────────────────────────────
print("Computing season batting stats...")
bat = legal.groupby(["season", "competition", "phase", "batter"]).agg(
    innings     =("match_id",    "nunique"),
    runs        =("runs_batter", "sum"),
    balls_faced =("is_legal",    "sum"),
    dismissed   =("bat_dismissed","sum"),
    fours       =("is_four",     "sum"),
    sixes       =("is_six",      "sum"),
    dots        =("is_dot",      "sum"),
).reset_index().rename(columns={"batter": "player"})

bat["avg"]     = (bat["runs"] / bat["dismissed"].replace(0, np.nan)).round(2)
bat["sr"]      = (bat["runs"] / bat["balls_faced"] * 100).round(2)
bat["dot_pct"] = (bat["dots"] / bat["balls_faced"] * 100).round(2)
bat.drop(columns=["dots"], inplace=True)

bat.to_parquet(os.path.join(DATA_DIR, "season_batting.parquet"), index=False)
print(f"  season_batting: {len(bat):,} rows → saved")

# ── SEASON BOWLING ─────────────────────────────────────────────────────────────
print("Computing season bowling stats...")
bowl = extended_bowl.groupby(
    ["season", "competition", "phase", "bowler"]
).agg(
    bowl_innings  =("match_id",     "nunique"),
    balls_bowled  =("is_legal",     "sum"),
    runs_conceded =("runs_total",   "sum"),
    wickets       =("bowl_wicket",  "sum"),
    dots          =("is_dot",       "sum"),
).reset_index().rename(columns={"bowler": "player"})

bowl["economy"]  = (bowl["runs_conceded"] / bowl["balls_bowled"] * 6).round(2)
bowl["bowl_avg"] = (bowl["runs_conceded"] / bowl["wickets"].replace(0, np.nan)).round(2)
bowl["bowl_sr"]  = (bowl["balls_bowled"]  / bowl["wickets"].replace(0, np.nan)).round(2)
bowl["dot_pct"]  = (bowl["dots"] / bowl["balls_bowled"] * 100).round(2)
bowl.drop(columns=["dots"], inplace=True)

bowl.to_parquet(os.path.join(DATA_DIR, "season_bowling.parquet"), index=False)
print(f"  season_bowling: {len(bowl):,} rows → saved")

# ── VENUE BATTING ──────────────────────────────────────────────────────────────
print("Computing venue batting stats...")
legal_base = base[~base["is_wide"]]
vbat = legal_base.groupby(["venue", "batter"]).agg(
    innings     =("match_id",     "nunique"),
    runs        =("runs_batter",  "sum"),
    balls_faced =("is_legal",     "sum"),
    dismissed   =("bat_dismissed","sum"),
    fours       =("is_four",      "sum"),
    sixes       =("is_six",       "sum"),
).reset_index().rename(columns={"batter": "player"})

vbat["avg"] = (vbat["runs"] / vbat["dismissed"].replace(0, np.nan)).round(2)
vbat["sr"]  = (vbat["runs"] / vbat["balls_faced"] * 100).round(2)

vbat.to_parquet(os.path.join(DATA_DIR, "venue_batting.parquet"), index=False)
print(f"  venue_batting: {len(vbat):,} rows → saved")

# ── VENUE BOWLING ──────────────────────────────────────────────────────────────
print("Computing venue bowling stats...")
vbowl = base.groupby(["venue", "bowler"]).agg(
    bowl_innings  =("match_id",    "nunique"),
    balls_bowled  =("is_legal",    "sum"),
    runs_conceded =("runs_total",  "sum"),
    wickets       =("bowl_wicket", "sum"),
).reset_index().rename(columns={"bowler": "player"})

vbowl["economy"]  = (vbowl["runs_conceded"] / vbowl["balls_bowled"] * 6).round(2)
vbowl["bowl_avg"] = (vbowl["runs_conceded"] / vbowl["wickets"].replace(0, np.nan)).round(2)

vbowl.to_parquet(os.path.join(DATA_DIR, "venue_bowling.parquet"), index=False)
print(f"  venue_bowling: {len(vbowl):,} rows → saved")

# ── TEAM VS TEAM ───────────────────────────────────────────────────────────────
print("Computing team vs team records...")
tmr = pd.read_parquet(os.path.join(DATA_DIR, "team_match_records.parquet"))

# Each match has two rows (one per team). Use won=True/False to determine winner.
# Build pair key so MI vs CSK == CSK vs MI
def make_pair(row):
    teams = sorted([row["team"], row["opponent"]])
    return teams[0], teams[1]

tmr[["team_a", "team_b"]] = tmr.apply(
    make_pair, axis=1, result_type="expand"
)

tvt = tmr.groupby(["team_a", "team_b", "competition", "season"]).agg(
    matches    =("match_id", "nunique"),
    team_a_wins=(
        "won",
        lambda x: tmr.loc[x.index[tmr.loc[x.index, "won"] == True],
                          "team"].eq(tmr.loc[x.index[tmr.loc[x.index, "won"] == True],
                                             "team_a"].values[0]
                                     if len(x.index[tmr.loc[x.index, "won"] == True]) > 0
                                     else "").sum()
    ),
).reset_index()

# Simpler approach — count wins per team then join
wins = tmr[tmr["won"] == True].groupby(
    ["team_a", "team_b", "competition", "season", "team"]
).size().reset_index(name="wins")

match_counts = tmr.groupby(
    ["team_a", "team_b", "competition", "season"]
)["match_id"].nunique().reset_index(name="matches")

# Build clean team_vs_team from match records directly
records = []
grouped = tmr.groupby(["team_a", "team_b", "competition"])

for (ta, tb, comp), grp in grouped:
    total    = grp["match_id"].nunique()
    ta_wins  = grp[grp["won"] == True]["team"].eq(ta).sum()
    tb_wins  = grp[grp["won"] == True]["team"].eq(tb).sum()
    no_result= total - ta_wins - tb_wins
    records.append({
        "team_a":     ta,
        "team_b":     tb,
        "competition":comp,
        "matches":    total,
        "team_a_wins":int(ta_wins),
        "team_b_wins":int(tb_wins),
        "no_result":  int(no_result),
    })

# Also per-season breakdown
records_season = []
grouped_s = tmr.groupby(["team_a", "team_b", "competition", "season"])

for (ta, tb, comp, season), grp in grouped_s:
    total   = grp["match_id"].nunique()
    ta_wins = grp[grp["won"] == True]["team"].eq(ta).sum()
    tb_wins = grp[grp["won"] == True]["team"].eq(tb).sum()
    records_season.append({
        "team_a":     ta,
        "team_b":     tb,
        "competition":comp,
        "season":     season,
        "matches":    total,
        "team_a_wins":int(ta_wins),
        "team_b_wins":int(tb_wins),
        "no_result":  int(total - ta_wins - tb_wins),
    })

tvt_all    = pd.DataFrame(records)
tvt_season = pd.DataFrame(records_season)

tvt_all.to_parquet(os.path.join(DATA_DIR, "team_vs_team.parquet"), index=False)
tvt_season.to_parquet(os.path.join(DATA_DIR, "team_vs_team_season.parquet"), index=False)
print(f"  team_vs_team: {len(tvt_all):,} rows → saved")
print(f"  team_vs_team_season: {len(tvt_season):,} rows → saved")

# ── SUMMARY ───────────────────────────────────────────────────────────────────
print()
print("=" * 50)
print("All files generated successfully:")
for fname in ["season_batting.parquet", "season_bowling.parquet",
              "venue_batting.parquet", "venue_bowling.parquet",
              "team_vs_team.parquet", "team_vs_team_season.parquet"]:
    path = os.path.join(DATA_DIR, fname)
    size = os.path.getsize(path) / 1024
    print(f"  {fname:40s} {size:8.0f} KB")

# Verify early IPL seasons
print()
print("Season verification (IPL):")
bat_check  = pd.read_parquet(os.path.join(DATA_DIR, "season_batting.parquet"))
bowl_check = pd.read_parquet(os.path.join(DATA_DIR, "season_bowling.parquet"))
ipl_bat  = bat_check[(bat_check["competition"]=="IPL") & (bat_check["phase"]=="ALL")]
ipl_bowl = bowl_check[(bowl_check["competition"]=="IPL") & (bowl_check["phase"]=="ALL")]
for yr in [2007, 2008, 2009, 2010, 2011, 2025, 2026]:
    b  = len(ipl_bat[ipl_bat["season"]==yr])
    bw = len(ipl_bowl[ipl_bowl["season"]==yr])
    top_bat  = ipl_bat[ipl_bat["season"]==yr].nlargest(1,"runs")["player"].values
    top_bowl = ipl_bowl[ipl_bowl["season"]==yr].nlargest(1,"wickets")["player"].values
    top_bat_str  = f" → top bat: {top_bat[0]}"  if len(top_bat)  else ""
    top_bowl_str = f" → top bowl: {top_bowl[0]}" if len(top_bowl) else ""
    print(f"  {yr}: bat={b} rows, bowl={bw} rows{top_bat_str}{top_bowl_str}")
print()
print("Next: git add AskAI_Data/*.parquet && git commit -m 'Regenerate: fix 2008 season year + 2010 bowling' && git push")
