SF360 AskAI — Data Layer
============================

This folder contains the cricket data files for the AskAI agent.

FILES
-----
  players.parquet              Per-player career and season totals.

  player_registry.parquet      Player registry used for name resolution.

  season_batting.parquet       Per-season batting aggregates by
                               competition.

  season_bowling.parquet       Per-season bowling aggregates by
                               competition.

  team_vs_team.parquet         Team head-to-head records across
                               all competitions (all-time).

  team_vs_team_season.parquet  Team head-to-head records per season.

  team_match_records.parquet   Team match records table.

  matchup/summary.parquet      Batter vs bowler matchup table.

  player_index.json            Name / alias → unique_name mapping
                               for entity resolution.

NOTES
-----
- Drop this folder into the agent project (replacing the existing
  AskAI_Data/ folder if any).
- The agent's data_loader.py reads these files by name from
  AskAI_Data/.
- This snapshot is static. To refresh it with newer match data,
  a Cricsheet → parquet pipeline will need to be set up separately.
