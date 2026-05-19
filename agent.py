import os
import re
import json
import random

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq

import data_loader as dl
from memory_store import save_context, get_context
from feed_engine import get_feed
from teams_engine import get_teams
from players_engine import get_players
from matches_engine import get_matches
from live_matches import get_live_matches

# ── App setup ──────────────────────────────────────────────────────────────────
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

_MODEL_TOOL = "llama-3.3-70b-versatile"

# Load data at startup
try:
    dl.load()
except Exception as e:
    print("Data layer load failed at startup:", e)


# ── Tool definitions for Groq ──────────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_career_leaderboard",
            "description": (
                "Get the top N players by a career stat. Use for questions like "
                "'most IPL runs', 'best T20I economy', 'most Overall sixes', "
                "'best IPL26 bowling average', 'top form batters 2025'. "
                "prefix options: IPL | T20I | IPL26 | Overall | 2025"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "stat": {
                        "type": "string",
                        "enum": ["runs","wickets","sixes","fours","avg","sr",
                                 "economy","bowl_avg","bowl_sr","dot_pct","balls_faced"],
                        "description": "The stat to rank by"
                    },
                    "prefix": {
                        "type": "string",
                        "enum": ["IPL","T20I","IPL26","Overall","2025"],
                        "description": "Competition/period context"
                    },
                    "n": {
                        "type": "integer",
                        "description": "Number of results (default 5)",
                        "default": 5
                    }
                },
                "required": ["stat", "prefix"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_season_leaderboard",
            "description": (
                "Get the top N players by a stat in a specific season and competition. "
                "Use for questions like 'most runs in IPL 2019', 'best economy in IPL 2023 powerplay', "
                "'Purple Cap IPL 2020', 'Orange Cap 2016', 'best SR in T20I 2024'. "
                "phase options: ALL | PP | MID | DEATH"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "stat": {
                        "type": "string",
                        "enum": ["runs","wickets","avg","sr","economy","bowl_avg","bowl_sr",
                                 "sixes","fours","dot_pct"],
                        "description": "The stat to rank by"
                    },
                    "season": {
                        "type": "integer",
                        "description": "Year e.g. 2016, 2023"
                    },
                    "competition": {
                        "type": "string",
                        "description": "e.g. IPL, T20I, BBL, PSL, SA20. Default IPL",
                        "default": "IPL"
                    },
                    "phase": {
                        "type": "string",
                        "enum": ["ALL","PP","MID","DEATH"],
                        "description": "Match phase. Default ALL",
                        "default": "ALL"
                    },
                    "n": {
                        "type": "integer",
                        "description": "Number of results (default 5)",
                        "default": 5
                    }
                },
                "required": ["stat", "season"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_player_career_stats",
            "description": (
                "Get career stats for a specific named player. "
                "Use for 'Kohli's IPL stats', 'Bumrah T20I record', 'Rohit's IPL26 numbers'. "
                "prefix options: IPL | T20I | IPL26 | Overall | 2025"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "player": {"type": "string", "description": "Player name"},
                    "prefix": {
                        "type": "string",
                        "enum": ["IPL","T20I","IPL26","Overall","2025"],
                        "description": "Competition context. Default IPL",
                        "default": "IPL"
                    }
                },
                "required": ["player"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_player_season_stats",
            "description": (
                "Get stats for a specific player in a specific season. "
                "Use for 'Kohli's runs in IPL 2016', 'Bumrah's economy in IPL 2020 death overs', "
                "'Narine's wickets in IPL 2024 powerplay'. "
                "phase options: ALL | PP | MID | DEATH"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "player":      {"type": "string"},
                    "season":      {"type": "integer", "description": "Year e.g. 2016"},
                    "competition": {"type": "string", "default": "IPL"},
                    "phase":       {"type": "string", "enum": ["ALL","PP","MID","DEATH"], "default": "ALL"}
                },
                "required": ["player", "season"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_matchup",
            "description": (
                "Get head-to-head stats between a batter and a bowler. "
                "Use for 'Kohli vs Narine', 'how does Rohit bat against Bumrah', "
                "'Narine vs Kohli in powerplay'. "
                "competition options: IPL | T20I | Career | BBL | PSL etc. "
                "phase options: ALL | PP | MID | DEATH"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "batter":      {"type": "string"},
                    "bowler":      {"type": "string"},
                    "competition": {"type": "string", "default": "IPL"},
                    "phase":       {"type": "string", "enum": ["ALL","PP","MID","DEATH"], "default": "ALL"}
                },
                "required": ["batter", "bowler"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_batter_weaknesses",
            "description": (
                "Find which bowlers trouble a batter most, or which bowlers a batter dominates. "
                "Use for 'who troubles Kohli', 'Rohit's nemesis', 'who does Warner struggle against', "
                "'which bowlers does Dhoni dominate'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "batter":      {"type": "string"},
                    "competition": {"type": "string", "default": "IPL"},
                    "phase":       {"type": "string", "enum": ["ALL","PP","MID","DEATH"], "default": "ALL"}
                },
                "required": ["batter"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_venue_player_stats",
            "description": (
                "Get a player's stats at a specific venue. "
                "Use for 'Kohli at Wankhede', 'Bumrah at Chepauk', "
                "'Rohit's record at Eden Gardens'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "player": {"type": "string"},
                    "venue":  {"type": "string", "description": "Partial venue name e.g. 'Wankhede', 'Chepauk', 'Eden'"}
                },
                "required": ["player", "venue"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_venue_leaderboard",
            "description": (
                "Get the best batters or bowlers at a specific venue. "
                "Use for 'best batters at Wankhede', 'top wicket takers at Chepauk', "
                "'who scores most at Eden Gardens'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "venue": {"type": "string", "description": "Partial venue name"},
                    "stat":  {
                        "type": "string",
                        "enum": ["runs","avg","sr","sixes","wickets","economy","bowl_avg"],
                        "description": "Stat to rank by. Default runs",
                        "default": "runs"
                    },
                    "n": {"type": "integer", "default": 5}
                },
                "required": ["venue"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_team_vs_team",
            "description": (
                "Get head-to-head win/loss record between exactly two named IPL teams. "
                "ONLY use when the question explicitly names two teams and asks for their record, "
                "e.g. 'MI vs CSK head to head', 'KKR vs RCB all time', 'RR vs SRH in 2023'. "
                "Do NOT use for questions about who won a tournament, championship, or title — "
                "use get_titles for that. Do NOT use when only one team is mentioned."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "team1":       {"type": "string", "description": "Team name or abbreviation e.g. MI, CSK, RCB"},
                    "team2":       {"type": "string"},
                    "competition": {"type": "string", "default": "IPL"},
                    "season":      {"type": "integer", "description": "Optional — specific year"}
                },
                "required": ["team1", "team2"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_compare_players",
            "description": (
                "Compare two players head-to-head across career stats. "
                "Use for 'compare Kohli vs Rohit', 'who is better Bumrah or Narine'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "player1": {"type": "string"},
                    "player2": {"type": "string"},
                    "prefix":  {
                        "type": "string",
                        "enum": ["IPL","T20I","IPL26","Overall","2025"],
                        "default": "IPL"
                    }
                },
                "required": ["player1", "player2"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_standings",
            "description": "Get the IPL 2026 points table / standings.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_titles",
            "description": (
                "Get IPL title/championship count by team — all-time history. "
                "Use for 'who won the first IPL', 'which team has most titles', "
                "'IPL champions list', 'who won IPL 2008', 'IPL trophy winners', "
                "'most successful team', 'how many titles has MI won'."
            ),
            "parameters": {"type": "object", "properties": {}}
        }
    },
]


# ── Tool executor ──────────────────────────────────────────────────────────────

def _execute_tool(name: str, args: dict) -> dict:
    """Call the appropriate data_loader function and return structured result."""
    # Groq sometimes sends n as a string — coerce to int defensively
    if "n" in args:
        try:
            args["n"] = int(args["n"])
        except (ValueError, TypeError):
            args["n"] = 5
    try:
        if name == "get_career_leaderboard":
            n = args.get("n", 5)
            rows = dl.get_career_leaderboard(
                stat=args["stat"],
                prefix=args.get("prefix", "IPL"),
                n=n
            )
            result = {"rows": rows, "stat": args["stat"], "prefix": args.get("prefix","IPL")}
            if n == 1 and rows:
                result["winner"] = rows[0]
            return result

        elif name == "get_season_leaderboard":
            n = args.get("n", 5)
            rows = dl.get_season_leaderboard(
                stat=args["stat"],
                season=args["season"],
                competition=args.get("competition", "IPL"),
                phase=args.get("phase", "ALL"),
                n=n
            )
            result = {"rows": rows, "stat": args["stat"], "season": args["season"],
                    "competition": args.get("competition","IPL"),
                    "phase": args.get("phase","ALL")}
            if n == 1 and rows:
                result["winner"] = rows[0]
            return result

        elif name == "get_player_career_stats":
            stats = dl.get_player_stats(args["player"], args.get("prefix","IPL"))
            if not stats:
                return {"error": f"Player not found: {args['player']}"}
            return {"stats": stats, "prefix": args.get("prefix","IPL")}

        elif name == "get_player_season_stats":
            stats = dl.get_season_player_stats(
                args["player"], args["season"],
                args.get("competition","IPL"), args.get("phase","ALL")
            )
            if not stats:
                return {"error": f"No data found for {args['player']} in {args.get('competition','IPL')} {args['season']}"}
            return {"stats": stats}

        elif name == "get_matchup":
            # Try both batter/bowler orientations
            result = dl.get_matchup(
                args["batter"], args["bowler"],
                args.get("competition","IPL"), args.get("phase","ALL")
            )
            if not result:
                # Try swapped
                result = dl.get_matchup(
                    args["bowler"], args["batter"],
                    args.get("competition","IPL"), args.get("phase","ALL")
                )
            if not result:
                # Try Career
                result = dl.get_matchup(
                    args["batter"], args["bowler"], "Career", args.get("phase","ALL")
                )
            if not result:
                return {"error": f"No matchup data found for {args['batter']} vs {args['bowler']}"}
            return {"matchup": result}

        elif name == "get_batter_weaknesses":
            result = dl.get_batter_vs_all_bowlers(
                args["batter"],
                args.get("competition","IPL"),
                args.get("phase","ALL")
            )
            return result  # already returns error key if not found

        elif name == "get_venue_player_stats":
            result = dl.get_venue_stats(args["player"], args["venue"])
            if not result:
                return {"error": f"No venue data found for {args['player']} at {args['venue']}"}
            return result

        elif name == "get_venue_leaderboard":
            rows = dl.get_venue_leaderboard(
                args["venue"], args.get("stat","runs"), args.get("n",5)
            )
            return {"rows": rows, "venue": args["venue"], "stat": args.get("stat","runs")}

        elif name == "get_team_vs_team":
            result = dl.get_team_vs_team(
                args["team1"], args["team2"],
                args.get("competition","IPL"),
                args.get("season")
            )
            if not result:
                return {"error": f"No head-to-head data found for {args['team1']} vs {args['team2']}"}
            return result

        elif name == "get_compare_players":
            result = dl.compare_players(args["player1"], args["player2"], args.get("prefix","IPL"))
            return result

        elif name == "get_standings":
            return {"standings": dl.get_standings()}

        elif name == "get_titles":
            return {"titles": dl.ipl_titles_table()}

        else:
            return {"error": f"Unknown tool: {name}"}

    except Exception as e:
        return {"error": str(e)}


# ── Chart data extractor ───────────────────────────────────────────────────────

def _format_tool_result_readable(result: dict) -> str:
    """Convert tool result to a human-readable string Groq can directly copy from."""
    if "error" in result:
        return f"ERROR: {result['error']}"

    if "rows" in result and result["rows"]:
        rows  = result["rows"]
        stat  = result.get("stat","value")
        season= result.get("season","")
        comp  = result.get("competition","")
        phase = result.get("phase","ALL")
        prefix= result.get("prefix","")
        ctx   = f"{comp} {season} {prefix}".strip() + (f" ({phase})" if phase != "ALL" else "")
        lines = [f"Top {len(rows)} by {stat} — {ctx}:"]
        for r in rows:
            lines.append(f"  {r['rank']}. {r['player']} — {r['value']} {stat}")
        if "winner" in result:
            w = result["winner"]
            lines.append(f"WINNER: {w['player']} with {w['value']} {stat}")
        return "\n".join(lines)

    if "matchup" in result:
        m = result["matchup"]
        return (
            f"MATCHUP — {m['batter']} vs {m['bowler']} in {m['competition']} {m['phase']}:\n"
            f"  Balls: {m['balls']}, Runs: {m['runs']}, Dismissed: {m['dismissed']}, "
            f"SR: {m.get('sr','N/A')}, Dot%: {m.get('dot_pct','N/A')}, "
            f"Dismiss rate: {m.get('dismiss_rate','N/A')}"
        )

    if "weak_against" in result:
        batter = result.get("batter","")
        lines  = [f"Bowlers who trouble {batter} most ({result.get('competition','')} {result.get('phase','')}):"]
        for r in result.get("weak_against", []):
            lines.append(f"  {r['bowler_display']} — {r['balls']} balls, {r['dismissed']} dismissals, SR {r['sr']}")
        lines.append(f"\nBowlers {batter} dominates:")
        for r in result.get("dominates", []):
            lines.append(f"  {r['bowler_display']} — {r['balls']} balls, {r['dismissed']} dismissals, SR {r['sr']}")
        return "\n".join(lines)

    if "stats" in result:
        s   = result["stats"]
        pfx = result.get("prefix","")
        lines = [f"PLAYER STATS — {s.get('name','')} ({pfx}):"]
        for k, v in s.items():
            if k not in ("name","unique_name","nation","ipl_ever") and v is not None:
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)

    if "standings" in result:
        lines = ["IPL 2026 STANDINGS:"]
        for r in result["standings"]:
            lines.append(f"  {r['position']}. {r['team']} — {r['points']} pts, NRR {r['nrr']}")
        return "\n".join(lines)

    if "titles" in result:
        lines = ["IPL TITLES:"]
        for r in result["titles"]:
            if r["titles"] > 0:
                lines.append(f"  {r['team']}: {r['titles']} title(s)")
        return "\n".join(lines)

    if "team1" in result:
        return (
            f"HEAD TO HEAD — {result['team1']} vs {result['team2']} ({result.get('competition','')}):\n"
            f"  Total matches: {result['matches']}\n"
            f"  {result['team1']} wins: {result['team1_wins']}\n"
            f"  {result['team2']} wins: {result['team2_wins']}\n"
            f"  No result: {result.get('no_result',0)}"
        )

    if "batting" in result or "bowling" in result:
        name  = result.get("name","")
        venue = result.get("venue_query","")
        lines = [f"VENUE STATS — {name} at {venue}:"]
        if "batting" in result:
            b = result["batting"]
            lines.append(f"  Batting: {b.get('innings',0)} innings, {b.get('runs',0)} runs, avg {b.get('avg','N/A')}, SR {b.get('sr','N/A')}")
        if "bowling" in result:
            b = result["bowling"]
            lines.append(f"  Bowling: {b.get('wickets',0)} wickets, econ {b.get('economy','N/A')}, avg {b.get('bowl_avg','N/A')}")
        return "\n".join(lines)

    # Fallback
    return json.dumps(result)


def _fmt(val):
    """Format a number cleanly — int if whole, 2dp if decimal."""
    if val is None:
        return "N/A"
    try:
        f = float(val)
        return str(int(f)) if f == int(f) else f"{f:.2f}"
    except:
        return str(val)

def _build_answer(question: str, tool_results: list) -> str:
    """
    Build a clean answer string directly from tool results — zero Groq involvement.
    This is the definitive fix for hallucination: Python formats, not LLM.
    """
    parts = []

    for tr in tool_results:
        r    = tr["result"]
        name = tr["tool_name"]

        if "error" in r:
            parts.append(f"I don't have data for that: {r['error']}")
            continue

        # ── Leaderboard ───────────────────────────────────────────────────────
        if "rows" in r and r["rows"]:
            rows   = r["rows"]
            stat   = r.get("stat", "value")
            season = r.get("season", "")
            comp   = r.get("competition", "")
            phase  = r.get("phase", "ALL")
            prefix = r.get("prefix", "")

            ctx_parts = [x for x in [comp, str(season) if season else "", prefix] if x]
            ctx = " ".join(ctx_parts)
            if phase and phase != "ALL":
                ctx += f" ({phase} overs)"

            stat_label = {
                "runs": "runs", "wickets": "wickets", "sixes": "sixes",
                "fours": "fours", "avg": "batting avg", "sr": "strike rate",
                "economy": "economy", "bowl_avg": "bowling avg",
                "bowl_sr": "bowling SR", "dot_pct": "dot ball %",
                "balls_faced": "balls faced",
            }.get(stat, stat)

            if "winner" in r:
                w = r["winner"]
                parts.append(f"**{w['player']}** won the {stat_label} title in **{ctx}** with **{_fmt(w['value'])} {stat_label}**.")
            else:
                lines = [f"Top {len(rows)} by {stat_label} — **{ctx}**:"]
                for row in rows:
                    lines.append(f"{row['rank']}. **{row['player']}** — {_fmt(row['value'])} {stat_label}")
                parts.append("\n".join(lines))

        # ── Matchup ───────────────────────────────────────────────────────────
        elif "matchup" in r:
            m = r["matchup"]
            phase_str = f" ({m['phase']} overs)" if m.get("phase") and m["phase"] != "ALL" else ""
            parts.append(
                f"**{m['batter']} vs {m['bowler']}** in {m['competition']}{phase_str}:\n"
                f"- **Balls faced:** {m['balls']}\n"
                f"- **Runs scored:** {m['runs']}\n"
                f"- **Dismissals:** {m['dismissed']}\n"
                f"- **Strike rate:** {_fmt(m.get('sr'))}\n"
                f"- **Dot ball %:** {_fmt(m.get('dot_pct'))}%\n"
                f"- **Dismissal rate:** every {_fmt(m.get('dismiss_rate'))} balls"
            )

        # ── Batter weaknesses ─────────────────────────────────────────────────
        elif "weak_against" in r:
            batter = r.get("batter", "")
            comp   = r.get("competition", "IPL")
            phase  = r.get("phase", "ALL")
            phase_str = f" ({phase} overs)" if phase != "ALL" else ""

            weak = r.get("weak_against", [])
            dom  = r.get("dominates", [])

            lines = [f"**{batter}** in {comp}{phase_str}:\n"]
            if weak:
                lines.append("🎯 **Bowlers who trouble him most:**")
                for w in weak:
                    lines.append(f"  - **{w['bowler_display']}** — {w['balls']} balls, {w['dismissed']} dismissals, SR {_fmt(w['sr'])}")
            if dom:
                lines.append("\n💪 **Bowlers he dominates:**")
                for w in dom:
                    lines.append(f"  - **{w['bowler_display']}** — {w['balls']} balls, {w['dismissed']} dismissals, SR {_fmt(w['sr'])}")
            parts.append("\n".join(lines))

        # ── Player career stats ───────────────────────────────────────────────
        elif "stats" in r:
            s   = r["stats"]
            pfx = r.get("prefix", "IPL")
            season = s.get("season")
            comp   = s.get("competition", "")
            phase  = s.get("phase", "ALL")

            ctx = f"{comp} {season}" if season else pfx
            if phase and phase != "ALL":
                ctx += f" ({phase})"

            name_str = s.get("name", "")
            lines = [f"**{name_str}** — {ctx} stats:"]

            # Batting
            if s.get("runs") is not None:
                lines.append(
                    f"🏏 **Batting:** {s.get('innings') or 0} innings, "
                    f"**{_fmt(s.get('runs'))} runs**, "
                    f"avg {_fmt(s.get('avg'))}, SR {_fmt(s.get('sr'))}, "
                    f"{s.get('sixes') or 0} sixes"
                )
            # Bowling
            if s.get("wickets") is not None:
                lines.append(
                    f"🎳 **Bowling:** {s.get('bowl_innings') or 0} innings, "
                    f"**{_fmt(s.get('wickets'))} wickets**, "
                    f"econ {_fmt(s.get('economy'))}, avg {_fmt(s.get('bowl_avg'))}"
                )
            parts.append("\n".join(lines))

        # ── Standings ─────────────────────────────────────────────────────────
        elif "standings" in r:
            lines = ["**IPL 2026 Points Table:**\n"]
            for row in r["standings"]:
                lines.append(
                    f"{row['position']}. **{row['team']}** — "
                    f"{row['points']} pts  (W{row['won']} L{row['lost']}, NRR {row['nrr']:+.3f})"
                )
            parts.append("\n".join(lines))

        # ── Titles ────────────────────────────────────────────────────────────
        elif "titles" in r:
            winners = [t for t in r["titles"] if t["titles"] > 0]
            lines   = ["**IPL Title Count:**\n"]
            for t in winners:
                lines.append(f"- **{t['team']}**: {t['titles']} 🏆")
            parts.append("\n".join(lines))

        # ── Team vs Team ──────────────────────────────────────────────────────
        elif "team1" in r:
            t1, t2 = r["team1"], r["team2"]
            comp   = r.get("competition", "IPL")
            season = r.get("season")
            ctx    = f"{comp} {season}" if season else f"all-time {comp}"
            parts.append(
                f"**{t1} vs {t2}** — {ctx}:\n"
                f"- Total matches: **{r['matches']}**\n"
                f"- **{t1}** wins: **{r['team1_wins']}**\n"
                f"- **{t2}** wins: **{r['team2_wins']}**\n"
                f"- No result: {r.get('no_result', 0)}"
            )

        # ── Venue stats ───────────────────────────────────────────────────────
        elif "batting" in r or "bowling" in r:
            name_str  = r.get("name", "")
            venue_str = r.get("venue_query", "")
            lines = [f"**{name_str}** at **{venue_str}**:"]
            if "batting" in r:
                b = r["batting"]
                lines.append(
                    f"🏏 **Batting:** {b.get('innings',0)} innings, "
                    f"**{_fmt(b.get('runs'))} runs**, "
                    f"avg {_fmt(b.get('avg'))}, SR {_fmt(b.get('sr'))}"
                )
            if "bowling" in r:
                b = r["bowling"]
                lines.append(
                    f"🎳 **Bowling:** **{_fmt(b.get('wickets'))} wickets**, "
                    f"econ {_fmt(b.get('economy'))}, avg {_fmt(b.get('bowl_avg'))}"
                )
            parts.append("\n".join(lines))

        # ── Venue leaderboard ─────────────────────────────────────────────────
        elif "venue" in r and "rows" in r:
            rows  = r["rows"]
            venue = r.get("venue", "")
            stat  = r.get("stat", "value")
            lines = [f"Top {len(rows)} by {stat} at **{venue}**:"]
            for row in rows:
                lines.append(f"{row['rank']}. **{row['player']}** — {_fmt(row['value'])} {stat}")
            parts.append("\n".join(lines))

        # ── Compare players ───────────────────────────────────────────────────
        elif "player1" in r and "player2" in r:
            s1 = r["player1"]
            s2 = r["player2"]
            e  = r.get("edges", {})
            lines = [f"**{s1['name']} vs {s2['name']}** — IPL career:\n"]
            lines.append(f"| Stat | {s1['name']} | {s2['name']} |")
            lines.append("|------|------|------|")
            for stat, label in [("runs","Runs"),("avg","Bat Avg"),("sr","Bat SR"),
                                  ("sixes","Sixes"),("wickets","Wickets"),("economy","Economy")]:
                v1 = _fmt(s1.get(stat))
                v2 = _fmt(s2.get(stat))
                lines.append(f"| {label} | {v1} | {v2} |")
            parts.append("\n".join(lines))

    return "\n\n".join(parts) if parts else "I don't have data for that."


def _extract_chart(tool_name: str, tool_result: dict) -> tuple[str, list]:
    """Build chart_title and chart_data from tool result."""
    if "rows" in tool_result and tool_result["rows"]:
        rows  = tool_result["rows"]
        stat  = tool_result.get("stat", "value")
        title = f"Top {len(rows)} — {stat}"
        data  = [{"player": r["player"], "value": r["value"]} for r in rows]
        return title, data

    if "matchup" in tool_result:
        m = tool_result["matchup"]
        title = f"{m['batter']} vs {m['bowler']} — {m['competition']} {m['phase']}"
        data  = [
            {"player": "Balls",      "value": m["balls"]},
            {"player": "Runs",       "value": m["runs"]},
            {"player": "Dismissals", "value": m["dismissed"]},
            {"player": "Fours",      "value": m["fours"]},
            {"player": "Sixes",      "value": m["sixes"]},
        ]
        return title, data

    if "weak_against" in tool_result:
        weak  = tool_result["weak_against"]
        batter= tool_result.get("batter","")
        title = f"Bowlers who trouble {batter} most"
        data  = [{"player": r["bowler_display"], "value": round(r["dismiss_rate"],1)} for r in weak]
        return title, data

    if "standings" in tool_result:
        rows  = tool_result["standings"]
        title = "IPL 2026 points table"
        data  = [{"player": r["team"], "value": r["points"]} for r in rows]
        return title, data

    if "titles" in tool_result:
        rows  = [t for t in tool_result["titles"] if t["titles"] > 0]
        title = "IPL titles by team"
        data  = [{"player": r["team"], "value": r["titles"]} for r in rows]
        return title, data

    if "player1" in tool_result and "player2" in tool_result:
        s1    = tool_result["player1"]
        s2    = tool_result["player2"]
        title = f"{s1['name']} vs {s2['name']}"
        data  = [
            {"player": s1["name"], "metric": "Runs",    "value": s1.get("runs") or 0},
            {"player": s2["name"], "metric": "Runs",    "value": s2.get("runs") or 0},
            {"player": s1["name"], "metric": "Wickets", "value": s1.get("wickets") or 0},
            {"player": s2["name"], "metric": "Wickets", "value": s2.get("wickets") or 0},
        ]
        return title, data

    return "", []


# ── System prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are AskSportsFan360, a cricket analytics assistant backed by a real ball-by-ball database.

ABSOLUTE RULES — these override everything:
1. You MUST ONLY report numbers and names that appear VERBATIM in the tool result. Copy them exactly.
2. You MUST NOT use your training knowledge for any statistic, ranking, or player name.
3. You MUST NOT reorder, modify, or supplement the tool result with recalled facts.
4. If tool result has rows, report them IN THAT EXACT ORDER with THOSE EXACT NUMBERS.
5. If tool returns empty or error, say "I don't have data for that" — never fill the gap.
6. For knowledge questions (rules, history, trivia) with no tool result, answer freely but say "Note: general knowledge, not from live database."
7. Keep answers concise. Bold key numbers and names with **.
8. For matchup answers always state the number of balls — context matters for small samples.
"""


# ── Main /ask route ────────────────────────────────────────────────────────────

@app.get("/ask")
def ask(question: str):
    if not dl._loaded:
        try:
            dl.load()
        except Exception as e:
            return {"answer": "Data is loading, please try again.", "chart_title": "", "chart_data": []}

    chart_title = ""
    chart_data  = []

    # Build conversation history for context
    context_turns = get_context()
    history = []
    for t in context_turns:
        history.append({"role": "user",      "content": t["question"]})
        history.append({"role": "assistant", "content": t["answer"]})

    messages = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + history
        + [{"role": "user", "content": question}]
    )

    # ── Round 1: Let Groq decide which tool(s) to call ────────────────────────
    # Retry up to 2 times — llama-3.3-70b occasionally generates malformed tool JSON
    response = None
    for attempt in range(2):
        try:
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.1,
                max_tokens=1000,
            )
            break  # success
        except Exception as e:
            err_str = str(e)
            print(f"Groq round 1 error (attempt {attempt+1}): {err_str}")
            if attempt == 1 or "tool_use_failed" not in err_str:
                # Non-retryable or out of retries — fall back to plain answer
                try:
                    fallback = groq_client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=messages,
                        temperature=0.1,
                        max_tokens=600,
                    )
                    answer = fallback.choices[0].message.content.strip()
                except Exception:
                    answer = "Sorry, I'm having trouble right now. Please try again."
                save_context(question, answer)
                return {"answer": answer, "chart_title": "", "chart_data": []}

    if response is None:
        return {"answer": "Sorry, I'm having trouble right now. Please try again.",
                "chart_title": "", "chart_data": []}

    msg = response.choices[0].message

    # ── No tool call — Groq answered directly (knowledge/trivia) ─────────────
    if not msg.tool_calls:
        answer = msg.content or "Sorry, I couldn't answer that."
        save_context(question, answer)
        return {"answer": answer, "chart_title": "", "chart_data": []}

    # ── Execute all tool calls ────────────────────────────────────────────────
    tool_results = []
    for tc in msg.tool_calls:
        tool_name = tc.function.name
        try:
            tool_args = json.loads(tc.function.arguments)
        except Exception:
            tool_args = {}

        print(f"Tool call: {tool_name}({tool_args})")
        result = _execute_tool(tool_name, tool_args)
        print(f"Tool result keys: {list(result.keys()) if isinstance(result, dict) else 'non-dict'}")

        # Extract chart from first successful tool result
        if not chart_title and "error" not in result:
            chart_title, chart_data = _extract_chart(tool_name, result)

        tool_results.append({
            "tool_call_id": tc.id,
            "tool_name":    tool_name,
            "result":       result,
        })

    # ── Build answer directly from tool results — NO Groq for data answers ──────
    # Groq cannot reliably copy numbers verbatim. We format directly in Python
    # and only call Groq Round 2 for pure knowledge questions (no tool results).
    formatted_results = []
    all_errors = True
    for tr in tool_results:
        result  = tr["result"]
        readable = _format_tool_result_readable(result)
        formatted_results.append(readable)
        if "error" not in result:
            all_errors = False

    if all_errors:
        # All tools returned errors — ask Groq to answer from knowledge
        messages.append({
            "role":       "assistant",
            "content":    msg.content or "",
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ]
        })
        for tr in tool_results:
            messages.append({
                "role": "tool", "tool_call_id": tr["tool_call_id"],
                "content": json.dumps(tr["result"]),
            })
        messages.append({"role": "user", "content": "The database returned no data. Answer from general cricket knowledge if possible, and note it's from general knowledge not live data."})
        try:
            r2 = groq_client.chat.completions.create(
                model=_MODEL_TOOL, messages=messages, temperature=0.1, max_tokens=400,
            )
            answer = r2.choices[0].message.content.strip()
        except Exception as e:
            print(f"Groq round 2 error: {e}")
            answer = "I don't have data for that."
    else:
        # Data found — format answer directly in Python, zero hallucination
        answer = _build_answer(question, tool_results)

    save_context(question, answer)
    return {"answer": answer, "chart_title": chart_title, "chart_data": chart_data}


# ── Other routes (unchanged) ───────────────────────────────────────────────────

@app.get("/")
def home():
    return {"message": "SportsFan360 AI running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/feed")
def feed():
    return get_feed()

@app.get("/teams")
def teams():
    return get_teams()

@app.get("/players")
def players(team: str = None):
    return get_players(team)

@app.get("/matches")
def matches():
    return get_matches()

@app.get("/standings")
def standings():
    return {"standings": dl.get_standings()}

@app.get("/live-matches")
def live_matches_route():
    return get_live_matches()

@app.get("/player-list")
def player_list():
    return {"players": dl.all_player_names()}

@app.get("/player-battle")
def player_battle(p1: str, p2: str):
    result = dl.compare_players(p1, p2, prefix="IPL")
    if "error" in result:
        return {"error": result["error"]}
    s1 = result["player1"]
    s2 = result["player2"]
    impact1 = (s1["runs"] or 0) + (s1["wickets"] or 0) * 20 + (s1["sixes"] or 0) * 2
    impact2 = (s2["runs"] or 0) + (s2["wickets"] or 0) * 20 + (s2["sixes"] or 0) * 2
    return {
        "player1": s1["name"], "player2": s2["name"],
        "stats1": s1, "stats2": s2,
        "impact1": impact1, "impact2": impact2,
        "winner": s1["name"] if impact1 >= impact2 else s2["name"],
    }

@app.get("/player-shotmap")
def player_shotmap(player: str):
    return {"data": {
        "off":      random.randint(10, 100),
        "leg":      random.randint(10, 100),
        "straight": random.randint(10, 100),
    }}

@app.get("/match-commentary")
def match_commentary(team1: str, team2: str, status: str):
    prompt = f"Match: {team1} vs {team2}\nStatus: {status}\nGive short live commentary in 2-3 lines."
    try:
        res = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a professional cricket commentator."},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.7, max_tokens=150,
        )
        return {"commentary": res.choices[0].message.content.strip()}
    except Exception as e:
        return {"commentary": f"{team1} vs {team2} is in progress."}

@app.get("/daily-challenge")
def daily_challenge(matchId: str = "default"):
    try:
        parts = matchId.split("-") if "-" in matchId else ["MI", "CSK"]
        team1, team2 = parts[0], parts[1]
    except Exception:
        team1, team2 = "MI", "CSK"

    batsmen = ["Virat Kohli", "Rohit Sharma", "KL Rahul", "Shubman Gill"]
    bowlers  = ["Jasprit Bumrah", "Rashid Khan", "Yuzvendra Chahal", "Mohammed Shami"]
    teams    = [team1, team2]
    random.shuffle(teams)

    return {
        "matchId": matchId,
        "questions": [
            {"id": "winner",      "question": "🏆 Who will win?",    "options": teams},
            {"id": "top_batsman", "question": "🔥 Top Batsman?",     "options": batsmen},
            {"id": "top_bowler",  "question": "🎯 Top Bowler?",      "options": bowlers},
            {"id": "total_runs",  "question": "💥 Total Runs?",      "options": ["<150","150-170","170-190","190+"]},
            {"id": "toss",        "question": "⚡ Toss Winner?",     "options": teams},
            {"id": "powerplay",   "question": "🎯 Powerplay Score?", "options": ["<40","40-60","60-80","80+"]},
        ],
    }
