import os
import re
import json
import random

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
from google import genai
from google.genai import types
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

import data_loader as dl
from memory_store import save_context, get_context
from feed_engine import get_feed
from teams_engine import get_teams
from players_engine import get_players
from matches_engine import get_matches
from live_matches import get_live_matches

# ── App setup ──────────────────────────────────────────────────────────────────
app = FastAPI()

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(status_code=429, content={"error": "Too many requests. Slow down!"})

origins = [
    "https://sportsfan360.com",
    "https://www.sportsfan360.com",
    "https://ask-ai-two-murex.vercel.app",
    "https://sportsfan-frontend.vercel.app",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load data at startup
try:
    dl.load()
except Exception as e:
    print("Data layer load failed at startup:", e)


# Initialize Gemini client via API key (permanent, reliable)
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

_MODEL_NAME = "gemini-2.5-flash"


# ── Tool executor ──────────────────────────────────────────────────────────────

def _execute_tool(name: str, args: dict) -> dict:
    if args is None:
        args = {}
    if "n" in args:
        try:
            args["n"] = int(args["n"])
        except (ValueError, TypeError):
            args["n"] = 5
    try:
        if name == "get_career_leaderboard":
            n      = args.get("n", 5)
            prefix = args.get("prefix", "IPL")
            stat   = args["stat"]
            if prefix.upper() in ["IPL", "OVERALL", "T20I"]:
                return {"error": f"The database only contains a subset of active roster players. For all-time/overall {prefix} {stat} career leaderboards, please use Google Search grounding fallback to get the correct factual answer."}
            if prefix == "2025":
                rows = dl.get_season_leaderboard(stat=stat, season=2025, competition="IPL", phase="ALL", n=n)
                result = {"rows": rows, "stat": stat, "season": 2025, "competition": "IPL", "phase": "ALL"}
                if n == 1 and rows:
                    result["winner"] = rows[0]
                return result
            rows = dl.get_career_leaderboard(stat=stat, prefix=prefix, n=n)
            result = {"rows": rows, "stat": stat, "prefix": prefix}
            if n == 1 and rows:
                result["winner"] = rows[0]
            return result

        elif name == "get_season_leaderboard":
            n = args.get("n", 5)
            season = int(args["season"])
            if season < 2025:
                return {"error": f"The database only contains data for the active 2026 players roster. For historical seasons like {season}, please use Google Search grounding fallback to get the correct factual answer."}
            rows = dl.get_season_leaderboard(
                stat=args["stat"], season=season,
                competition=args.get("competition", "IPL"),
                phase=args.get("phase", "ALL"), n=n
            )
            result = {"rows": rows, "stat": args["stat"], "season": season,
                    "competition": args.get("competition","IPL"), "phase": args.get("phase","ALL")}
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
            result = dl.get_matchup(args["batter"], args["bowler"], args.get("competition","IPL"), args.get("phase","ALL"))
            if not result:
                result = dl.get_matchup(args["bowler"], args["batter"], args.get("competition","IPL"), args.get("phase","ALL"))
            if not result:
                result = dl.get_matchup(args["batter"], args["bowler"], "Career", args.get("phase","ALL"))
            if not result:
                return {"error": f"No matchup data found for {args['batter']} vs {args['bowler']}"}
            return {"matchup": result}

        elif name == "get_batter_weaknesses":
            result = dl.get_batter_vs_all_bowlers(args["batter"], args.get("competition","IPL"), args.get("phase","ALL"))
            return result

        elif name == "get_venue_player_stats":
            result = dl.get_venue_stats(args["player"], args["venue"])
            if not result:
                return {"error": f"No venue data found for {args['player']} at {args['venue']}"}
            return result

        elif name == "get_venue_leaderboard":
            rows = dl.get_venue_leaderboard(args["venue"], args.get("stat","runs"), args.get("n",5))
            return {"rows": rows, "venue": args["venue"], "stat": args.get("stat","runs")}

        elif name == "get_team_vs_team":
            result = dl.get_team_vs_team(args["team1"], args["team2"], args.get("competition","IPL"), args.get("season"))
            if not result:
                return {"error": f"No head-to-head data found for {args['team1']} vs {args['team2']}"}
            return result

        elif name == "get_compare_players":
            return dl.compare_players(args["player1"], args["player2"], args.get("prefix","IPL"))

        elif name == "get_team_win_rate":
            rows = dl.get_team_win_rate(team=args.get("team"), competition=args.get("competition", "IPL"))
            return {"rows": rows, "type": "win_rate", "team": args.get("team"), "competition": args.get("competition","IPL")}

        elif name == "get_standings":
            return {"standings": dl.get_standings()}

        elif name == "get_titles":
            return {"titles": dl.ipl_titles_table()}

        else:
            return {"error": f"Unknown tool: {name}"}

    except Exception as e:
        return {"error": str(e)}


def _format_tool_result_readable(result: dict) -> str:
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
            name = r.get('player', r.get('team', '?'))
            val  = r.get('value', r.get('win_rate', r.get('wins', '?')))
            lines.append(f"  {r['rank']}. {name} — {val} {stat}")
        if "winner" in result:
            w = result["winner"]
            lines.append(f"WINNER: {w['player']} with {w['value']} {stat}")
        return "\n".join(lines)
    if "matchup" in result:
        m = result["matchup"]
        return (f"MATCHUP — {m['batter']} vs {m['bowler']} in {m['competition']} {m['phase']}:\n"
                f"  Balls: {m['balls']}, Runs: {m['runs']}, Dismissed: {m['dismissed']}, "
                f"SR: {m.get('sr','N/A')}, Dot%: {m.get('dot_pct','N/A')}, Dismiss rate: {m.get('dismiss_rate','N/A')}")
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
        return (f"HEAD TO HEAD — {result['team1']} vs {result['team2']} ({result.get('competition','')}):\n"
                f"  Total matches: {result['matches']}\n"
                f"  {result['team1']} wins: {result['team1_wins']}\n"
                f"  {result['team2']} wins: {result['team2_wins']}\n"
                f"  No result: {result.get('no_result',0)}")
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
    return json.dumps(result)


def _fmt(val):
    if val is None:
        return "N/A"
    try:
        f = float(val)
        return str(int(f)) if f == int(f) else f"{f:.2f}"
    except:
        return str(val)


def _build_answer(question: str, tool_results: list) -> str:
    parts = []
    for tr in tool_results:
        r    = tr["result"]
        name = tr["tool_name"]
        if "error" in r:
            parts.append(f"I don't have data for that: {r['error']}")
            continue
        if r.get("type") == "win_rate":
            rows = r.get("rows", [])
            comp = r.get("competition", "IPL")
            team = r.get("team")
            if not rows:
                parts.append("I don't have win rate data for that team.")
                continue
            if team and len(rows) == 1:
                row = rows[0]
                parts.append(f"**{row['team']}** win rate in {comp}:\n- **Matches played:** {row['matches']}\n- **Wins:** {row['wins']}\n- **Win rate:** **{row['win_rate']}%**")
            else:
                lines_out = [f"**{comp} Win Rate — All Teams:**\n"]
                for row in rows:
                    lines_out.append(f"{row['rank']}. **{row['team']}** — {row['win_rate']}% ({row['wins']}W / {row['matches']} matches)")
                parts.append("\n".join(lines_out))
        elif "rows" in r and r["rows"]:
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
            stat_label = {"runs":"runs","wickets":"wickets","sixes":"sixes","fours":"fours","avg":"batting avg","sr":"strike rate","economy":"economy","bowl_avg":"bowling avg","bowl_sr":"bowling SR","dot_pct":"dot ball %","balls_faced":"balls faced"}.get(stat, stat)
            if "winner" in r:
                w = r["winner"]
                parts.append(f"**{w['player']}** won the {stat_label} title in **{ctx}** with **{_fmt(w['value'])} {stat_label}**.")
            else:
                lines = [f"Top {len(rows)} by {stat_label} — **{ctx}**:"]
                for row in rows:
                    lines.append(f"{row['rank']}. **{row['player']}** — {_fmt(row['value'])} {stat_label}")
                parts.append("\n".join(lines))
        elif "matchup" in r:
            m = r["matchup"]
            phase_str = f" ({m['phase']} overs)" if m.get("phase") and m["phase"] != "ALL" else ""
            dismissed = m.get("dismissed") or 0
            balls     = m.get("balls") or 0
            # BUG FIX: dismiss_rate is % not balls-per-wicket
            balls_per_wicket = (_fmt(round(balls / dismissed, 1)) if dismissed else "N/A")
            parts.append(f"**{m['batter']} vs {m['bowler']}** in {m['competition']}{phase_str}:\n- **Balls faced:** {m['balls']}\n- **Runs scored:** {m['runs']}\n- **Dismissals:** {m['dismissed']}\n- **Strike rate:** {_fmt(m.get('sr'))}\n- **Dot ball %:** {_fmt(m.get('dot_pct'))}%\n- **Dismissal rate:** {_fmt(m.get('dismiss_rate'))}% (once every {balls_per_wicket} balls)")
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
            if s.get("runs") is not None:
                lines.append(f"🏏 **Batting:** {s.get('innings') or 0} innings, **{_fmt(s.get('runs'))} runs**, avg {_fmt(s.get('avg'))}, SR {_fmt(s.get('sr'))}, {s.get('sixes') or 0} sixes")
            if s.get("wickets") is not None:
                lines.append(f"🎳 **Bowling:** {s.get('bowl_innings') or 0} innings, **{_fmt(s.get('wickets'))} wickets**, econ {_fmt(s.get('economy'))}, avg {_fmt(s.get('bowl_avg'))}")
            parts.append("\n".join(lines))
        elif "standings" in r:
            lines = ["**IPL 2026 Points Table:**\n"]
            for row in r["standings"]:
                lines.append(f"{row['position']}. **{row['team']}** — {row['points']} pts  (W{row['won']} L{row['lost']}, NRR {row['nrr']:+.3f})")
            parts.append("\n".join(lines))
        elif "titles" in r:
            winners = [t for t in r["titles"] if t["titles"] > 0]
            lines   = ["**IPL Title Count:**\n"]
            for t in winners:
                lines.append(f"- **{t['team']}**: {t['titles']} 🏆")
            parts.append("\n".join(lines))
        elif "team1" in r:
            t1, t2   = r["team1"], r["team2"]
            comp     = r.get("competition", "IPL")
            season   = r.get("season")
            ctx      = f"{comp} {season}" if season else f"all-time {comp}"
            matches  = r["matches"]
            t1_wins  = r["team1_wins"]
            t2_wins  = r["team2_wins"]
            nr       = r.get("no_result", 0)
            decided  = matches - nr
            t1_pct   = round(t1_wins / decided * 100, 1) if decided else 0
            t2_pct   = round(t2_wins / decided * 100, 1) if decided else 0
            parts.append(f"**{t1} vs {t2}** — {ctx}:\n- Total matches: **{matches}**\n- **{t1}** wins: **{t1_wins}** ({t1_pct}%)\n- **{t2}** wins: **{t2_wins}** ({t2_pct}%)\n- No result: {nr}")
        elif "batting" in r or "bowling" in r:
            name_str  = r.get("name", "")
            venue_str = r.get("venue_query", "")
            lines = [f"**{name_str}** at **{venue_str}**:"]
            if "batting" in r:
                b = r["batting"]
                lines.append(f"🏏 **Batting:** {b.get('innings',0)} innings, **{_fmt(b.get('runs'))} runs**, avg {_fmt(b.get('avg'))}, SR {_fmt(b.get('sr'))}")
            if "bowling" in r:
                b = r["bowling"]
                lines.append(f"🎳 **Bowling:** **{_fmt(b.get('wickets'))} wickets**, econ {_fmt(b.get('economy'))}, avg {_fmt(b.get('bowl_avg'))}")
            parts.append("\n".join(lines))
        elif "venue" in r and "rows" in r:
            rows  = r["rows"]
            venue = r.get("venue", "")
            stat  = r.get("stat", "value")
            lines = [f"Top {len(rows)} by {stat} at **{venue}**:"]
            for row in rows:
                lines.append(f"{row['rank']}. **{row['player']}** — {_fmt(row['value'])} {stat}")
            parts.append("\n".join(lines))
        elif "player1" in r and "player2" in r:
            s1 = r["player1"]
            s2 = r["player2"]
            lines = [f"**{s1['name']} vs {s2['name']}** — IPL career:\n"]
            lines.append(f"| Stat | {s1['name']} | {s2['name']} |")
            lines.append("|------|------|------|")
            for stat, label in [("runs","Runs"),("avg","Bat Avg"),("sr","Bat SR"),("sixes","Sixes"),("wickets","Wickets"),("economy","Economy")]:
                v1 = _fmt(s1.get(stat))
                v2 = _fmt(s2.get(stat))
                lines.append(f"| {label} | {v1} | {v2} |")
            parts.append("\n".join(lines))
    return "\n\n".join(parts) if parts else "I don't have data for that."


def _extract_chart(tool_name: str, tool_result: dict) -> tuple:
    # BUG FIX A: Suppress chart immediately for matchups
    if tool_name == "get_matchup" or "matchup" in tool_result:
        return "", []
    if "rows" in tool_result and tool_result["rows"]:
        rows  = tool_result["rows"]
        stat  = tool_result.get("stat", "value")
        if tool_result.get("type") == "win_rate":
            return "", []
        title = f"Top {len(rows)} — {stat}"
        data  = [{"player": r.get("player", r.get("team", "?")), "value": r["value"]} for r in rows]
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


SYSTEM_PROMPT = """You are AskSportsFan360, a premium cricket analytics assistant.

INSTRUCTION RULES:
1. For specific player matching queries, use the database tools. If the tools return detailed stats, report them accurately.
2. For global, historical, or "all-time" statistics and queries (like "all-time most runs in IPL", "all-time purple cap winners", etc.), if you see the database tools return values that exclude famous historical figures (e.g. Shikhar Dhawan, David Warner, Chris Gayle), you MUST use your Google Search tool grounding or general knowledge intelligence to deliver a 100% correct, verified all-time factual response.
3. Combine database tool outputs with your live search/knowledge grounding to ensure the user gets complete and correct sports history.
4. Bold key names and numbers using **. Make answers professional and concise.
"""



def _run_agent(question: str, conversation_history: list = None) -> dict:
    if not dl._loaded:
        try:
            dl.load()
        except Exception as e:
            return {"answer": "Data is loading, please try again.", "chart_title": "", "chart_data": []}

    chart_title = ""
    chart_data  = []

    history_msgs = []
    if conversation_history:
        for turn in conversation_history:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if role in ("user", "model", "assistant") and content:
                gemini_role = "model" if role in ("assistant", "model") else "user"
                history_msgs.append(
                    types.Content(
                        role=gemini_role,
                        parts=[types.Part.from_text(text=content)]
                    )
                )

    # Prepare user query content
    user_content = types.Content(
        role="user",
        parts=[types.Part.from_text(text=question)]
    )
    all_contents = history_msgs + [user_content]

    # Prepare Gemini declarative tools mapping
    declarations = [
        types.FunctionDeclaration(
            name="get_career_leaderboard",
            description="Get the top N players by a career stat. Use for questions like 'most IPL runs', 'best T20I economy', 'most Overall sixes', prefix options: IPL | T20I | IPL26 | Overall | 2025",
            parameters={
                "type": "OBJECT",
                "properties": {
                    "stat": {"type": "STRING", "description": "Stat to rank by e.g. runs, wickets, avg, sr, economy"},
                    "prefix": {"type": "STRING", "description": "IPL | T20I | IPL26 | Overall"},
                    "n": {"type": "INTEGER", "description": "Number of results to return", "default": 5}
                },
                "required": ["stat", "prefix"]
            }
        ),
        types.FunctionDeclaration(
            name="get_season_leaderboard",
            description="Get the top N players by a stat in a specific season and competition. ALWAYS use this for any specific year like 2025, 2024, 2019.",
            parameters={
                "type": "OBJECT",
                "properties": {
                    "stat": {"type": "STRING", "description": "runs, wickets, avg, sr, economy"},
                    "season": {"type": "INTEGER", "description": "Year e.g. 2016, 2023"},
                    "competition": {"type": "STRING", "description": "e.g. IPL, T20I", "default": "IPL"},
                    "phase": {"type": "STRING", "description": "ALL | PP | MID | DEATH", "default": "ALL"},
                    "n": {"type": "INTEGER", "default": 5}
                },
                "required": ["stat", "season"]
            }
        ),
        types.FunctionDeclaration(
            name="get_player_career_stats",
            description="Get career stats for a specific named player.",
            parameters={
                "type": "OBJECT",
                "properties": {
                    "player": {"type": "STRING", "description": "Player name"},
                    "prefix": {"type": "STRING", "description": "IPL | T20I | IPL26 | Overall", "default": "IPL"}
                },
                "required": ["player"]
            }
        ),
        types.FunctionDeclaration(
            name="get_player_season_stats",
            description="Get stats for a specific player in a specific season.",
            parameters={
                "type": "OBJECT",
                "properties": {
                    "player": {"type": "STRING"},
                    "season": {"type": "INTEGER"},
                    "competition": {"type": "STRING", "default": "IPL"},
                    "phase": {"type": "STRING", "description": "ALL | PP | MID | DEATH", "default": "ALL"}
                },
                "required": ["player", "season"]
            }
        ),
        types.FunctionDeclaration(
            name="get_matchup",
            description="Get head-to-head stats between a batter and a bowler.",
            parameters={
                "type": "OBJECT",
                "properties": {
                    "batter": {"type": "STRING"},
                    "bowler": {"type": "STRING"},
                    "competition": {"type": "STRING", "default": "IPL"},
                    "phase": {"type": "STRING", "description": "ALL | PP | MID | DEATH", "default": "ALL"}
                },
                "required": ["batter", "bowler"]
            }
        ),
        types.FunctionDeclaration(
            name="get_batter_weaknesses",
            description="Find which bowlers trouble a batter most, or which bowlers a batter dominates.",
            parameters={
                "type": "OBJECT",
                "properties": {
                    "batter": {"type": "STRING"},
                    "competition": {"type": "STRING", "default": "IPL"},
                    "phase": {"type": "STRING", "description": "ALL | PP | MID | DEATH", "default": "ALL"}
                },
                "required": ["batter"]
            }
        ),
        types.FunctionDeclaration(
            name="get_venue_player_stats",
            description="Get a player's stats at a specific venue.",
            parameters={
                "type": "OBJECT",
                "properties": {
                    "player": {"type": "STRING"},
                    "venue": {"type": "STRING", "description": "Partial venue name e.g. Wankhede, Chepauk"}
                },
                "required": ["player", "venue"]
            }
        ),
        types.FunctionDeclaration(
            name="get_venue_leaderboard",
            description="Get the best batters or bowlers at a specific venue.",
            parameters={
                "type": "OBJECT",
                "properties": {
                    "venue": {"type": "STRING"},
                    "stat": {"type": "STRING", "description": "runs, avg, sr, wickets, economy"},
                    "n": {"type": "INTEGER", "default": 5}
                },
                "required": ["venue"]
            }
        ),
        types.FunctionDeclaration(
            name="get_team_vs_team",
            description="Get head-to-head win/loss record between exactly two named IPL teams.",
            parameters={
                "type": "OBJECT",
                "properties": {
                    "team1": {"type": "STRING", "description": "Team name or abbreviation e.g. MI, CSK"},
                    "team2": {"type": "STRING"},
                    "competition": {"type": "STRING", "default": "IPL"},
                    "season": {"type": "INTEGER"}
                },
                "required": ["team1", "team2"]
            }
        ),
        types.FunctionDeclaration(
            name="get_compare_players",
            description="Compare two players head-to-head across career stats.",
            parameters={
                "type": "OBJECT",
                "properties": {
                    "player1": {"type": "STRING"},
                    "player2": {"type": "STRING"},
                    "prefix": {"type": "STRING", "default": "IPL"}
                },
                "required": ["player1", "player2"]
            }
        ),
        types.FunctionDeclaration(
            name="get_team_win_rate",
            description="Get overall win rate for a SINGLE IPL team. ONLY use when exactly ONE team is mentioned with no opponent.",
            parameters={
                "type": "OBJECT",
                "properties": {
                    "team": {"type": "STRING", "description": "Team name or abbreviation"},
                    "competition": {"type": "STRING", "default": "IPL"}
                }
            }
        ),
        types.FunctionDeclaration(
            name="get_standings",
            description="Get the IPL 2026 points table / standings.",
            parameters={"type": "OBJECT", "properties": {}}
        ),
        types.FunctionDeclaration(
            name="get_titles",
            description="Get IPL title/championship count by team — all-time history.",
            parameters={"type": "OBJECT", "properties": {}}
        ),
    ]

    # Pre-process question to check for routing errors (e.g. MI vs KKR win rate should call get_team_vs_team)
    # Bug Fix: "MI vs KKR win rate" calling get_team_win_rate twice instead of get_team_vs_team
    routing_hack = False
    if "vs" in question.lower() and ("win rate" in question.lower() or "win percentage" in question.lower()):
        # Extract potential teams
        teams_found = []
        for t in ["mi", "csk", "rcb", "kkr", "srh", "rr", "pbks", "dc", "gt", "lsg", "mumbai", "chennai", "kolkata", "hyderabad", "bangalore"]:
            if re.search(r'\b' + t + r'\b', question.lower()):
                teams_found.append(t)
        if len(teams_found) >= 2:
            routing_hack = True
            team1, team2 = teams_found[0], teams_found[1]

    # Pre-check: If query asks for overall/all-time leaderboards of runs, wickets, or sixes, route directly to Gemini search grounding fallback to prevent database filtering issues
    use_search_fallback = False
    q_lower = question.lower()
    
    # Check for career / all-time leaderboard keywords
    has_leaderboard_keyword = any(x in q_lower for x in ["most", "highest", "leading", "top", "greatest", "maximum", "best", "orange cap", "purple cap", "records", "record"])
    has_stat_keyword = any(x in q_lower for x in ["runs", "run", "wickets", "wicket", "sixes", "six", "fours", "four", "boundaries", "boundary", "strike", "average", "avg", "economy", "scores", "score", "scorer", "taker", "cap", "century", "centuries", "fifties", "fifty", "match", "matches"])
    
    # If the query is a general leaderboard/records query, use search fallback
    if has_leaderboard_keyword and has_stat_keyword:
        # We only want to use the DB for the current/active seasons (2025/2026/IPL26)
        has_historical_year = any(yr in q_lower for yr in ["2008","2009","2010","2011","2012","2013","2014","2015","2016","2017","2018","2019","2020","2021","2022","2023","2024"])
        # If it has a historical year or NO year at all, use search fallback!
        if has_historical_year or not any(yr in q_lower for yr in ["2025","2026","ipl26","ipl 26","ipl 25","this season"]):
            use_search_fallback = True

    tool_results = []

    if routing_hack:
        # Route manually to bypass LLM tool calling choice logic errors
        print(f"Routing hack triggered: get_team_vs_team({team1}, {team2})")
        res = _execute_tool("get_team_vs_team", {"team1": team1, "team2": team2})
        tool_results.append({"tool_name": "get_team_vs_team", "result": res})
        answer = _build_answer(question, tool_results)
        save_context(question, answer)
        return {"answer": answer, "chart_title": "", "chart_data": []}

    if use_search_fallback:
        print("Historical query detected. Routing directly to Gemini Search Grounding fallback...")
        try:
            fallback_res = client.models.generate_content(
                model=_MODEL_NAME,
                contents=all_contents,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    tools=[types.Tool(google_search=types.GoogleSearch())],
                    temperature=0.2
                )
            )
            answer = fallback_res.text.strip()
            save_context(question, answer)
            return {"answer": answer, "chart_title": "", "chart_data": []}
        except Exception as e:
            print(f"Search fallback failed: {e}")

    try:
        # Round 1: Call Gemini with strict DB tools enabled
        response = client.models.generate_content(
            model=_MODEL_NAME,
            contents=all_contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                tools=[types.Tool(function_declarations=declarations)],
                temperature=0.1,
            )
        )

    except Exception as e:
        print(f"Gemini round 1 error: {e}")
        return {"answer": "Sorry, I'm having trouble right now. Please try again.", "chart_title": "", "chart_data": []}

    # Process tool calls
    calls = response.function_calls
    if not calls:
        # If model did not choose any database tool, it means it's a general question or knowledge query.
        # Immediately fall back to Gemini model grounded with Google Search!
        print("No DB tool chosen. Executing Search-grounded fallback...")
        try:
            fallback_res = client.models.generate_content(
                model=_MODEL_NAME,
                contents=all_contents,
                config=types.GenerateContentConfig(
                    system_instruction="You are AskSportsFan360, a sports analytics assistant. Answer using search results.",
                    tools=[types.Tool(google_search=types.GoogleSearch())],
                    temperature=0.2
                )
            )
            answer = fallback_res.text.strip()
            save_context(question, answer)
            return {"answer": answer, "chart_title": "", "chart_data": []}
        except Exception as e:
            print(f"Search fallback failed: {e}")
            return {"answer": "I don't have data for that right now.", "chart_title": "", "chart_data": []}

    # Execute tool calls
    for call in calls:
        tool_name = call.name
        tool_args = call.args
        print(f"Tool call: {tool_name}({tool_args})")
        res = _execute_tool(tool_name, tool_args)
        print(f"Tool result keys: {list(res.keys()) if isinstance(res, dict) else 'non-dict'}")
        
        # Suppress chart properly for matchup matching Bug Fix A
        if not chart_title and "error" not in res:
            chart_title, chart_data = _extract_chart(tool_name, res)
        tool_results.append({"tool_name": tool_name, "result": res})

    all_errors = all("error" in tr["result"] for tr in tool_results)

    if all_errors:
        # If DB returned errors, fall back to Google Search grounding to deliver actual real answers
        print("All database calls returned errors. Falling back to Google Search...")
        try:
            fallback_res = client.models.generate_content(
                model=_MODEL_NAME,
                contents=all_contents,
                config=types.GenerateContentConfig(
                    system_instruction="You are AskSportsFan360, a sports analytics assistant. The database has no match. Answer using Google search.",
                    tools=[types.Tool(google_search=types.GoogleSearch())],
                    temperature=0.2
                )
            )
            answer = fallback_res.text.strip()
        except Exception as e:
            print(f"Gemini fallback error: {e}")
            answer = "I don't have data for that."
    else:
        answer = _build_answer(question, tool_results)

    save_context(question, answer)
    return {"answer": answer, "chart_title": chart_title, "chart_data": chart_data}


# ── GET /ask ──────────────────────────────────────────────────────────────────

@app.get("/ask")
def ask(question: str):
    context_turns = get_context()
    history = []
    for t in context_turns:
        history.append({"role": "user", "content": t["question"]})
        history.append({"role": "assistant", "content": t["answer"]})
    return _run_agent(question, conversation_history=history)


# ── POST /chat ────────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    query: str
    conversation_history: Optional[List[ChatMessage]] = []
    user_id: Optional[str] = None
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    sources: List[str] = []
    metadata: dict = {}

@app.post("/chat", response_model=ChatResponse)
@limiter.limit("30/minute")
def chat(request: Request, req: ChatRequest):

    history = [{"role": m.role, "content": m.content} for m in (req.conversation_history or [])]
    result  = _run_agent(req.query, conversation_history=history)
    return ChatResponse(
        answer=result["answer"],
        sources=[],
        metadata={"chart_title": result.get("chart_title", ""), "chart_data": result.get("chart_data", [])}
    )


# ── Other routes ──────────────────────────────────────────────────────────────

@app.get("/debug-gemini")
def debug_gemini():
    try:
        res = client.models.generate_content(
            model=_MODEL_NAME,
            contents="test",
            config=types.GenerateContentConfig(max_output_tokens=10)
        )
        return {"status": "ok", "response": res.text, "project": os.getenv("GCP_PROJECT_ID"), "location": os.getenv("GCP_LOCATION")}
    except Exception as e:
        import traceback
        return {
            "status": "error",
            "error_message": str(e),
            "traceback": traceback.format_exc(),
            "gcp_project": os.getenv("GCP_PROJECT_ID"),
            "gcp_location": os.getenv("GCP_LOCATION"),
            "creds_file_exists": os.path.exists(os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "/etc/secrets/google_creds.json")),
            "creds_path": os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        }

@app.get("/")
def home():
    return {"message": "SportsFan360 AI running", "version": "v1.0.4-historical-seasons-fix"}

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
    return {"data": {"off": random.randint(10, 100), "leg": random.randint(10, 100), "straight": random.randint(10, 100)}}

@app.get("/match-commentary")
def match_commentary(team1: str, team2: str, status: str):
    # Fallback to Gemini commentary generation
    prompt = f"Match: {team1} vs {team2}\nStatus: {status}\nGive short live commentary in 2-3 lines."
    try:
        res = client.models.generate_content(
            model=_MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction="You are a professional cricket commentator.",
                max_output_tokens=150
            )
        )
        return {"commentary": res.text.strip()}
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