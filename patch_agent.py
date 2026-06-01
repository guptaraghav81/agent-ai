"""
patch_agent.py — Run from C:\SD\SF360\sportsfan360-ai-agent-main
Patches _build_answer to check win_rate type before generic rows handler.
"""
import re

src = open('agent.py', encoding='utf-8').read()

# Find _build_answer function
ba_start = src.find('def _build_answer(')
if ba_start == -1:
    print("ERROR: _build_answer not found")
    exit(1)

# Find the first 'if "rows" in r' inside _build_answer
rows_pattern = re.compile(r'( +)if "rows" in r and r\["rows"\]:', re.MULTILINE)
m = rows_pattern.search(src, ba_start)
if not m:
    print("ERROR: 'if \"rows\" in r and r[\"rows\"]:' not found in _build_answer")
    # Show what's around _build_answer
    print(src[ba_start:ba_start+500])
    exit(1)

indent = m.group(1)  # capture the indentation level
pos = m.start()

print(f"Found 'rows' check at position {pos}, line {src[:pos].count(chr(10))+1}")
print(f"Indentation: {repr(indent)}")

# Build the win_rate block with matching indentation
i = indent  # shorthand
win_rate_block = f"""{i}if r.get("type") == "win_rate":
{i}    rows = r.get("rows", [])
{i}    comp = r.get("competition", "IPL")
{i}    team = r.get("team")
{i}    if not rows:
{i}        parts.append("I don't have win rate data for that team.")
{i}        continue
{i}    if team and len(rows) == 1:
{i}        row = rows[0]
{i}        parts.append(
{i}            f"**{{row['team']}}** win rate in {{comp}}:\\n"
{i}            f"- **Matches:** {{row['matches']}}\\n"
{i}            f"- **Wins:** {{row['wins']}}\\n"
{i}            f"- **Win rate:** **{{row['win_rate']}}%**"
{i}        )
{i}    else:
{i}        out = [f"**{{comp}} Win Rate:**"]
{i}        for row in rows:
{i}            out.append(
{i}                f"{{row['rank']}}. **{{row['team']}}** — "
{i}                f"{{row['win_rate']}}% ({{row['wins']}}W/{{row['matches']}} matches)"
{i}            )
{i}        parts.append("\\n".join(out))

{i}elif "rows" in r and r["rows"]:"""

# Replace the old 'if "rows"' with win_rate check + elif
old_line = f'{indent}if "rows" in r and r["rows"]:'
new_src = src[:pos] + win_rate_block + src[pos + len(old_line):]

# Verify syntax
import ast
try:
    ast.parse(new_src)
    print("Syntax: OK")
except SyntaxError as e:
    print(f"SYNTAX ERROR: {e}")
    exit(1)

# Verify the fix is in place
if 'r.get("type") == "win_rate"' in new_src:
    idx_wr = new_src.find('r.get("type") == "win_rate"', ba_start)
    idx_rows = new_src.find('elif "rows" in r and r["rows"]:', ba_start)
    print(f"win_rate check at line {new_src[:idx_wr].count(chr(10))+1}")
    print(f"rows elif at line {new_src[:idx_rows].count(chr(10))+1}")
    print(f"Order correct: {idx_wr < idx_rows}")

open('agent.py', 'w', encoding='utf-8').write(new_src)
print("\nPATCHED SUCCESSFULLY — now run:")
print("  git add agent.py")
print("  git commit -m 'Fix: win_rate check before rows in _build_answer'")
print("  git push")
