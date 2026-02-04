"""
Extract Summary Generator

Compares new extraction data against previous extracts and generates
summary reports (JSON + HTML) showing what changed.
"""

import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.settings import get_fixtures, FIXTURE_DATES, TEAM_COLOURS


def compare_extract(old_data: Optional[Dict], new_data: Dict, round_num: int, year: int) -> Dict:
    """
    Compare new extract data against previous data for a single round.

    Returns a diff dict describing what changed.
    """
    output_file = f"six_nations_stats_{year}_round_{round_num}.json"
    new_players = new_data.get("players", [])

    # Check if new data actually has stats populated
    has_data = any(
        any(v != "" and v != 0 for v in p.get("stats", {}).values())
        for p in new_players
    )

    result = {
        "round": round_num,
        "file": output_file,
        "extraction_date": new_data.get("extraction_date", str(date.today())),
        "total_players": len(new_players),
        "has_data": has_data,
        "players_added": [],
        "players_removed": [],
        "stat_changes": [],
    }

    if old_data is None:
        # First extract - all players are new
        result["players_added"] = [p["name"] for p in new_players]
        return result

    old_players = old_data.get("players", [])
    old_by_name = {p["name"]: p for p in old_players}
    new_by_name = {p["name"]: p for p in new_players}

    # Players added / removed
    result["players_added"] = sorted(set(new_by_name) - set(old_by_name))
    result["players_removed"] = sorted(set(old_by_name) - set(new_by_name))

    # Stat changes for players present in both
    for name in sorted(set(old_by_name) & set(new_by_name)):
        old_stats = old_by_name[name].get("stats", {})
        new_stats = new_by_name[name].get("stats", {})
        all_keys = sorted(set(old_stats) | set(new_stats))
        for key in all_keys:
            old_val = old_stats.get(key, "")
            new_val = new_stats.get(key, "")
            if old_val != new_val:
                result["stat_changes"].append({
                    "player": name,
                    "field": key,
                    "old": old_val,
                    "new": new_val,
                })

    return result


def generate_summary(year: int, output_dir: str) -> Dict:
    """
    Read all per-round diff files and produce a combined summary JSON + HTML.
    """
    output_path = Path(output_dir)
    rounds = []

    for r in range(1, 6):
        diff_file = output_path / f"extract_diff_{year}_round_{r}.json"
        if diff_file.exists():
            with open(diff_file) as f:
                rounds.append(json.load(f))

    summary = {
        "generated_date": str(date.today()),
        "year": year,
        "rounds": rounds,
    }

    # Write summary JSON
    summary_json_path = output_path / f"extract_summary_{year}.json"
    with open(summary_json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary JSON written to {summary_json_path}")

    # Write summary HTML
    html = _build_summary_html(summary)
    html_path = output_path / f"extract_summary_{year}.html"
    with open(html_path, "w") as f:
        f.write(html)
    print(f"Summary HTML written to {html_path}")

    return summary


def _build_summary_html(summary: Dict) -> str:
    """Build a standalone HTML summary page."""
    year = summary["year"]
    gen_date = summary["generated_date"]
    rounds = summary["rounds"]

    # Determine latest extraction date across rounds
    extract_dates = [r["extraction_date"] for r in rounds if r.get("extraction_date")]
    latest_extract = max(extract_dates) if extract_dates else "N/A"

    # Build overview table rows
    overview_rows = ""
    for r in rounds:
        n_changes = len(r.get("stat_changes", []))
        n_added = len(r.get("players_added", []))
        n_removed = len(r.get("players_removed", []))
        data_badge = "Yes" if r.get("has_data") else "No"
        overview_rows += f"""
        <tr>
          <td>Round {r['round']}</td>
          <td><code>{r['file']}</code></td>
          <td>{r['total_players']}</td>
          <td>{data_badge}</td>
          <td>{n_added}</td>
          <td>{n_removed}</td>
          <td>{n_changes}</td>
        </tr>"""

    # Build fixtures section
    fixtures = get_fixtures(year)
    fixture_dates = FIXTURE_DATES.get(year, {})
    fixtures_html = ""
    if fixtures:
        fixtures_html += "<h2>Fixtures</h2>\n"
        for rnd in sorted(fixtures.keys()):
            date_str = fixture_dates.get(rnd, "")
            date_label = f" &mdash; {date_str}" if date_str else ""
            fixtures_html += f'    <h3>Round {rnd}{date_label}</h3>\n'
            fixtures_html += '    <div class="fixtures-round">\n'
            for home, away in fixtures[rnd]:
                home_colour = TEAM_COLOURS.get(home, '#333')
                away_colour = TEAM_COLOURS.get(away, '#333')
                fixtures_html += f"""      <div class="fixture-card">
        <span class="team" style="border-left: 4px solid {home_colour}; padding-left: 0.5rem;">{home}</span>
        <span class="vs">v</span>
        <span class="team" style="border-right: 4px solid {away_colour}; padding-right: 0.5rem; text-align: right;">{away}</span>
      </div>\n"""
            fixtures_html += '    </div>\n'

    # Build per-round detail sections
    detail_sections = ""
    for r in rounds:
        changes = r.get("stat_changes", [])
        added = r.get("players_added", [])
        removed = r.get("players_removed", [])

        if not changes and not added and not removed:
            detail_sections += f"""
        <details open>
          <summary>Round {r['round']} &mdash; no changes</summary>
          <p>No differences from previous extract.</p>
        </details>"""
            continue

        inner = ""
        if added:
            inner += "<h4>Players Added</h4><ul>"
            for name in added:
                inner += f"<li>{name}</li>"
            inner += "</ul>"

        if removed:
            inner += "<h4>Players Removed</h4><ul>"
            for name in removed:
                inner += f"<li>{name}</li>"
            inner += "</ul>"

        if changes:
            inner += """<h4>Stat Changes</h4>
            <table>
              <thead><tr><th>Player</th><th>Stat</th><th>Previous</th><th>Current</th><th>Diff</th></tr></thead>
              <tbody>"""
            for c in changes:
                old_display = c['old'] if c['old'] != "" else "&mdash;"
                new_display = c['new'] if c['new'] != "" else "&mdash;"
                # Calculate numeric diff where possible
                diff_display = ""
                try:
                    old_num = float(c['old']) if c['old'] != "" else None
                    new_num = float(c['new']) if c['new'] != "" else None
                    if old_num is not None and new_num is not None:
                        diff_val = new_num - old_num
                        sign = "+" if diff_val > 0 else ""
                        diff_display = f'<span class="{"pos" if diff_val > 0 else "neg" if diff_val < 0 else ""}">{sign}{diff_val:g}</span>'
                    elif old_num is None and new_num is not None:
                        diff_display = f'<span class="pos">+{new_num:g}</span>'
                    elif old_num is not None and new_num is None:
                        diff_display = f'<span class="neg">-{old_num:g}</span>'
                except (ValueError, TypeError):
                    pass
                inner += f"<tr><td>{c['player']}</td><td>{c['field']}</td><td>{old_display}</td><td>{new_display}</td><td>{diff_display}</td></tr>"
            inner += "</tbody></table>"

        detail_sections += f"""
        <details open>
          <summary>Round {r['round']} &mdash; {len(added)} added, {len(removed)} removed, {len(changes)} stat changes</summary>
          {inner}
        </details>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Extract Summary &mdash; {year}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; color: #222; }}
    h1 {{ border-bottom: 2px solid #333; padding-bottom: 0.5rem; }}
    .meta {{ color: #666; margin-bottom: 1.5rem; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
    th, td {{ border: 1px solid #ccc; padding: 0.4rem 0.8rem; text-align: left; }}
    th {{ background: #f5f5f5; }}
    tr:nth-child(even) {{ background: #fafafa; }}
    code {{ background: #eee; padding: 0.1rem 0.3rem; border-radius: 3px; font-size: 0.9em; }}
    details {{ margin: 0.8rem 0; }}
    summary {{ cursor: pointer; font-weight: 600; padding: 0.4rem 0; }}
    details table {{ font-size: 0.9em; }}
    a {{ color: #1a6; }}
    .fixtures-round {{ display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1rem; }}
    .fixture-card {{ display: flex; align-items: center; gap: 0.6rem; background: #f9f9f9; border: 1px solid #ddd; border-radius: 6px; padding: 0.5rem 1rem; min-width: 220px; }}
    .fixture-card .team {{ font-weight: 600; flex: 1; }}
    .fixture-card .vs {{ color: #999; font-size: 0.85em; }}
  </style>
</head>
<body>
  <h1>Extract Summary &mdash; {year}</h1>
  <p class="meta">
    Last extraction: <strong>{latest_extract}</strong> &nbsp;|&nbsp;
    Summary generated: <strong>{gen_date}</strong>
  </p>

  <h2>Overview</h2>
  <table>
    <thead>
      <tr>
        <th>Round</th><th>File</th><th>Players</th><th>Has Data</th>
        <th>Added</th><th>Removed</th><th>Stat Changes</th>
      </tr>
    </thead>
    <tbody>{overview_rows}
    </tbody>
  </table>

  {fixtures_html}

  <h2>Details</h2>
  {detail_sections}

  <p style="margin-top:2rem;color:#999;font-size:0.85em;">
    <a href="../index.html">&larr; Back to index</a>
  </p>
</body>
</html>"""
