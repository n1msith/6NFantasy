# CLAUDE.md - 6NFantasy Project Context

## Overview

6NFantasy is a Python-based analytics tool for Six Nations Fantasy Rugby. It extracts player statistics from the official Six Nations Fantasy Rugby API, combines them with fantasy value data from Google Sheets, and generates interactive visualizations to support team selection.

## Tech Stack

- **Language**: Python 3.10+
- **Data**: pandas, numpy
- **Visualization**: Plotly (interactive HTML charts)
- **HTTP**: requests
- **Auth**: PyJWT (token decoding)
- **Config**: python-dotenv

## Project Structure

```
6NFantasy/
├── src/
│   ├── api/
│   │   ├── client.py          # Six Nations API client, JWT auth
│   │   └── constants.py       # French→English field mappings, position corrections
│   ├── auth/
│   │   └── token_validator.py # JWT token validation (30-day expiry)
│   └── processors/
│       ├── api_stats_extractor.py    # Main extraction orchestrator
│       ├── fantasy_value_extractor.py # CSV parsing, name matching
│       └── extract_summary.py        # Diff generation, HTML reports
├── analysis/
│   └── analyse_stats.py       # All visualization functions (1000+ lines)
├── config/
│   └── settings.py            # API URLs, fixtures, team colors
├── data/
│   ├── raw/                   # Input CSVs from Google Sheets
│   └── output/                # Generated JSON/HTML outputs
├── main.py                    # CLI entry point
├── strategy.py                # Team selection notes
└── .env                       # API token (not in git)
```

## Key Commands

```bash
# Extract fresh data for a round
python main.py --extract --round 1

# Analyze existing data
python main.py --round 1

# Filter to specific match within round
python main.py --round 1 --match 2

# Generate summary across all rounds
python main.py --summary --year 2026

# Set max players per position in charts
python main.py --ppp 20
```

## Data Flow

1. **Six Nations API** (JWT auth) → Raw player stats
2. **Google Sheets CSV** → Fantasy values by player
3. **Name matching** → Merge API data with fantasy values
4. **JSON output** → `data/output/six_nations_stats_YYYY_round_N.json`
5. **Diff comparison** → Track changes between extracts
6. **Analysis** → Interactive Plotly HTML charts

## Key Modules

### src/api/client.py
- `extract_six_nations_stats()` - Fetches all players from API with pagination
- `correct_player_position()` - Fixes position numbering from API
- Handles JWT token validation

### src/processors/fantasy_value_extractor.py
- `match_player_name()` - Intelligent name matching (handles "G. Alldritt" vs "Gregory Alldritt")
- Supports compound surnames (van der, de, du, le)
- `combine_with_api_data()` - Merges fantasy values with player stats

### analysis/analyse_stats.py
- `run_analysis()` - Main entry point for all visualizations
- `calculate_fantasy_points()` - Position-dependent scoring
- Multiple chart functions: `ppm_vs_player_bar_chart()`, `plot_player_points_breakdown()`, etc.
- Filter functions: `filter_by_team()`, `filter_by_value()`, `filter_by_supersubs()`

### config/settings.py
- `API_BASE_URL` = 'https://fantasy.sixnationsrugby.com/v1'
- `FIXTURES` - Hard-coded match data with scores
- `TEAM_COLOURS` - RGB hex colors for each nation

## Data Models

Player data structure (JSON):
```python
{
  "name": "G. Alldritt",
  "club": "France",
  "position": "Back Row",
  "stats": {
    "matches_played": 80,
    "points": "71.0",
    "tries": 1,
    "tackles": 18,
    # ... 18 stat categories
  },
  "fantasy_value": 17.0,
  "fantasy_position_group": "Back Row"
}
```

## Authentication

- JWT tokens from Six Nations (30-day validity)
- Token stored in `.env` as `API_TOKEN`
- Tokens decoded without signature verification (claims inspection only)

## Fantasy Scoring Notes

- **Points**: Come directly from the API (pre-calculated by Six Nations)
- **PPM**: Points per minute (key efficiency metric for analysis)
- **Supersubs**: 3× points multiplier for bench players
- **Budget cap**: 230 points for team selection

**Note**: `calculate_fantasy_points()` in analyse_stats.py reconstructs scoring by category (tries, tackles, etc.) but is only used for the `plot_points_distribution_by_category` visualization. Most analysis uses the API-provided `points` directly.

## CI/CD

- **nightly-extract.yml**: Runs 2am UTC daily, extracts all rounds, commits changes
- **pages.yml**: Deploys HTML reports to GitHub Pages
- **ci.yml**: Basic CI on push/PR

## Common Development Tasks

- **Add new visualization**: Add function to `analysis/analyse_stats.py`, call from `run_analysis()`
- **Update fixtures**: Edit `FIXTURES` dict in `config/settings.py`
- **Add new stat category**: Update `PARAM_MAP` in `src/api/constants.py`
- **Debug name matching**: Check `match_player_name()` in `fantasy_value_extractor.py`

## Output Files

- `six_nations_stats_YYYY_round_N.json` - Full round data
- `extract_diff_YYYY_round_N.json` - Changes from previous extract
- `*.html` - Interactive Plotly visualizations (in project root)
- `6N_combined_stats.csv` - Flattened data for external analysis
