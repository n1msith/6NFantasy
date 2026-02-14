"""
Betting Odds Scraper for Six Nations Fantasy Rugby

Scrapes match betting odds from web sources and converts them to features
for the prediction model. Falls back to manual entry if scraping fails.

Output format: data/output/betting_odds_{year}.json
"""

import json
import re
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_OUTPUT = PROJECT_ROOT / "data" / "output"

# Team name normalisations (bookmaker names -> our names)
TEAM_ALIASES = {
    'rep. of ireland': 'Ireland', 'rep of ireland': 'Ireland',
    'republic of ireland': 'Ireland',
    'eng': 'England', 'ire': 'Ireland', 'fra': 'France',
    'wal': 'Wales', 'sco': 'Scotland', 'ita': 'Italy',
}


def normalise_team(name: str) -> str:
    """Normalise team name to our standard names."""
    clean = name.strip().lower()
    if clean in TEAM_ALIASES:
        return TEAM_ALIASES[clean]
    # Title case match
    for standard in ['England', 'France', 'Ireland', 'Italy', 'Scotland', 'Wales']:
        if standard.lower() in clean:
            return standard
    return name.strip().title()


def fractional_to_probability(odds_str: str) -> float:
    """Convert fractional odds (e.g. '3/1') to implied probability."""
    try:
        if '/' in odds_str:
            num, den = odds_str.split('/')
            decimal = 1 + float(num) / float(den)
        else:
            decimal = float(odds_str)
        return 1.0 / decimal
    except (ValueError, ZeroDivisionError):
        return 0.5


def decimal_to_probability(odds: float) -> float:
    """Convert decimal odds to implied probability."""
    try:
        return 1.0 / odds if odds > 0 else 0.5
    except ZeroDivisionError:
        return 0.5


def scrape_oddspedia(year: int = 2026) -> Optional[list]:
    """Try to scrape match odds from Oddspedia (more scrape-friendly than others).

    Returns list of match dicts or None if scraping fails.
    """
    url = f"https://oddspedia.com/rugby-union/europe/six-nations"

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                       '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    }

    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, 'html.parser')

        matches = []
        # Look for match containers - Oddspedia uses various class patterns
        match_elements = soup.find_all(['div', 'a'], class_=re.compile(r'match|event|game', re.I))

        for elem in match_elements:
            text = elem.get_text(separator=' ', strip=True)
            # Try to extract team names and odds from text
            for home_team in ['England', 'France', 'Ireland', 'Italy', 'Scotland', 'Wales']:
                for away_team in ['England', 'France', 'Ireland', 'Italy', 'Scotland', 'Wales']:
                    if home_team != away_team and home_team in text and away_team in text:
                        # Found a match - try to extract odds
                        odds_pattern = r'(\d+\.\d+)'
                        odds_found = re.findall(odds_pattern, text)

                        if len(odds_found) >= 2:
                            home_odds = float(odds_found[0])
                            away_odds = float(odds_found[-1])
                            matches.append({
                                'home': home_team,
                                'away': away_team,
                                'home_win_prob': round(decimal_to_probability(home_odds), 3),
                                'away_win_prob': round(decimal_to_probability(away_odds), 3),
                                'total_points': 45.0,  # Default if not found
                                'handicap': 0.0,
                            })

        return matches if matches else None

    except Exception as e:
        print(f"Oddspedia scrape failed: {e}")
        return None


def scrape_bettingodds(year: int = 2026) -> Optional[list]:
    """Try to scrape from bettingodds.com."""
    url = "https://www.bettingodds.com/rugby-union/six-nations"

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                       '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    }

    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, 'html.parser')
        matches = []

        # Look for match/event elements
        for row in soup.find_all(['tr', 'div', 'article'], class_=re.compile(r'event|match|fixture', re.I)):
            text = row.get_text(separator=' ', strip=True)

            # Find "Team1 v Team2" or "Team1 vs Team2" patterns
            vs_match = re.search(
                r'(England|France|Ireland|Italy|Scotland|Wales)\s+v(?:s)?\s+(England|France|Ireland|Italy|Scotland|Wales)',
                text, re.I
            )
            if vs_match:
                home = normalise_team(vs_match.group(1))
                away = normalise_team(vs_match.group(2))

                # Extract fractional odds like "3/1" or decimal odds
                frac_odds = re.findall(r'(\d+/\d+)', text)
                dec_odds = re.findall(r'(\d+\.\d+)', text)

                if len(frac_odds) >= 2:
                    home_prob = fractional_to_probability(frac_odds[0])
                    away_prob = fractional_to_probability(frac_odds[-1])
                elif len(dec_odds) >= 2:
                    home_prob = decimal_to_probability(float(dec_odds[0]))
                    away_prob = decimal_to_probability(float(dec_odds[-1]))
                else:
                    home_prob = 0.5
                    away_prob = 0.5

                matches.append({
                    'home': home,
                    'away': away,
                    'home_win_prob': round(home_prob, 3),
                    'away_win_prob': round(away_prob, 3),
                    'total_points': 45.0,
                    'handicap': 0.0,
                })

        return matches if matches else None

    except Exception as e:
        print(f"BettingOdds scrape failed: {e}")
        return None


def generate_template(year: int = 2026) -> dict:
    """Generate a template betting odds JSON for manual entry."""
    from config.settings import FIXTURES

    fixtures = FIXTURES.get(year, {})
    output = {'year': year, 'rounds': []}

    for round_num in sorted(fixtures.keys()):
        round_data = {'round': round_num, 'matches': []}
        for home, away, score in fixtures[round_num]:
            round_data['matches'].append({
                'home': home,
                'away': away,
                'home_win_prob': 0.5,
                'away_win_prob': 0.5,
                'total_points': 45.0,
                'handicap': 0.0,
                '_comment': f'Enter odds as probabilities (0-1). home_win_prob + away_win_prob + draw ~ 1.0'
            })
        output['rounds'].append(round_data)

    return output


def assign_odds_to_rounds(matches: list, year: int = 2026) -> dict:
    """Map scraped match odds to the correct rounds using fixture data."""
    from config.settings import FIXTURES

    fixtures = FIXTURES.get(year, {})
    output = {'year': year, 'rounds': []}

    # Index scraped odds by team pairing
    odds_lookup = {}
    for m in matches:
        key = tuple(sorted([m['home'], m['away']]))
        odds_lookup[key] = m

    for round_num in sorted(fixtures.keys()):
        round_data = {'round': round_num, 'matches': []}
        for home, away, score in fixtures[round_num]:
            key = tuple(sorted([home, away]))
            if key in odds_lookup:
                scraped = odds_lookup[key]
                # Ensure home/away probabilities match our fixture orientation
                if scraped['home'] == home:
                    match_odds = scraped.copy()
                else:
                    match_odds = {
                        'home': home, 'away': away,
                        'home_win_prob': scraped['away_win_prob'],
                        'away_win_prob': scraped['home_win_prob'],
                        'total_points': scraped.get('total_points', 45.0),
                        'handicap': -scraped.get('handicap', 0.0),
                    }
            else:
                match_odds = {
                    'home': home, 'away': away,
                    'home_win_prob': 0.5, 'away_win_prob': 0.5,
                    'total_points': 45.0, 'handicap': 0.0,
                }
            round_data['matches'].append(match_odds)
        output['rounds'].append(round_data)

    return output


def scrape_and_save(year: int = 2026) -> Path:
    """Main entry point: scrape odds from available sources and save to JSON.

    Tries multiple sources in order, falls back to template if all fail.
    """
    output_path = DATA_OUTPUT / f"betting_odds_{year}.json"

    print(f"Scraping Six Nations {year} betting odds...")

    # Try scrapers in order of reliability
    scrapers = [
        ('Oddspedia', lambda: scrape_oddspedia(year)),
        ('BettingOdds.com', lambda: scrape_bettingodds(year)),
    ]

    matches = None
    for name, scraper_fn in scrapers:
        print(f"  Trying {name}...", end=' ')
        matches = scraper_fn()
        if matches:
            print(f"found {len(matches)} matches")
            break
        else:
            print("no data")

    if matches:
        odds_data = assign_odds_to_rounds(matches, year)
        source = name
    else:
        print("\n  All scrapers failed. Generating template for manual entry.")
        print(f"  Edit {output_path} with real odds from your preferred bookmaker.")
        odds_data = generate_template(year)
        source = 'template'

    odds_data['source'] = source
    odds_data['_instructions'] = (
        "Probabilities should sum to ~1.0 per match (home + away + draw margin). "
        "handicap is from the home team perspective (positive = home favoured). "
        "total_points is the expected combined match score."
    )

    with open(output_path, 'w') as f:
        json.dump(odds_data, f, indent=2)

    print(f"\nSaved to: {output_path}")
    return output_path


if __name__ == '__main__':
    scrape_and_save(2026)
