# config/settings.py

import os
from dotenv import load_dotenv

load_dotenv()

YEAR = 2026

# API Configuration
API_BASE_URL = 'https://fantasy.sixnationsrugby.com/v1'
API_ENDPOINTS = {
    'stats': f'{API_BASE_URL}/private/stats',
    'user': f'{API_BASE_URL}/private/user',
    'players': f'{API_BASE_URL}/private/searchjoueurs'
}

# Token (valid for 30 days from issue)
#API_TOKEN = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpYXQiOjE3Mzg3ODgwMjQsImV4cCI6MTc0MTIwNzIyNCwianRpIjoiaUF1XC8zTng1MjJ0eUdPdlRqRWdLd3c9PSIsImlzcyI6Imh0dHBzOlwvXC9mYW50YXN5LnNpeG5hdGlvbnNydWdieS5jb21cL202biIsInN1YiI6eyJpZCI6IjQ0ODg1MyIsIm1haWwiOiJuaWNranNtaXRoMTBAZ21haWwuY29tIiwibWFuYWdlciI6IkVsIEJpZ2kiLCJpZGwiOiIxIiwiaWRnIjoiNDY3NzMiLCJmdXNlYXUiOiJFdXJvcGVcL0xvbmRvbiIsIm1lcmNhdG8iOjAsImlkamciOiI0Nzk1NDEiLCJpc2FkbWluY2xpZW50IjpmYWxzZSwiaXNhZG1pbiI6ZmFsc2UsImlzc3VwZXJhZG1pbiI6ZmFsc2UsImludml0ZSI6ZmFsc2UsInZpcCI6ZmFsc2UsImlkZW50aXR5IjoiNjAwIiwiaWdub3JlY29kZSI6ZmFsc2UsImNvZGUiOiI2MDAuMiIsImNvZGVGNSI6IjYwMC4yMSIsImRlY28iOjN9fQ.gTdeTqIk7-dNZPXZuvC7_V7tKxEweoMOV1xjB7ze0Fc"
API_TOKEN = os.getenv('API_TOKEN')

API_DEFAULT_HEADERS = {
    'Content-Type': 'application/json',
    'X-Access-Key': '600@16.11@'
}

# Request Configuration
REQUEST_TIMEOUT = 10
PAGE_SIZE = 10

# File Path Configuration
DATA_DIR = 'data'
RAW_DATA_DIR = f'{DATA_DIR}/raw'
OUTPUT_DATA_DIR = f'{DATA_DIR}/output'

# FIXTURES[year][round] = [(home, away, score), ...]
# score is (home_pts, away_pts) or None if not yet played.
FIXTURES = {
    2025: {
        1: [('France', 'Wales', (43, 0)), ('Scotland', 'Italy', (31, 19)), ('Ireland', 'England', (27, 22))],
        2: [('Italy', 'Wales', (22, 15)), ('England', 'France', (26, 25)), ('Scotland', 'Ireland', (18, 32))],
        3: [('Wales', 'Ireland', (18, 27)), ('England', 'Scotland', (16, 15)), ('Italy', 'France', (24, 73))],
        4: [('Ireland', 'France', (27, 42)), ('Scotland', 'Wales', (35, 29)), ('England', 'Italy', (47, 24))],
        5: [('Italy', 'Ireland', (17, 22)), ('Wales', 'Englang', (14, 68)), ('France', 'Scotland', (35, 16))],
    },
    2026: {
        1: [('France', 'Ireland', None), ('Italy', 'Scotland', None), ('England', 'Wales', None)],
        2: [('Ireland', 'Italy', None), ('Scotland', 'England', None), ('Wales', 'France', None)],
        3: [('England', 'Ireland', None), ('Wales', 'Scotland', None), ('France', 'Italy', None)],
        4: [('Ireland', 'Wales', None), ('Scotland', 'France', None), ('Italy', 'England', None)],
        5: [('Ireland', 'Scotland', None), ('Wales', 'Italy', None), ('France', 'England', None)],
    },
}

FIXTURE_DATES = {
    2026: {
        1: 'Thu 5 Feb / Sat 7 Feb',
        2: 'Sat 14 Feb / Sun 15 Feb',
        3: 'Sat 21 Feb / Sun 22 Feb',
        4: 'Fri 6 Mar / Sat 7 Mar',
        5: 'Sat 14 Mar',
    },
}

TEAM_COLOURS = {
    'France':   '#003399',
    'Wales':    '#D4213D',
    'Scotland': '#005EB8',
    'Italy':    '#008C45',
    'Ireland':  '#169B62',
    'England':  '#E8003A',
}

# Fantasy scoring rules by year
# Keys match DataFrame column names from the API
SCORING_RULES = {
    2025: {
        # Attacking
        'tries_back': 10,
        'tries_forward': 15,
        'assists': 4,
        'conversions': 2,
        'penalties': 3,
        'drop_goals': 5,
        'defenders_beaten': 2,
        'meters_carried': 0.1,  # 1 pt per 10m
        'kick_50_22': 7,
        'offloads': 2,
        'scrum_wins': 1,
        # Defensive
        'tackles': 1,
        'breakdown_steals': 5,
        'lineout_steals': 7,
        'penalties_conceded': -1,
        # General
        'man_of_match': 15,
        'yellow_cards': -5,
        'red_cards': -8,
        # Budget
        'budget': 230,
    },
    2026: {
        # Attacking
        'tries_back': 10,
        'tries_forward': 15,
        'assists': 4,
        'conversions': 2,
        'penalties': 3,
        'drop_goals': 5,
        'defenders_beaten': 2,
        'meters_carried': 0.1,  # 1 pt per 10m
        'kick_50_22': 7,
        'kicks_retained': 2,  # NEW in 2026
        'offloads': 2,
        'scrum_wins': 1,
        # Defensive
        'tackles': 1,
        'breakdown_steals': 5,
        'lineout_steals': 7,
        'penalties_conceded': -1,
        # General
        'man_of_match': 15,
        'yellow_cards': -5,
        'red_cards': -8,
        # Budget
        'budget': 200,
    },
}


def get_scoring_rules(year=None) -> dict:
    """Get scoring rules for a given year. Falls back to most recent year if not found."""
    if year is None:
        year = YEAR
    if year in SCORING_RULES:
        return SCORING_RULES[year]
    # Fall back to most recent year
    return SCORING_RULES[max(SCORING_RULES.keys())]


def get_fixtures(year: int = None) -> dict:
    """Get fixtures for a given year. Falls back to empty dict if year not found."""
    if year is None:
        year = YEAR
    return FIXTURES.get(year, {})


def get_input_filename(matchday: int) -> str:
    """Generate input filename based on matchday, using default YEAR"""
    return f'{RAW_DATA_DIR}/Fantasy 6 Nations {YEAR}.xlsx - Round {matchday}.csv'

def get_output_filename(matchday: int) -> str:
    """Generate output filename based on matchday, using default YEAR"""
    return f'{OUTPUT_DATA_DIR}/six_nations_stats_{YEAR}_round_{matchday}.json'
