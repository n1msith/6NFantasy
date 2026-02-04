# config/settings.py

import os
from dotenv import load_dotenv

load_dotenv()

YEAR = 2026

# API Configuration
API_BASE_URL = 'https://fantasy.sixnationsrugby.com/v1'
API_ENDPOINTS = {
    'stats': f'{API_BASE_URL}/private/stats',
    'user': f'{API_BASE_URL}/private/user'
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

FIXTURES = {
    2025: {
        1: [('France', 'Wales'), ('Scotland', 'Italy'), ('Ireland', 'England')],
        2: [('Italy', 'Wales'), ('England', 'France'), ('Scotland', 'Ireland')],
        3: [('Wales', 'Ireland'), ('England', 'Scotland'), ('Italy', 'France')],
        4: [('Ireland', 'France'), ('Scotland', 'Wales'), ('England', 'Italy')],
        5: [('Ireland', 'Italy'), ('England', 'Wales'), ('France', 'Scotland')],
    },
    2026: {
        1: [('France', 'Ireland'), ('Italy', 'Scotland'), ('England', 'Wales')],
        2: [('Ireland', 'Italy'), ('Scotland', 'England'), ('Wales', 'France')],
        3: [('England', 'Ireland'), ('Wales', 'Scotland'), ('France', 'Italy')],
        4: [('Ireland', 'Wales'), ('Scotland', 'France'), ('Italy', 'England')],
        5: [('Ireland', 'Scotland'), ('Wales', 'Italy'), ('France', 'England')],
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
