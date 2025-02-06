# config/settings.py

import os
from dotenv import load_dotenv

print(os.getcwd())
load_dotenv()
#print(os.getenv('API_TOKEN'))


# API Configuration
API_BASE_URL = 'https://fantasy.sixnationsrugby.com/v1'
API_ENDPOINTS = {
    'stats': f'{API_BASE_URL}/private/stats',
    'user': f'{API_BASE_URL}/private/user'
}

# Token (valid for 30 days from issue)
#API_TOKEN = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpYXQiOjE3Mzg3ODgwMjQsImV4cCI6MTc0MTIwNzIyNCwianRpIjoiaUF1XC8zTng1MjJ0eUdPdlRqRWdLd3c9PSIsImlzcyI6Imh0dHBzOlwvXC9mYW50YXN5LnNpeG5hdGlvbnNydWdieS5jb21cL202biIsInN1YiI6eyJpZCI6IjQ0ODg1MyIsIm1haWwiOiJuaWNranNtaXRoMTBAZ21haWwuY29tIiwibWFuYWdlciI6IkVsIEJpZ2kiLCJpZGwiOiIxIiwiaWRnIjoiNDY3NzMiLCJmdXNlYXUiOiJFdXJvcGVcL0xvbmRvbiIsIm1lcmNhdG8iOjAsImlkamciOiI0Nzk1NDEiLCJpc2FkbWluY2xpZW50IjpmYWxzZSwiaXNhZG1pbiI6ZmFsc2UsImlzc3VwZXJhZG1pbiI6ZmFsc2UsImludml0ZSI6ZmFsc2UsInZpcCI6ZmFsc2UsImlkZW50aXR5IjoiNjAwIiwiaWdub3JlY29kZSI6ZmFsc2UsImNvZGUiOiI2MDAuMiIsImNvZGVGNSI6IjYwMC4yMSIsImRlY28iOjN9fQ.gTdeTqIk7-dNZPXZuvC7_V7tKxEweoMOV1xjB7ze0Fc"
API_TOKEN = os.getenv('API_TOKEN')
print(API_TOKEN)
#exit()

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

def get_input_filename(matchday: int) -> str:
    """Generate input filename based on matchday"""
    return f'{RAW_DATA_DIR}/Fantasy 6 Nations.xlsx - Round {matchday}.csv'

def get_output_filename(matchday: int) -> str:
    """Generate output filename based on matchday"""
    return f'{OUTPUT_DATA_DIR}/six_nations_stats_round_{matchday}.json'