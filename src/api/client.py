# src/api/client.py

import json
import requests
from datetime import datetime
from typing import List, Dict, Any

from ..auth.token_validator import TokenError, validate_token
from .constants import PARAM_MAP, POSITION_MAPPING
from config.settings import (
    API_ENDPOINTS, 
    API_DEFAULT_HEADERS, 
    REQUEST_TIMEOUT, 
    PAGE_SIZE,
    get_output_filename
)

def correct_player_position(player: Dict[str, Any]) -> Dict[str, Any]:
    """
    Correct player position based on the predefined mapping
    
    Args:
        player: Player data dictionary
        
    Returns:
        Dict with corrected position
    """
    corrected_player = player.copy()
    position = corrected_player['position']
    if position in POSITION_MAPPING:
        corrected_player['position'] = POSITION_MAPPING[position]
    return corrected_player

def extract_six_nations_stats(token: str, matchday: int = 1) -> List[Dict[str, Any]]:
    """
    Extract stats from API and save to JSON with improved error handling
    
    Args:
        token (str): Valid JWT token for authentication
        matchday (int): Round number to extract (default: 1)
        
    Returns:
        List[Dict[str, Any]]: List of player statistics
        
    Raises:
        TokenError: If token is invalid or expired
        ValueError: If no data is collected
        requests.RequestException: For network related errors
    """
    
    # First validate the token
    is_valid, message = validate_token(token)
    if not is_valid:
        raise TokenError(f"Token validation failed: {message}")
    
    # Prepare headers with token
    headers = {
        **API_DEFAULT_HEADERS,
        'Authorization': f'Token {token}'
    }

    all_data = []
    page = 0

    try:
        print(f"\nStarting data extraction from Six Nations API for round {matchday}...")
        while True:
            payload = {
                "credentials": {
                    "critereRecherche": {
                        "nom": "",
                        "club": "",
                        "position": "",
                        "journee": str(matchday)
                    },
                    "critereTri": "moyenne_points",
                    "loadSelect": 1,
                    "pageIndex": page,
                    "pageSize": PAGE_SIZE
                }
            }

            try:
                response = requests.post(
                    API_ENDPOINTS['stats'] + '?lg=en',
                    headers=headers,
                    json=payload,
                    timeout=REQUEST_TIMEOUT
                )
                
                # Check for specific status codes
                if response.status_code == 401:
                    raise TokenError("Authentication failed. Token may be expired or invalid.")
                elif response.status_code == 403:
                    raise TokenError("Access forbidden. Check if token has correct permissions.")
                
                response.raise_for_status()
                
                data = response.json()
                
                if not data.get('joueurs'):
                    if page == 0:
                        raise ValueError("No player data returned. API response may have changed format.")
                    break

                for player in data['joueurs']:
                    translated_stats = {}
                    for stat in player['criteres']:
                        stat_name = PARAM_MAP.get(stat['nom'], stat['nom'])
                        translated_stats[stat_name] = stat['value']

                    player_data = {
                        'name': player['nomaffiche'],
                        'club': player['club'],
                        'position': player['position'],
                        'stats': translated_stats
                    }
                    
                    # Correct the position before adding to all_data
                    corrected_player = correct_player_position(player_data)
                    all_data.append(corrected_player)

                page += 1

            except requests.exceptions.Timeout:
                print(f"Request timeout on page {page}. Retrying...")
                continue
            except requests.exceptions.RequestException as e:
                print(f"Network error on page {page}: {str(e)}")
                break

        # Only save if we got some data
        if all_data:
            output_file = get_output_filename(matchday)
            
            # Create the final data structure
            output_data = {
                "extraction_date": datetime.now().strftime("%Y-%m-%d"),
                "round": matchday,
                "players": [
                    {**player, "round": matchday} for player in all_data
                ]
            }
            
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"Successfully saved data for {len(all_data)} players to {output_file}")
        else:
            raise ValueError("No data was collected")

    except TokenError as e:
        print("\nToken Error:")
        print(f"Error: {str(e)}")
        print("\nTo get a new token:")
        print("1. Go to fantasy.sixnationsrugby.com and log in")
        print("2. Open Developer Tools (F12)")
        print("3. Go to the Network tab")
        print("4. Look for requests to URLs containing 'private' or 'stats'")
        print("5. Find the token in the Authorization header or authentication response")
        raise

    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        raise

    return all_data