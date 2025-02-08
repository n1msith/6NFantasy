# src/data/fantasy_value_extractor.py

import pandas as pd
from typing import List, Dict, Any, Union
from pathlib import Path

def extract_fantasy_values(file_path: str) -> List[Dict[str, Any]]:
    """
    Parse the Fantasy Six Nations CSV file for player values
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        List[Dict[str, Any]]: List of player data with values
    """
    print(f"Starting extraction from value spreadhseet: {file_path}")
    # Read the CSV file
    df = pd.read_csv(file_path, header=None)

    # Initialize list to store player data
    players = []

    # Columns for players and points (0-based indexing)
    player_columns = [2, 4, 6, 8, 10, 12]
    point_columns = [3, 5, 7, 9, 11, 13]

    # Current position group to track
    current_position_group = 'Unknown'

    # Rows to process (rows 5-29 in Google Sheet numbering, which is indices 4-28 in Python)
    for i in range(4, 29):
        row = df.iloc[i]

        # Update position group if column A is not blank
        if not pd.isna(row.iloc[0]) and isinstance(row.iloc[0], str):
            current_position_group = row.iloc[0]

        # Skip referee rows
        if current_position_group == 'Referee':
            continue

        # If column A is blank, set position group to 'Subs'
        if pd.isna(row.iloc[0]):
            current_position_group = 'Subs'

        # Extract players and their points
        for p_col, pt_col in zip(player_columns, point_columns):
            player_name = row.iloc[p_col]
            player_points = row.iloc[pt_col]

            # Validate player name and points
            if (isinstance(player_name, str) and
                not pd.isna(player_points) and
                player_points != 'TBC'):
                try:
                    players.append({
                        'name': player_name.strip(),
                        'fantasy_position_group': current_position_group,
                        'fantasy_value': float(player_points)
                    })
                except (ValueError, TypeError):
                    # Skip if points can't be converted to float
                    continue
    
    print(f"Finished extracting {len(players)} players from value spreadsheet")
    return players


def match_player_name(api_player_name: str, fantasy_names: List[str]) -> str:
    """
    Find the best matching name from fantasy names list
   
    Args:
        api_player_name: API player name
        fantasy_names: List of names from fantasy spreadsheet
   
    Returns:
        Matched name or None if no match found
    """
    fantasy_name = fantasy_names[0]
    
    # Remove any additional info like (C), (SH)
    clean_fantasy_name = fantasy_name.split('(')[0].strip()
    
    # Split names
    api_parts = api_player_name.split('.')
    fantasy_parts = clean_fantasy_name.split()
    
    # Check if last name matches and first initial matches
    if (len(api_parts) > 1 and 
        len(fantasy_parts) > 1):
        
        api_last_name = ' '.join(api_parts[1:]).strip()
        api_first_name_letter = (api_parts[0]).lower()
        # Handle van der cases and compound last names
        fantasy_last_name = ' '.join(fantasy_parts[-2:]) if len(fantasy_parts) > 2 and fantasy_parts[-2].lower() in ['van', 'de', 'du', 'le'] else fantasy_parts[-1]
        fantasy_first_name_letter = (fantasy_parts[0][0]).lower()
        
        # debug
        if api_last_name == "some_name":
            print(api_parts)
            print(api_last_name)
            print(api_first_name_letter)
            
            print(fantasy_parts)
            print(fantasy_last_name)
            print(fantasy_first_name_letter)
            #exit()
        
        if (fantasy_last_name.lower() in api_last_name.lower() and
            api_first_name_letter == fantasy_first_name_letter):
            return fantasy_name
    
    return None

def combine_with_api_data(api_data: Dict[str, Any],
                         fantasy_values: List[Dict[str, Any]],
                         round_number: int) -> Dict[str, Any]:
    """
    Combine API data with fantasy values for a specific round
   
    Args:
        api_data: Dictionary containing extraction_date, round, and players list
        fantasy_values: List of player fantasy values (selected players)
        round_number: The round number to update
       
    Returns:
        Updated API data with fantasy values added
    """
    # Verify we have the correct round
    if api_data['round'] != round_number:
        raise ValueError(f"API data is for round {api_data['round']}, but round {round_number} was requested")

    # Tracking variables
    total_fantasy_players = len(fantasy_values)
    combined_players = 0
    missed_players = 0
   
    # Create a copy of api_data to avoid modifying the original
    updated_api_data = api_data.copy()
    
    # Loop through fantasy names
    for fantasy_player in fantasy_values:
        # Try to find a matching player in API data
        matched_player = None
        for api_player in updated_api_data['players']:
            matched_name = match_player_name(api_player['name'], [fantasy_player['name']])
            if matched_name:
                matched_player = api_player
                break

        # Update player with fantasy value if matched
        if matched_player:
            matched_player['fantasy_value'] = fantasy_player['fantasy_value']
            matched_player['fantasy_position_group'] = fantasy_player['fantasy_position_group']
            combined_players += 1
        else:
            missed_players += 1
            # Raise an exception if no match found to highlight the need for investigation
            #raise ValueError(f"No API player found for {fantasy_player['name']} - requires investigation")
            print(f"No API player found for {fantasy_player['name']} - requires investigation")
            continue
            
        if matched_player['position'].lower() != fantasy_player['fantasy_position_group'].lower() and fantasy_player['fantasy_position_group'] != "Subs":
            print(f"{matched_player['name']} out of position. Normally {matched_player['position']} but playing at {fantasy_player['fantasy_position_group']}")
   
    # Print summary of combination
    print(f"Fantasy Value Combination Summary:")
    print(f"Total Fantasy Players: {total_fantasy_players}")
    print(f"Players Matched: {combined_players}")
    print(f"Players Missed: {missed_players}")
   
    # Return the updated API data
    return updated_api_data