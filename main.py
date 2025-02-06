#!/usr/bin/env python3
print("Starting script...")
import jwt
print("JWT imported")
print(jwt.__file__)

"""
Six Nations Fantasy Rugby Stats Extractor

Description:
    Extracts player statistics from the Six Nations Fantasy Rugby API
    and combines with fantasy values from spreadsheet data.
    Spreadsheet:
    https://docs.google.com/spreadsheets/d/1L77DVaq1ILyRjT9R5zoo5aiW_kbvXiOT/edit?gid=1409607056#gid=1409607056
"""

import json
from pathlib import Path
from src.auth.token_validator import display_token_info
from src.processors.api_stats_extractor import analyze_six_nations
from src.processors.fantasy_value_extractor import extract_fantasy_values, combine_with_api_data
from config.settings import (
    API_TOKEN, 
    RAW_DATA_DIR, 
    OUTPUT_DATA_DIR,
    get_input_filename,
    get_output_filename
)


def setup_directories():
    """Create necessary data directories if they don't exist"""
    Path(RAW_DATA_DIR).mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_DATA_DIR).mkdir(parents=True, exist_ok=True)

def main():
    """Main entry point for the stats extraction"""
    try:
        # Ensure directories exist
        setup_directories()
        print(f"Token (first 10 chars): {API_TOKEN[:10] if API_TOKEN else 'No token found'}")
        # Check token validity
        if display_token_info(API_TOKEN):
            # Set which round to process
            matchday = 1
            
            # Extract API data
            api_data = analyze_six_nations(extract_data=True, token=API_TOKEN, matchday=matchday)
            
            # Extract fantasy spreadsheet data
            input_file = Path(get_input_filename(matchday))
            if input_file.exists():
                fantasy_values = extract_fantasy_values(str(input_file))
                
                # Combine the data
                combined_data = combine_with_api_data(api_data, fantasy_values)
                
                # Save combined data
                output_file = get_output_filename(matchday)
                with open(output_file, 'w') as f:
                    json.dump(combined_data, f, indent=2)
                print(f"Successfully saved combined data to {output_file}")
            else:
                print(f"Warning: Fantasy values file not found at {input_file}")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()