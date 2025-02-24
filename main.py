#!/usr/bin/env python3
"""
Six Nations Fantasy Rugby Stats Extractor

Description:
    Extracts player statistics from the Six Nations Fantasy Rugby API
    and combines with fantasy values from spreadsheet data.
    Spreadsheet:
    https://docs.google.com/spreadsheets/d/1L77DVaq1ILyRjT9R5zoo5aiW_kbvXiOT/edit?gid=1409607056#gid=1409607056
"""

import json
import argparse
from pathlib import Path
from src.auth.token_validator import display_token_info
from src.processors.api_stats_extractor import get_six_nations_stats
from src.processors.fantasy_value_extractor import extract_fantasy_values, combine_with_api_data
from analysis.analyse_stats import run_analysis
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

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Six Nations Fantasy Rugby Stats Extractor')
    parser.add_argument('--extract', 
                       action='store_true',
                       default=False,
                       help='Set to true to extract new data from API')
    parser.add_argument('--round',
                       type=int,
                       default=1,
                       choices=range(1, 6),
                       help='Specify which round to process (1-5)')
    return parser.parse_args()

def main():
    """Main entry point for the stats extraction"""
    try:
        args = parse_arguments()
        # Ensure directories exist
        setup_directories()
        print(f"Token (first 10 chars): {API_TOKEN[:10] if API_TOKEN else 'No token found'}")
        # Check token validity
        if display_token_info(API_TOKEN):
            
            # Extract API data (optional based on args)
            round_stats = get_six_nations_stats(extract_data=args.extract, token=API_TOKEN, matchday=args.round)
            
            # Extract fantasy spreadsheet data
            input_file = Path(get_input_filename(args.round))
            output_file = get_output_filename(args.round)
            
            if input_file.exists():
                fantasy_values = extract_fantasy_values(str(input_file))
                
                # Combine the data
                combined_data = combine_with_api_data(round_stats, fantasy_values, args.round)
                
                # Save combined data
                with open(output_file, 'w') as f:
                    json.dump(combined_data, f, indent=2)
                print(f"Successfully saved combined data to {output_file}")
            else:
                print(f"Warning: Fantasy values file not found at {input_file}")
                with open(output_file, 'w') as f:
                    json.dump(round_stats, f, indent=2)
                print(f"Successfully saved combined data to {output_file}")                
            
            # run the plots
            run_analysis()                
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
