#!/usr/bin/env python3
"""
Six Nations Fantasy Rugby - CLI Entry Point

Extracts player statistics and fantasy values from the Six Nations API,
tracks changes between extracts, and generates interactive Plotly visualizations
to support fantasy team selection.

Usage:
    python main.py --extract --round 1    # Extract fresh data for round 1
    python main.py --round 1              # Analyze existing round 1 data
    python main.py --round 1 --match 2    # Filter to a specific match
    python main.py --summary --year 2026  # Generate cross-round summary
"""

import json
import argparse
from pathlib import Path
from src.auth.token_validator import display_token_info
from src.processors.api_stats_extractor import get_six_nations_stats
from src.processors.fantasy_value_extractor import extract_fantasy_values, combine_with_api_data
from src.api.client import extract_player_values
from src.processors.extract_summary import compare_extract, generate_summary
from analysis.analyse_stats import run_analysis
from config.settings import (
    API_TOKEN,
    RAW_DATA_DIR,
    OUTPUT_DATA_DIR,
    YEAR,
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
                       default=None,
                       choices=range(1, 6),
                       help='Specify which round to process (1-5). Omit for all rounds.')
    parser.add_argument('--match',
                       type=int,
                       default=None,
                       choices=range(1, 4),
                       help='Filter to a specific match (1-3). Requires --round.')
    parser.add_argument('--ppp',
                       type=int,
                       default=15,
                       help='Max players per position on breakdown charts (default: 15).')
    parser.add_argument('--year',
                       type=int,
                       default=YEAR,
                       help=f'Filter analysis to a specific year (default: {YEAR}).')
    parser.add_argument('--summary',
                       action='store_true',
                       default=False,
                       help='Generate extract summary from per-round diffs.')
    return parser.parse_args()

def main():
    """Main entry point for the stats extraction"""
    try:
        args = parse_arguments()
        # Ensure directories exist
        setup_directories()
        if args.extract:
            print(f"Token (first 10 chars): {API_TOKEN[:10] if API_TOKEN else 'No token found'}")
            # Check token validity
            if not display_token_info(API_TOKEN):
                print("Token validation failed. Exiting.")
                return
            #exit(99)
            # Extract API data
            round_stats = get_six_nations_stats(extract_data=True, token=API_TOKEN, matchday=args.round)

            output_file = Path(get_output_filename(args.round))

            # Load previous data for diff comparison
            previous_data = None
            if output_file.exists():
                with open(output_file) as f:
                    previous_data = json.load(f)

            # Extract player values from API (pass player names for lookup)
            player_names = [p['name'] for p in round_stats['players']]
            player_values = extract_player_values(API_TOKEN, player_names)

            # Add values to player stats
            combined_data = round_stats.copy()
            matched = 0
            for player in combined_data['players']:
                player_name = player['name']
                if player_name in player_values:
                    player['fantasy_value'] = player_values[player_name]
                    matched += 1

            print(f"Matched values for {matched}/{len(combined_data['players'])} players")

            # Save combined data
            with open(output_file, 'w') as f:
                json.dump(combined_data, f, indent=2)
            print(f"Successfully saved combined data to {output_file}")

            # Save per-round diff
            diff = compare_extract(previous_data, combined_data, args.round, args.year)
            diff_file = Path(OUTPUT_DATA_DIR) / f"extract_diff_{args.year}_round_{args.round}.json"
            with open(diff_file, 'w') as f:
                json.dump(diff, f, indent=2)
            print(f"Diff saved to {diff_file}")

        if args.summary:
            generate_summary(args.year, OUTPUT_DATA_DIR)
            return

        if args.match and not args.round:
            print("Error: --match requires --round to be specified.")
            return

        # run the plots
        run_analysis(round=args.round, match=args.match, max_players_per_position=args.ppp, year=args.year)                
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
