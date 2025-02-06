# src/data/processor.py

import json
from pathlib import Path
from typing import Dict, List, Any, Optional

from ..auth.token_validator import TokenError
from ..api.client import extract_six_nations_stats
from config.settings import get_output_filename

def analyze_six_nations(
    extract_data: bool = False, 
    token: Optional[str] = None,
    matchday: int = 1
) -> List[Dict[str, Any]]:
    """
    Main function with improved error handling
    
    Args:
        extract_data (bool): Whether to extract fresh data from API
        token (Optional[str]): JWT token for authentication if extracting data
        
    Returns:
        List[Dict[str, Any]]: List of player data and statistics
        
    Raises:
        TokenError: If token is invalid when extracting data
        FileNotFoundError: If data file doesn't exist when not extracting
        ValueError: If token is needed but not provided
    """
    try:
        if extract_data:
            if not token:
                raise ValueError("Token required for data extraction. Please provide a valid token.")
            data = extract_six_nations_stats(token, matchday)
        else:
            data_file = Path(get_output_filename(matchday))
            if not data_file.exists():
                raise FileNotFoundError(
                    "Stats file not found. Either:\n"
                    "1. Run with extract_data=True and provide a token, or\n"
                    "2. Ensure six_nations_stats.json exists in the current directory"
                )

            with open(data_file) as f:
                data = json.load(f)

        return data

    except TokenError:
        # TokenError already provides detailed instructions
        raise
    except Exception as e:
        print(f"\nError analyzing Six Nations data: {str(e)}")
        raise