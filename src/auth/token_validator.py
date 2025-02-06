# src/auth/token_validator.py

from datetime import datetime
import jwt
import traceback
print(jwt.__version__)
print(jwt.__file__)

class TokenError(Exception):
    """Custom exception for token-related errors"""
    pass

def check_token_expiry(token: str) -> dict:
    """
    Check token expiration and return detailed information.
    
    Args:
        token (str): JWT token to validate
        
    Returns:
        dict: Validation results including expiry information
    """
    try:
        # Decode token without verification (we don't have the secret)
        decoded = jwt.decode(token, options={"verify_signature": False})
        
        # Get expiration time
        exp_timestamp = decoded.get('exp')
        if not exp_timestamp:
            return {
                'is_valid': False,
                'message': 'Token has no expiration date',
                'expires_in': None,
                'expiry_date': None
            }
            
        exp_date = datetime.fromtimestamp(exp_timestamp)
        now = datetime.now()
        time_until_expiry = exp_date - now
        
        # Convert to hours and minutes
        hours_until_expiry = time_until_expiry.total_seconds() / 3600
        
        if exp_date < now:
            return {
                'is_valid': False,
                'message': f'Token expired on {exp_date.strftime("%Y-%m-%d %H:%M:%S")}',
                'expires_in': None,
                'expiry_date': exp_date
            }
        
        return {
            'is_valid': True,
            'message': 'Token is valid',
            'expires_in': f'{hours_until_expiry:.1f} hours',
            'expiry_date': exp_date
        }
        
    #except PyJWTError:
    #    return {
    #        'is_valid': False,
    #        'message': 'Invalid token format',
    #        'expires_in': None,
    #        'expiry_date': None
    #    }
    except Exception as e:
        print(f"Full exception details:")
        traceback.print_exc()
        return {
            'is_valid': False,
            'message': f'Error validating token: {str(e)}',
            'expires_in': None,
            'expiry_date': None
        }

def validate_token(token: str) -> tuple[bool, str]:
    """
    Validate token format and expiration
    
    Args:
        token (str): JWT token to validate
        
    Returns:
        tuple[bool, str]: (is_valid, message)
    """
    result = check_token_expiry(token)
    return result['is_valid'], result['message']

def display_token_info(token: str) -> bool:
    """
    Display detailed information about a token's validity and expiration.
    
    Args:
        token (str): JWT token to display info for
        
    Returns:
        bool: Token validity
    """
    print("\nToken Validation Results:")
    print("-" * 50)
    
    info = check_token_expiry(token)
    
    if info['is_valid']:
        print("✓ Token is VALID")
        print(f"• Expires in: {info['expires_in']}")
        print(f"• Expiry date: {info['expiry_date'].strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        print("✗ Token is INVALID")
        print(f"• Reason: {info['message']}")
    
    print("-" * 50)
    return info['is_valid']