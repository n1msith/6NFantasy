# src/api/constants.py

# API parameter mapping
PARAM_MAP = {
    'nom': 'name',
    'club': 'club',
    'position': 'position',
    'journee': 'matchday',
    'nb_matchs': 'matches_played',
    'critere_15': 'man_of_match',
    'critere_4': 'meters_carried',
    'critere_3': 'defenders_beaten',
    'critere_17': 'offloads',
    'critere_8': 'tries',
    'critere_9': 'assists',
    'critere_10': 'conversions',
    'critere_11': 'penalties',
    'critere_12': 'drop_goals',
    'critere_16': 'scrum_wins',
    'critere_5': 'kick_50_22',
    'critere_1': 'tackles',
    'critere_7': 'breakdown_steals',
    'critere_6': 'lineout_steals',
    'critere_2': 'penalties_conceded',
    'critere_13': 'yellow_cards',
    'critere_14': 'red_cards',
    'critere_18': 'kicks_retained',  # New in 2026
    'moyenne_points': 'points'
}

# Normalise club/country names (API sometimes returns French names)
CLUB_NAME_MAPPING = {
    'Angleterre': 'England',
    'Écosse': 'Scotland',
    'Ecosse': 'Scotland',
    'Irlande': 'Ireland',
    'Italie': 'Italy',
    'Pays de Galles': 'Wales',
    # Already English - pass through
    'England': 'England',
    'Scotland': 'Scotland',
    'Ireland': 'Ireland',
    'Italy': 'Italy',
    'Wales': 'Wales',
    'France': 'France',
}

# Mapping to correct incorrect position numbers from the website
POSITION_MAPPING  = {
    '13': 'Hooker',
    '12': 'Prop',
    '11': 'Second Row',
    '10': 'Back Row',
    '9': 'Scrum-Half',
    '8': 'Fly-Half',
    '7': 'Centre',
    '6': 'Back Three'
}