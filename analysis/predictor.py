"""
Fantasy Rugby Prediction Module

Uses XGBoost to predict player fantasy points based on multi-dimensional features:
- Historical performance & rolling stats
- Betting odds (scraped market data)
- Fixture context (opponent strength, home/away)
- Team form & momentum
- Player form trajectory
- Historical matchups (player vs specific opponent)
- Position-specific interactions
"""

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import xgboost as xgb
except ImportError:
    print("XGBoost not installed. Run: pip install xgboost scikit-learn")
    xgb = None


# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_OUTPUT = PROJECT_ROOT / "data" / "output"


def detect_target_round(year: int = 2026) -> int:
    """Auto-detect the next unplayed round from fixtures data."""
    from config.settings import FIXTURES

    fixtures = FIXTURES.get(year, {})
    for round_num in sorted(fixtures.keys()):
        matches = fixtures[round_num]
        # A round is unplayed if any match has score=None
        if any(score is None for _, _, score in matches):
            return round_num

    # All rounds played - return last round
    return max(fixtures.keys()) if fixtures else 1


def load_round_data(year: int, round_num: int) -> Optional[pd.DataFrame]:
    """Load a single round's data from JSON."""
    filepath = DATA_OUTPUT / f"six_nations_stats_{year}_round_{round_num}.json"

    if not filepath.exists():
        return None

    with open(filepath, 'r') as f:
        data = json.load(f)

    players = data.get('players', [])
    if not players:
        return None

    # Flatten the nested stats structure
    records = []
    for player in players:
        record = {
            'name': player['name'],
            'club': player['club'],
            'position': player['position'],
            'fantasy_value': player.get('fantasy_value', 0),
            'year': year,
            'round': round_num,
        }

        # Add all stats
        stats = player.get('stats', {})
        for key, value in stats.items():
            # Convert empty strings to 0
            if value == '' or value is None:
                record[key] = 0
            elif isinstance(value, str):
                try:
                    record[key] = float(value)
                except ValueError:
                    record[key] = 0
            else:
                record[key] = value

        records.append(record)

    return pd.DataFrame(records)


def load_all_data(years: list[int] = [2025, 2026], max_rounds: int = 5) -> pd.DataFrame:
    """Load all available round data for specified years."""
    all_dfs = []

    for year in years:
        for round_num in range(1, max_rounds + 1):
            df = load_round_data(year, round_num)
            if df is not None and len(df) > 0:
                # Check if there's actual data (not just placeholders)
                if df['points'].sum() > 0:
                    all_dfs.append(df)
                    print(f"Loaded {year} Round {round_num}: {len(df)} players")

    if not all_dfs:
        raise ValueError("No data found!")

    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal records: {len(combined)}")
    return combined


def did_play(row: pd.Series) -> bool:
    """Detect if a player actually played in a round.

    Players who didn't play have 0 across all activity stats.
    """
    activity_cols = ['meters_carried', 'tackles', 'tries', 'assists',
                     'conversions', 'penalties', 'breakdown_steals']
    for col in activity_cols:
        if col in row and row[col] > 0:
            return True
    return False


def filter_played_games(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to only games where players actually played."""
    df = df.copy()
    df['played'] = df.apply(did_play, axis=1)
    return df[df['played']].drop(columns=['played'])


def calculate_dynamic_opponent_strength() -> dict:
    """Calculate opponent strength from actual match results instead of hardcoded values.

    Returns dict of {team: strength_score} where higher = harder to score against.
    Uses points conceded across all available fixture data.
    """
    from config.settings import FIXTURES

    team_stats = {}  # {team: {'scored': [], 'conceded': []}}

    for year in sorted(FIXTURES.keys()):
        for round_num in sorted(FIXTURES[year].keys()):
            for home, away, score in FIXTURES[year][round_num]:
                if score is None:
                    continue
                home_pts, away_pts = score

                for team in [home, away]:
                    if team not in team_stats:
                        team_stats[team] = {'scored': [], 'conceded': []}

                team_stats[home]['scored'].append(home_pts)
                team_stats[home]['conceded'].append(away_pts)
                team_stats[away]['scored'].append(away_pts)
                team_stats[away]['conceded'].append(home_pts)

    if not team_stats:
        return {}

    # Calculate average conceded per team
    avg_conceded = {}
    for team, stats in team_stats.items():
        if stats['conceded']:
            avg_conceded[team] = np.mean(stats['conceded'])

    if not avg_conceded:
        return {}

    # Normalize to 0-1 scale (lowest conceded = 1.0 = hardest)
    min_c = min(avg_conceded.values())
    max_c = max(avg_conceded.values())
    range_c = max_c - min_c if max_c != min_c else 1

    return {team: 1.0 - (avg - min_c) / range_c for team, avg in avg_conceded.items()}


def get_team_results() -> dict:
    """Get all match results per team from fixtures. Returns {team: [list of result dicts]}."""
    from config.settings import FIXTURES

    team_results = {}

    for year in sorted(FIXTURES.keys()):
        for round_num in sorted(FIXTURES[year].keys()):
            for home, away, score in FIXTURES[year][round_num]:
                if score is None:
                    continue
                home_pts, away_pts = score

                for team in [home, away]:
                    if team not in team_results:
                        team_results[team] = []

                team_results[home].append({
                    'year': year, 'round': round_num,
                    'scored': home_pts, 'conceded': away_pts,
                    'is_home': True, 'opponent': away,
                    'won': home_pts > away_pts
                })
                team_results[away].append({
                    'year': year, 'round': round_num,
                    'scored': away_pts, 'conceded': home_pts,
                    'is_home': False, 'opponent': home,
                    'won': away_pts > home_pts
                })

    return team_results


# Calculate once at module level, refreshed on each run
OPPONENT_STRENGTH = calculate_dynamic_opponent_strength()


def get_fixture_context(club: str, year: int, round_num: int) -> dict:
    """Get fixture context for a player's team in a specific round."""
    from config.settings import FIXTURES

    fixtures = FIXTURES.get(year, {}).get(round_num, [])
    for home, away, score in fixtures:
        if club == home:
            opponent = away
            is_home = True
            break
        elif club == away:
            opponent = home
            is_home = False
            break
    else:
        return {
            'opponent': None, 'opponent_strength': 0.5, 'is_home': False,
            'opponent_pts_conceded_avg': 25.0, 'opponent_pts_scored_avg': 25.0,
        }

    # Get detailed opponent stats from results
    team_results = get_team_results()
    opponent_results = team_results.get(opponent, [])

    if opponent_results:
        opp_conceded_avg = np.mean([r['conceded'] for r in opponent_results])
        opp_scored_avg = np.mean([r['scored'] for r in opponent_results])
    else:
        opp_conceded_avg = 25.0
        opp_scored_avg = 25.0

    return {
        'opponent': opponent,
        'opponent_strength': OPPONENT_STRENGTH.get(opponent, 0.5),
        'is_home': is_home,
        'opponent_pts_conceded_avg': opp_conceded_avg,
        'opponent_pts_scored_avg': opp_scored_avg,
    }


def add_fixture_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add fixture-based features to the dataframe."""
    df = df.copy()

    # Get fixture context for each row
    fixture_data = df.apply(
        lambda row: pd.Series(get_fixture_context(row['club'], row['year'], row['round'])),
        axis=1
    )
    df['opponent'] = fixture_data['opponent']
    df['opponent_strength'] = fixture_data['opponent_strength'].astype(float)
    df['is_home'] = fixture_data['is_home'].astype(int)
    df['opponent_pts_conceded_avg'] = fixture_data['opponent_pts_conceded_avg'].astype(float)
    df['opponent_pts_scored_avg'] = fixture_data['opponent_pts_scored_avg'].astype(float)

    return df


def add_team_form_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add team-level form features derived from match results."""
    df = df.copy()
    team_results = get_team_results()

    def get_team_form(club, year, round_num):
        """Get team form metrics for matches BEFORE this round."""
        results = team_results.get(club, [])
        # Only use results before this game
        prior = [r for r in results if (r['year'], r['round']) < (year, round_num)]

        if not prior:
            return {'team_wins_last_3': 0, 'team_pts_scored_avg': 25.0,
                    'team_pts_conceded_avg': 25.0, 'team_form_momentum': 0.5}

        # Last 3 results with recency weighting
        recent = prior[-3:]
        weights = list(range(1, len(recent) + 1))  # [1, 2, 3] - more recent = higher

        wins_last_3 = sum(1 for r in recent if r['won'])
        pts_scored_avg = np.mean([r['scored'] for r in recent])
        pts_conceded_avg = np.mean([r['conceded'] for r in recent])

        # Weighted momentum: weighted average of win (1.0) / loss (0.0)
        momentum = np.average([1.0 if r['won'] else 0.0 for r in recent], weights=weights)

        return {
            'team_wins_last_3': wins_last_3,
            'team_pts_scored_avg': pts_scored_avg,
            'team_pts_conceded_avg': pts_conceded_avg,
            'team_form_momentum': momentum,
        }

    form_data = df.apply(
        lambda row: pd.Series(get_team_form(row['club'], row['year'], row['round'])),
        axis=1
    )
    for col in form_data.columns:
        df[col] = form_data[col].astype(float)

    return df


def add_matchup_features(df: pd.DataFrame, historical_df: pd.DataFrame = None) -> pd.DataFrame:
    """Add historical matchup features: how did this player/position score against this opponent.

    Args:
        df: DataFrame to add features to
        historical_df: Optional separate DataFrame with historical data for lookups.
                       If None, uses df itself (training mode).
    """
    df = df.copy()

    if 'opponent' not in df.columns:
        df['vs_opponent_avg'] = 0.0
        df['vs_opponent_max'] = 0.0
        df['position_vs_opponent_avg'] = 0.0
        return df

    # Use historical data for lookups if provided, otherwise use df itself
    lookup_df = historical_df if historical_df is not None else df

    # Drop existing matchup columns if they exist (avoid duplicates on re-merge)
    for col in ['vs_opponent_avg', 'vs_opponent_max', 'position_vs_opponent_avg']:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Calculate player vs specific opponent historical averages
    if 'opponent' in lookup_df.columns and 'points' in lookup_df.columns:
        player_vs_opp = lookup_df.groupby(['name', 'opponent'])['points'].agg(['mean', 'max']).reset_index()
        player_vs_opp.columns = ['name', 'opponent', 'vs_opponent_avg', 'vs_opponent_max']

        pos_vs_opp = lookup_df.groupby(['position', 'opponent'])['points'].mean().reset_index()
        pos_vs_opp.columns = ['position', 'opponent', 'position_vs_opponent_avg']
    else:
        player_vs_opp = pd.DataFrame(columns=['name', 'opponent', 'vs_opponent_avg', 'vs_opponent_max'])
        pos_vs_opp = pd.DataFrame(columns=['position', 'opponent', 'position_vs_opponent_avg'])

    # Merge back
    df = df.merge(player_vs_opp, on=['name', 'opponent'], how='left')
    df = df.merge(pos_vs_opp, on=['position', 'opponent'], how='left')

    # Fill missing with overall averages
    overall_avg = lookup_df['points'].mean() if 'points' in lookup_df.columns else 10.0
    overall_max = lookup_df['points'].max() * 0.5 if 'points' in lookup_df.columns else 20.0
    df['vs_opponent_avg'] = df['vs_opponent_avg'].fillna(overall_avg)
    df['vs_opponent_max'] = df['vs_opponent_max'].fillna(overall_max)
    df['position_vs_opponent_avg'] = df['position_vs_opponent_avg'].fillna(overall_avg)

    return df


def add_home_away_history(df: pd.DataFrame, historical_df: pd.DataFrame = None) -> pd.DataFrame:
    """Add player-specific home vs away performance history.

    Args:
        df: DataFrame to add features to
        historical_df: Optional separate DataFrame with historical data for lookups.
    """
    df = df.copy()

    if 'is_home' not in df.columns:
        df['player_home_avg'] = 0.0
        df['player_away_avg'] = 0.0
        df['home_away_diff'] = 0.0
        return df

    lookup_df = historical_df if historical_df is not None else df

    # Drop existing columns to avoid merge conflicts
    for col in ['player_home_avg', 'player_away_avg', 'home_away_diff']:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Calculate per-player home and away averages from lookup data
    if 'is_home' in lookup_df.columns:
        home_avg = lookup_df[lookup_df['is_home'] == 1].groupby('name')['points'].mean().reset_index()
        home_avg.columns = ['name', 'player_home_avg']

        away_avg = lookup_df[lookup_df['is_home'] == 0].groupby('name')['points'].mean().reset_index()
        away_avg.columns = ['name', 'player_away_avg']
    else:
        home_avg = pd.DataFrame(columns=['name', 'player_home_avg'])
        away_avg = pd.DataFrame(columns=['name', 'player_away_avg'])

    df = df.merge(home_avg, on='name', how='left')
    df = df.merge(away_avg, on='name', how='left')

    overall_avg = lookup_df['points'].mean() if 'points' in lookup_df.columns else 10.0
    df['player_home_avg'] = df['player_home_avg'].fillna(overall_avg)
    df['player_away_avg'] = df['player_away_avg'].fillna(overall_avg)
    df['home_away_diff'] = df['player_home_avg'] - df['player_away_avg']

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features for prediction."""
    df = df.copy()

    # Value efficiency metrics
    df['points_per_value'] = df['points'] / df['fantasy_value'].replace(0, 1)

    # Attacking efficiency
    df['meters_per_tackle'] = df['meters_carried'] / df['tackles'].replace(0, 1)

    # Breakdown contribution (for forwards)
    df['breakdown_total'] = df['breakdown_steals'] + df['lineout_steals']

    # Scoring contribution
    df['try_involvement'] = df['tries'] + df['assists']

    # Kicking points (for fly-halves)
    df['kicking_points'] = df['conversions'] * 2 + df['penalties'] * 3 + df['drop_goals'] * 3

    # Discipline (negative indicator)
    df['discipline_issues'] = df['penalties_conceded'] + df['yellow_cards'] * 2 + df['red_cards'] * 5

    # --- Enhanced position features ---
    # Binary position flags
    forward_positions = {'Prop', 'Hooker', 'Second Row', 'Back Row'}
    df['is_forward'] = df['position'].isin(forward_positions).astype(int)

    # Kicker detection (data-driven: did they kick in any game?)
    kicker_names = df[df['conversions'] + df['penalties'] > 0]['name'].unique()
    df['is_kicker'] = df['name'].isin(kicker_names).astype(int)

    # Position average points (how much does this position typically score?)
    pos_avg = df.groupby('position')['points'].mean()
    df['position_avg_points'] = df['position'].map(pos_avg)

    # Position ceiling (90th percentile)
    pos_ceiling = df.groupby('position')['points'].quantile(0.9)
    df['position_ceiling'] = df['position'].map(pos_ceiling)

    # Position × opponent interaction
    if 'opponent' in df.columns:
        pos_opp_avg = df.groupby(['position', 'opponent'])['points'].mean()
        df['position_vs_opponent_strength'] = df.apply(
            lambda row: pos_opp_avg.get((row['position'], row.get('opponent', '')), row.get('position_avg_points', 0)),
            axis=1
        )
    else:
        df['position_vs_opponent_strength'] = df['position_avg_points']

    return df


def calculate_rolling_stats(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """Calculate rolling averages and form trajectory for each player across rounds."""
    df = df.sort_values(['name', 'year', 'round'])

    # Stats to calculate rolling averages for
    rolling_cols = ['points', 'tries', 'tackles', 'meters_carried',
                    'defenders_beaten', 'points_per_value']

    # Group by player and calculate rolling stats
    for col in rolling_cols:
        if col in df.columns:
            df[f'{col}_rolling_avg'] = df.groupby('name')[col].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
            )
            df[f'{col}_rolling_std'] = df.groupby('name')[col].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).std()
            )

    # --- Player form trajectory features ---
    # Form trend: slope of recent points (positive = improving, negative = declining)
    def calc_trend(series):
        shifted = series.shift(1)
        result = shifted.rolling(window=window, min_periods=2).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 2 else 0.0,
            raw=True
        )
        return result

    df['form_trend'] = df.groupby('name')['points'].transform(calc_trend)

    # Ceiling and floor scores (expanding - uses all prior games)
    df['ceiling_score'] = df.groupby('name')['points'].transform(
        lambda x: x.shift(1).expanding(min_periods=1).max()
    )
    df['floor_score'] = df.groupby('name')['points'].transform(
        lambda x: x.shift(1).expanding(min_periods=1).min()
    )

    # Games since peak performance
    def games_since_peak(series):
        result = []
        peak = -np.inf
        games = 0
        for val in series:
            if val >= peak:
                peak = val
                games = 0
            else:
                games += 1
            result.append(games)
        return result

    df['games_since_peak'] = df.groupby('name')['points'].transform(
        lambda x: pd.Series(games_since_peak(x.values), index=x.index)
    )

    # Above average streak
    def above_avg_streak(arr):
        result = []
        running_sum = 0.0
        streak = 0
        for i, val in enumerate(arr):
            running_sum += val
            avg = running_sum / (i + 1)
            if val > avg:
                streak += 1
            else:
                streak = 0
            result.append(streak)
        return result

    df['above_avg_streak'] = df.groupby('name')['points'].transform(
        lambda x: pd.Series(above_avg_streak(x.values), index=x.index)
    )

    # Fill NaN rolling/trajectory stats with overall mean
    trajectory_cols = [col for col in df.columns
                       if any(s in col for s in ['rolling', 'form_trend', 'ceiling_score',
                                                  'floor_score', 'games_since_peak', 'above_avg_streak'])]
    for col in trajectory_cols:
        col_mean = df[col].mean()
        df[col] = df[col].fillna(col_mean if not np.isnan(col_mean) else 0.0)

    return df


def merge_betting_odds(df: pd.DataFrame) -> pd.DataFrame:
    """Merge betting odds data into the dataframe if available.

    Looks for betting_odds_{year}.json in the data output directory.
    Adds: team_win_probability, opponent_win_probability, match_total_implied, handicap_line
    """
    df = df.copy()
    odds_cols = ['team_win_probability', 'opponent_win_probability',
                 'match_total_implied', 'handicap_line']

    # Try to load odds for each year in the data
    years = df['year'].unique() if 'year' in df.columns else []
    all_odds = []

    for year in years:
        odds_path = DATA_OUTPUT / f"betting_odds_{int(year)}.json"
        if odds_path.exists():
            try:
                with open(odds_path, 'r') as f:
                    odds_data = json.load(f)
                for round_odds in odds_data.get('rounds', []):
                    round_num = round_odds['round']
                    for match in round_odds.get('matches', []):
                        home = match['home']
                        away = match['away']
                        home_prob = match.get('home_win_prob', 0.5)
                        away_prob = match.get('away_win_prob', 0.5)
                        total = match.get('total_points', 45.0)
                        handicap = match.get('handicap', 0.0)

                        # Home team row
                        all_odds.append({
                            'year': year, 'round': round_num, 'club': home,
                            'team_win_probability': home_prob,
                            'opponent_win_probability': away_prob,
                            'match_total_implied': total,
                            'handicap_line': handicap,
                        })
                        # Away team row
                        all_odds.append({
                            'year': year, 'round': round_num, 'club': away,
                            'team_win_probability': away_prob,
                            'opponent_win_probability': home_prob,
                            'match_total_implied': total,
                            'handicap_line': -handicap,
                        })
                print(f"Loaded betting odds for {int(year)}")
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not parse betting odds for {int(year)}: {e}")

    if all_odds:
        odds_df = pd.DataFrame(all_odds)
        df = df.merge(odds_df, on=['year', 'round', 'club'], how='left')

    # Fill missing odds with neutral values
    defaults = {'team_win_probability': 0.5, 'opponent_win_probability': 0.5,
                'match_total_implied': 45.0, 'handicap_line': 0.0}
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default
        else:
            df[col] = df[col].fillna(default)

    return df


def prepare_training_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list, list]:
    """Prepare features and target for model training.

    Applies all feature engineering steps:
    1. Filter to played games only
    2. Fixture context (opponent strength, home/away, opponent stats)
    3. Team form features (wins, momentum, recent results)
    4. Matchup features (player/position vs specific opponent)
    5. Home/away player history
    6. Engineered features (efficiency, position, kicker detection)
    7. Rolling stats & form trajectory
    8. Betting odds (if available)
    """
    # Filter to only games where players actually played
    df = filter_played_games(df)
    print(f"After filtering non-played games: {len(df)} records")

    # Add fixture context (opponent strength, home/away, opponent stats)
    df = add_fixture_features(df)

    # Add team form features
    df = add_team_form_features(df)

    # Add historical matchup features
    df = add_matchup_features(df)

    # Add home/away player history
    df = add_home_away_history(df)

    # Engineered features (including enhanced position features)
    df = engineer_features(df)

    # Rolling stats and form trajectory
    df = calculate_rolling_stats(df)

    # Load and merge betting odds if available
    df = merge_betting_odds(df)

    # Encode categorical variables
    position_encoder = LabelEncoder()
    club_encoder = LabelEncoder()

    df['position_encoded'] = position_encoder.fit_transform(df['position'])
    df['club_encoded'] = club_encoder.fit_transform(df['club'])

    # Feature columns (exclude target and identifiers)
    exclude_cols = ['name', 'club', 'position', 'points', 'year', 'round',
                    'matches_played', 'man_of_match', 'opponent', 'played']

    feature_cols = [col for col in df.columns
                    if col not in exclude_cols
                    and df[col].dtype in ['int64', 'float64', 'int32', 'float32', 'bool']]

    # Target variable
    target = df['points']
    features = df[feature_cols]

    # Handle any remaining NaN values
    features = features.fillna(0)

    print(f"Total features: {len(feature_cols)}")

    return features, target, feature_cols, df


def train_model(features: pd.DataFrame, target: pd.Series) -> tuple:
    """Train XGBoost model with cross-validation."""
    if xgb is None:
        raise ImportError("XGBoost not installed")

    # XGBoost parameters optimized for small datasets
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 4,  # Shallow trees to avoid overfitting
        'learning_rate': 0.1,
        'n_estimators': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 5,
        'random_state': 42,
    }

    model = xgb.XGBRegressor(**params)

    # Time series cross-validation (respects temporal ordering)
    tscv = TimeSeriesSplit(n_splits=3)
    cv_scores = cross_val_score(model, features, target,
                                 cv=tscv, scoring='neg_mean_absolute_error')

    print(f"\nCross-validation MAE: {-cv_scores.mean():.2f} (+/- {cv_scores.std():.2f})")

    # Train on full data
    model.fit(features, target)

    # Feature importance
    importance = pd.DataFrame({
        'feature': features.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    return model, importance


def analyze_trends(df: pd.DataFrame) -> dict:
    """Extract key trends from historical data."""
    trends = {}

    # 1. Position scoring patterns
    position_stats = df.groupby('position').agg({
        'points': ['mean', 'std', 'max'],
        'fantasy_value': 'mean',
        'points_per_value': 'mean' if 'points_per_value' in df.columns else 'count'
    }).round(2)
    trends['position_scoring'] = position_stats

    # 2. Top performers by value efficiency
    player_avg = df.groupby('name').agg({
        'points': 'mean',
        'fantasy_value': 'mean',
        'club': 'first',
        'position': 'first'
    })
    player_avg['ppv'] = player_avg['points'] / player_avg['fantasy_value']
    player_avg = player_avg.sort_values('ppv', ascending=False)
    trends['value_efficiency'] = player_avg.head(20)

    # 3. Consistency leaders (low std dev relative to mean)
    player_consistency = df.groupby('name').agg({
        'points': ['mean', 'std', 'count'],
        'club': 'first',
        'position': 'first'
    })
    player_consistency.columns = ['avg_points', 'std_points', 'games', 'club', 'position']
    player_consistency = player_consistency[player_consistency['games'] >= 3]
    player_consistency['consistency'] = player_consistency['avg_points'] / player_consistency['std_points'].replace(0, 1)
    player_consistency = player_consistency.sort_values('consistency', ascending=False)
    trends['consistency'] = player_consistency.head(20)

    # 4. Country performance
    country_stats = df.groupby('club').agg({
        'points': ['mean', 'sum'],
        'tries': 'sum',
        'tackles': 'mean'
    }).round(2)
    trends['country_performance'] = country_stats

    # 5. Round-over-round improvement
    round_stats = df.groupby(['year', 'round']).agg({
        'points': ['mean', 'max'],
        'tries': 'sum'
    }).round(2)
    trends['round_trends'] = round_stats

    return trends


def predict_round(model, df: pd.DataFrame, target_year: int, target_round: int,
                  feature_cols: list, full_df: pd.DataFrame,
                  min_games: int = 4) -> pd.DataFrame:
    """Generate predictions for a future round.

    Applies the same feature engineering pipeline used in training so that
    the prediction features are consistent with what the model learned.

    Args:
        min_games: Minimum games required for a prediction to be considered reliable
    """
    # Filter to only games where players actually played
    played_df = filter_played_games(full_df)

    # Count games per player for confidence scoring (only games they played)
    player_games = played_df.groupby('name').agg({
        'round': 'count',
        'points': ['mean', 'std']
    }).reset_index()
    player_games.columns = ['name', 'games_played', 'historical_avg', 'historical_std']

    # Only predict for players who are in the target year's squad
    current_year_players = full_df[full_df['year'] == target_year]['name'].unique()

    # Get the latest game each player ACTUALLY PLAYED (not just appeared in squad)
    latest_data = played_df.sort_values(['year', 'round']).groupby('name').last().reset_index()

    # Filter to only players in the current year's squad
    latest_data = latest_data[latest_data['name'].isin(current_year_players)]

    # Set fixture context for the TARGET round (not the player's last game)
    latest_data['year'] = target_year
    latest_data['round'] = target_round

    # Apply the same feature pipeline as training
    # Enrich historical data with fixture features for lookups
    historical_with_fixtures = add_fixture_features(played_df)

    latest_data = add_fixture_features(latest_data)
    latest_data = add_team_form_features(latest_data)
    latest_data = add_matchup_features(latest_data, historical_df=historical_with_fixtures)
    latest_data = add_home_away_history(latest_data, historical_df=historical_with_fixtures)
    latest_data = engineer_features(latest_data)
    latest_data = merge_betting_odds(latest_data)

    # Ensure all feature columns exist (some rolling stats may be missing)
    for col in feature_cols:
        if col not in latest_data.columns:
            latest_data[col] = 0

    # Use rolling averages from historical data
    features = latest_data[feature_cols].fillna(0)

    # Generate predictions
    predictions = model.predict(features)

    # Combine with player info
    results = pd.DataFrame({
        'name': latest_data['name'],
        'club': latest_data['club'],
        'position': latest_data['position'],
        'fantasy_value': latest_data['fantasy_value'],
        'predicted_points': predictions,
        'predicted_ppv': predictions / latest_data['fantasy_value'].replace(0, 1)
    })

    # Merge with games played for confidence
    results = results.merge(player_games, on='name', how='left')

    # Add confidence tier
    results['confidence'] = results['games_played'].apply(
        lambda x: 'HIGH' if x >= 5 else ('MEDIUM' if x >= 3 else 'LOW')
    )

    return results.sort_values('predicted_points', ascending=False)


def run_prediction_analysis(output_html: bool = True) -> dict:
    """Main entry point for prediction analysis."""
    print("=" * 60)
    print("6N Fantasy Rugby Prediction Analysis")
    print("=" * 60)

    # Load all data
    df = load_all_data([2025, 2026])

    # Analyze trends
    print("\n--- Extracting Trends from 2025 Data ---")
    df_with_features = engineer_features(df)
    trends = analyze_trends(df_with_features)

    print("\nTop 10 by Value Efficiency (Points per Fantasy Value):")
    print(trends['value_efficiency'][['points', 'fantasy_value', 'ppv', 'club', 'position']].head(10).to_string())

    print("\nMost Consistent Performers (3+ games):")
    print(trends['consistency'][['avg_points', 'std_points', 'consistency', 'club', 'position']].head(10).to_string())

    print("\nScoring by Position:")
    print(trends['position_scoring'].to_string())

    print("\nCountry Performance:")
    print(trends['country_performance'].to_string())

    # Train prediction model
    print("\n--- Training XGBoost Model ---")
    features, target, feature_cols, prepared_df = prepare_training_data(df)
    model, importance = train_model(features, target)

    print("\nTop 15 Most Important Features:")
    print(importance.head(15).to_string(index=False))

    # Auto-detect the next unplayed round
    target_round = detect_target_round(2026)
    print(f"\n--- Predictions for 2026 Round {target_round} (auto-detected) ---")
    predictions = predict_round(model, df, 2026, target_round, feature_cols, prepared_df)

    # Data quality summary
    print("\n=== DATA QUALITY SUMMARY ===")
    print(f"Total players: {len(predictions)}")
    print(f"HIGH confidence (5+ games): {len(predictions[predictions['confidence'] == 'HIGH'])}")
    print(f"MEDIUM confidence (3-4 games): {len(predictions[predictions['confidence'] == 'MEDIUM'])}")
    print(f"LOW confidence (1-2 games): {len(predictions[predictions['confidence'] == 'LOW'])}")

    # Only show high confidence predictions
    high_conf = predictions[predictions['confidence'] == 'HIGH']

    print("\n=== TOP 20 PREDICTED PERFORMERS (HIGH CONFIDENCE ONLY) ===")
    display_cols = ['name', 'club', 'position', 'fantasy_value', 'predicted_points',
                    'games_played', 'historical_avg', 'confidence']
    print(high_conf[display_cols].head(20).to_string(index=False))

    print("\n=== TOP 15 VALUE PICKS (HIGH CONFIDENCE ONLY) ===")
    value_picks = high_conf.sort_values('predicted_ppv', ascending=False)
    print(value_picks[display_cols].head(15).to_string(index=False))

    print("\n=== BEST BY POSITION (HIGH CONFIDENCE) ===")
    for pos in high_conf['position'].unique():
        pos_top = high_conf[high_conf['position'] == pos].head(3)
        print(f"\n{pos}:")
        print(pos_top[['name', 'club', 'predicted_points', 'fantasy_value', 'historical_avg']].to_string(index=False))

    # Save results
    output_path = DATA_OUTPUT / "predictions_2026.json"
    predictions.to_json(output_path, orient='records', indent=2)
    print(f"\nPredictions saved to: {output_path}")

    if output_html:
        generate_prediction_report(predictions, trends, importance)

    return {
        'predictions': predictions,
        'trends': trends,
        'feature_importance': importance,
        'model': model
    }


def generate_prediction_report(predictions: pd.DataFrame, trends: dict,
                                importance: pd.DataFrame) -> None:
    """Generate HTML report with predictions and analysis."""
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("Plotly required for HTML report")
        return

    # Create figure with subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Top 20 Predicted Performers',
            'Feature Importance',
            'Predicted Points by Position',
            'Value Picks (Points per Value)'
        ),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "box"}, {"type": "bar"}]]
    )

    # 1. Top performers bar chart
    top20 = predictions.head(20)
    fig.add_trace(
        go.Bar(x=top20['name'], y=top20['predicted_points'],
               marker_color='steelblue', name='Predicted Points'),
        row=1, col=1
    )

    # 2. Feature importance
    top_features = importance.head(15)
    fig.add_trace(
        go.Bar(x=top_features['importance'], y=top_features['feature'],
               orientation='h', marker_color='coral', name='Importance'),
        row=1, col=2
    )

    # 3. Box plot by position
    for position in predictions['position'].unique():
        pos_data = predictions[predictions['position'] == position]
        fig.add_trace(
            go.Box(y=pos_data['predicted_points'], name=position),
            row=2, col=1
        )

    # 4. Value picks
    value_top = predictions.sort_values('predicted_ppv', ascending=False).head(15)
    fig.add_trace(
        go.Bar(x=value_top['name'], y=value_top['predicted_ppv'],
               marker_color='green', name='Points per Value'),
        row=2, col=2
    )

    fig.update_layout(
        height=900,
        title_text="6N Fantasy Rugby - 2026 Predictions",
        showlegend=False
    )

    output_path = PROJECT_ROOT / "prediction_report.html"
    fig.write_html(str(output_path))
    print(f"\nHTML report saved to: {output_path}")


if __name__ == "__main__":
    results = run_prediction_analysis()
