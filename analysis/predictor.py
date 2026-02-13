"""
Fantasy Rugby Prediction Module

Uses XGBoost to predict player fantasy points based on historical performance.
Analyzes 2025 data to predict 2026 outcomes.
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


# Opponent strength ratings based on 2025 results (lower = easier to score against)
OPPONENT_STRENGTH = {
    'Wales': 0.0,     # Easiest - conceded most points
    'Italy': 0.2,     # Second easiest
    'Scotland': 0.5,  # Mid-tier
    'England': 0.6,   # Mid-tier
    'France': 0.8,    # Strong
    'Ireland': 1.0,   # Hardest - conceded fewest points
}


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
        return {'opponent': None, 'opponent_strength': 0.5, 'is_home': False}

    return {
        'opponent': opponent,
        'opponent_strength': OPPONENT_STRENGTH.get(opponent, 0.5),
        'is_home': is_home,
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
    df['opponent_strength'] = fixture_data['opponent_strength']
    df['is_home'] = fixture_data['is_home'].astype(int)

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

    return df


def calculate_rolling_stats(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """Calculate rolling averages for each player across rounds."""
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

    # Fill NaN rolling stats with overall mean for that column
    for col in df.columns:
        if 'rolling' in col:
            df[col] = df[col].fillna(df[col].mean())

    return df


def prepare_training_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list, list]:
    """Prepare features and target for model training."""
    # Filter to only games where players actually played
    df = filter_played_games(df)
    print(f"After filtering non-played games: {len(df)} records")

    # Add fixture context (opponent strength, home/away)
    df = add_fixture_features(df)

    df = engineer_features(df)
    df = calculate_rolling_stats(df)

    # Encode categorical variables
    position_encoder = LabelEncoder()
    club_encoder = LabelEncoder()

    df['position_encoded'] = position_encoder.fit_transform(df['position'])
    df['club_encoded'] = club_encoder.fit_transform(df['club'])

    # Feature columns (exclude target and identifiers)
    exclude_cols = ['name', 'club', 'position', 'points', 'year', 'round',
                    'matches_played', 'man_of_match', 'opponent']

    feature_cols = [col for col in df.columns
                    if col not in exclude_cols
                    and df[col].dtype in ['int64', 'float64']]

    # Target variable
    target = df['points']
    features = df[feature_cols]

    # Handle any remaining NaN values
    features = features.fillna(0)

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

    # Get the latest game each player ACTUALLY PLAYED (not just appeared in squad)
    latest_data = played_df.sort_values(['year', 'round']).groupby('name').last().reset_index()

    # Set fixture context for the TARGET round (not the player's last game)
    latest_data['year'] = target_year
    latest_data['round'] = target_round
    latest_data = add_fixture_features(latest_data)

    # Prepare features for prediction
    latest_data = engineer_features(latest_data)

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

    # Generate predictions for 2026 Round 2
    print("\n--- Predictions for 2026 Round 2 ---")
    predictions = predict_round(model, df, 2026, 2, feature_cols, prepared_df)

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
        title_text="6N Fantasy Rugby - 2026 Round 2 Predictions",
        showlegend=False
    )

    output_path = PROJECT_ROOT / "prediction_report.html"
    fig.write_html(str(output_path))
    print(f"\nHTML report saved to: {output_path}")


if __name__ == "__main__":
    results = run_prediction_analysis()
