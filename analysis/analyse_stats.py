import json
from pathlib import Path
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from itertools import cycle
import plotly.express as px
import math
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, LpStatus, PULP_CBC_CMD

from config.settings import get_fixtures, get_scoring_rules
from analysis.predictor import run_prediction_analysis
from src.processors.extract_summary import generate_fixtures_page


def run_analysis(round=None, match=None, max_players_per_position=15, year=None):
    """
    Run all analysis visualizations for all available data
    """
    print("\nRunning analysis...")

    # Load data
    data_dir = Path('data/output')
    df = load_json_data(data_dir, year=year)

    if df.empty:
        print("No data available for analysis")
        return

    fixtures = get_fixtures(year)

    # clean the data
    df_all_rounds = df_clean(df)

    # round filtering
    if round != None:
        df_cleaned = df_all_rounds[df_all_rounds['round'] == round]
    else:
        df_cleaned = df_all_rounds

    # match filtering (requires round)
    if match is not None and round is not None and round in fixtures:
        fixture = fixtures[round][match - 1]
        match_clubs = [fixture[0], fixture[1]]
        df_cleaned = df_cleaned[df_cleaned['club'].isin(match_clubs)]
        print(f"Filtered to match {match}: {fixture[0]} v {fixture[1]}")

    # Build filter subtitle for plot titles
    filter_parts = []
    if year:
        filter_parts.append(str(year))
    if round is not None:
        filter_parts.append(f"Round {round}")
    if match is not None and round is not None and round in fixtures:
        filter_parts.append(f"{fixture[0]} v {fixture[1]}")
    subtitle = ' | '.join(filter_parts) if filter_parts else 'All Data'

    print("Unique round values in dataframe:", df_cleaned['round'].unique())
    print("Unique countries values in dataframe:", df_cleaned['club'].unique())
    print("Unique position values in dataframe:", df_cleaned['position'].unique())
    
    # save df to csv
    output_file = data_dir / '6N_combined_stats.csv'
    df_cleaned.to_csv(output_file)
    print(f"Saved combined stats to: {output_file}")
    
    sub_list = [
   "E. Lloyd", 
   "G. Thomas", 
   "H. Thomas", 
   "T. Williams", 
   "A. Wainwright", 
   "R. Williams", 
   "J. Evans", 
   "J. Roberts",
    "E. Ashman", 
   "R. Sutherland", 
   "W. Hurd", 
   "S. Skinner", 
   "G. Brown", 
   "M. Fagerson", 
   "J. Dobie", 
   "S. McDowall",
   "J. George", 
   "F. Baxter", 
   "J. Heyes", 
   "G. Martin", 
   "C. Cunningham-South", 
   "B. Curry", 
   "H. Randall", 
   "E. Daly"   
    ]   
    
    try:
        # Generate fixtures page
        generate_fixtures_page(year=year)

        # Run visualizations
        #ppm_vs_player_bar_chart(df_cleaned)

        df_supersub, filter_text = filter_by_supersubs(df_cleaned)
        ppm_vs_player_bar_chart(df_supersub, filter_text, subtitle=subtitle)

        ppm_per_position_bar_chart(df_supersub, filter_text, subtitle=subtitle)
        
        #df_filt = filter_out_tries(df_cleaned)
        #df_filt = filter_by_round(df_cleaned, [1,2,3])
        #df_filt = filter_by_team(df_cleaned, ['Wales','England','Scotland'])
        #df_filt = filter_by_value(df_filt, 14.7)
        #df_filt = filter_by_players(df_filt, sub_list)
        
        df_filt = df_cleaned
        plot_player_points_breakdown(df_filt, min_points=10, max_players_per_position=max_players_per_position, is_ppm=True, subtitle=subtitle, year=year)
        plot_player_points_breakdown(df_filt, min_points=10, max_players_per_position=max_players_per_position, is_ppm=False, subtitle=subtitle, year=year)

        plot_points_distribution_by_category(df_cleaned, subtitle=subtitle, year=year)
        plot_match_fantasy_comparison(df_cleaned, fixtures=fixtures, subtitle=subtitle)
        plot_player_round_heatmap(df_cleaned, max_players=max_players_per_position, subtitle=subtitle)
        plot_scatter_matrix(df_cleaned, subtitle=subtitle)
        plot_value_profile(df_cleaned, max_players_per_position=max_players_per_position, subtitle=subtitle)
        optimize_squad(df_cleaned, year=year, subtitle=subtitle)

        # Generate prediction report
        try:
            run_prediction_analysis(output_html=True)
        except Exception as pred_e:
            print(f"Warning: Prediction report generation failed: {pred_e}")

        # calculate points for a selected team based on previous rounds
        player_list1 = [
        "L. Bielle-Biarrey",
        "J. Lowe",
        "D. Graham",
        "H. Jones",
        "T. Menoncello",
        "S. Prendergast",
        "J. Gibson-Park",
        "G. Alldritt",
        "L. Cannone",
        "C. Doris",
        "D. Jenkins",
        "W. Rowlands",
        "D. Fischetti", 
        "F. Baxter",
        "J. Marchand"
        ]

        selected_players = df_cleaned[df_cleaned['name'].isin(player_list1)]
        #plot_player_points_breakdown(selected_players, min_points=10, max_players_per_position=20)
        
    except Exception as e:
        print(f"Error during visualization: {str(e)}")
        
def load_json_data(data_dir, year=None):
    """
    Load all JSON files and flatten into a single DataFrame.
    If year is specified, only load files matching that year.
    """
    if year:
        files = list(data_dir.glob(f'six_nations_stats_{year}_*.json'))
    else:
        files = list(data_dir.glob('six_nations_stats_*.json'))
    print(f"Loading json files: {files}")
    
    # Load all JSON files and compile player data
    data_list = [
        {
            'extraction_date': json_data['extraction_date'],
            'round': json_data['round'],
            'name': player['name'],
            'club': player['club'],
            'position': player['position'],
            'fantasy_value': player.get('fantasy_value', 0),
            'fantasy_position_group': player.get('fantasy_position_group', 'Unknown'),
            **player['stats']  # Unpack all stats fields
        }
        for f in files
        for json_data in [json.load(f.open())]
        for player in json_data['players']
    ]
    
    return pd.DataFrame(data_list)

def df_clean(df):
    # Clean and convert points and matches_played to numeric
    df.rename(columns={'matches_played': 'minutes'}, inplace=True)

    # Replace all empty strings with 0 across the entire DataFrame    
    
    # 1. First handle empty values
    pd.set_option('future.no_silent_downcasting', True)
    df = df.fillna(0).replace('', 0)

    # 2. Simpler conversion approach - try to convert each column
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            pass  # Keep as is if can't convert

    # Calculate points per minute
    df['points_per_min'] = df.apply(
        lambda row: int((row['points'] / row['minutes']) * 100) if row['minutes'] > 0 else 0,
        axis=1
    )

    # Filter out players with zero points
    df_cleaned = df[df['points'] != 0]
    return df_cleaned
    
def calculate_fantasy_points(df, year=None):
    """
    Calculate fantasy points breakdown by category using year-specific scoring rules.

    Args:
        df: DataFrame with player stats
        year: Scoring year (uses data year if None)
    """
    # Get scoring rules for the year
    rules = get_scoring_rules(year)

    # Define position groups
    backs = ['Back Three', 'Centre', 'Fly-Half', 'Scrum-Half']
    forwards = ['Prop', 'Hooker', 'Second Row', 'Back Row']

    # Initialize try points based on position
    tries_back = df.apply(
        lambda row: row['tries'] * rules['tries_back'] if row['position'] in backs else 0, axis=1
    )
    tries_forward = df.apply(
        lambda row: row['tries'] * rules['tries_forward'] if row['position'] in forwards else 0, axis=1
    )

    fantasy_points = {
        # Attacking
        'tries_back': tries_back,
        'tries_forward': tries_forward,
        'assists': df['assists'] * rules['assists'],
        'conversions': df['conversions'] * rules['conversions'],
        'penalties': df['penalties'] * rules['penalties'],
        'drop_goals': df['drop_goals'] * rules['drop_goals'],
        'defenders_beaten': df['defenders_beaten'] * rules['defenders_beaten'],
        'meters': (df['meters_carried'] * rules['meters_carried']).apply(np.floor),  # Floor to nearest 10m
        'kick_50_22': df['kick_50_22'] * rules['kick_50_22'],
        'offloads': df['offloads'] * rules['offloads'],
        'scrum_wins': df['scrum_wins'] * rules['scrum_wins'],
        # Defensive
        'tackles': df['tackles'] * rules['tackles'],
        'breakdown_steals': df['breakdown_steals'] * rules['breakdown_steals'],
        'lineout_steals': df['lineout_steals'] * rules['lineout_steals'],
        'penalties_conceded': df['penalties_conceded'] * rules['penalties_conceded'],
        # General
        'man_of_match': df['man_of_match'] * rules['man_of_match'],
        'yellow_cards': df['yellow_cards'] * rules['yellow_cards'],
        'red_cards': df['red_cards'] * rules['red_cards'],
    }

    # Add kicks_retained if present in rules (new in 2026)
    if 'kicks_retained' in rules and 'kicks_retained' in df.columns:
        fantasy_points['kicks_retained'] = df['kicks_retained'] * rules['kicks_retained']

    return fantasy_points

# ==================================================================================================================

def filter_by_supersubs(df):
    filter_text = "supersubs"
    df_filter = df[df['fantasy_position_group'] == 'Subs']
    return df_filter, filter_text

def filter_by_team(df, country):
    df_filter = df[df['club'].isin(country)]
    return df_filter

def filter_by_players(df, players):
    df_filter = df[df['name'].isin(players)]
    return df_filter

def filter_by_value(df, value):
    df_filter = df[df['fantasy_value'] <= value]
    return df_filter

def filter_out_tries(df):
    df_filter = df[df['tries'] == 0]
    return df_filter

def ppm_vs_player_bar_chart(df, filter_text='all_players', subtitle=''):
    # Create a horizontal bar chart comparing points per minute and minutes played for substitutes.
    # Points per minute on one set of bars on primary axis
    # Minutes played on another set of bars on secondary axis
    # The bars should not be stacked on top of each other, but should be offset
    # The x axis 0 point should line up with the seconday axis 0 point
    
    plot_title = f'Points per Minute and Minutes Played. Filterered by {filter_text}<br><sup>{subtitle}</sup>' if subtitle else f'Points per Minute and Minutes Played. Filterered by {filter_text}'
    
    # Sort by points per minute
    df_sorted = df.sort_values('points_per_min', ascending=True)

    # Determine max values for axis scaling
    max_ppm = df_sorted['points_per_min'].max()
    max_minutes = df_sorted['minutes'].max()

    # Create figure
    fig = go.Figure()

    # Define bar width for alignment
    bar_width = 0.4

    # Add points per minute trace (Teal)
    fig.add_trace(
        go.Bar(
            y=df_sorted['name'],
            x=df_sorted['points_per_min'],
            orientation='h',
            name='Points per Minute',
            width=bar_width,
            offset=-bar_width/2,  # Shift left
            marker_color='#1f77b4',  # TEAL
            xaxis='x1'
        )
    )
    
    # Add minutes played trace (Orange)
    fig.add_trace(
        go.Bar(
            y=df_sorted['name'],
            x=df_sorted['minutes'],
            orientation='h',
            name='Minutes Played',
            width=bar_width,
            offset=bar_width/2,  # Shift right
            marker_color='#ff7f0e',  # ORANGE
            xaxis='x2'
        )
    )

    # Update layout with separate x-axes
    fig.update_layout(
        title=plot_title, # 'Super Sub Analysis - Points per Minute and Minutes Played',
        margin=dict(l=200),

        xaxis=dict(
            title='Points per Minute',
            side='top',
            range=[0, max_ppm * 1.1],
            overlaying='x2',
            showgrid=False,
            zeroline=False
        ),
        xaxis2=dict(
            title='Minutes Played',
            side='bottom',
            range=[0, max_minutes * 1.1],
            showgrid=False,
            zeroline=False
        ),
        
        yaxis=dict(
            title='Player Name',
            autorange='reversed'
        ),
        
        barmode='group',  
        bargap=0.2,
        bargroupgap=0.1
    )
    
    # Save visualization
    html_name = f'ppm_vs_player_bar_chart_{filter_text}.html'
    fig.write_html(html_name)

    print(f"Written to {html_name}")
    return fig  # Optional: return the figure for further manipulation

def ppm_per_position_bar_chart(df, filter_text='all_players', subtitle=''):
    """
    Create a horizontal bar chart showing points per minute for substitutes, grouped by position.
    Positions are ordered by their average points.
    """
    plot_title = f'Points per Minute and Minutes Played. Grouped by position. Filtered by {filter_text}<br><sup>{subtitle}</sup>' if subtitle else f'Points per Minute and Minutes Played. Grouped by position. Filtered by {filter_text}'
    
    # Calculate average points per position
    position_averages = df.groupby('position')['points_per_min'].mean().sort_values(ascending=False)
    
    # Sort by points per minute within each position
    # Use position_averages index to maintain position order
    df_sorted = df.sort_values(['position', 'points_per_min'], ascending=[True, False])
    
    # Create figure
    fig = go.Figure()
    
    # Create a bar trace for each position, using ordered positions
    for position in position_averages.index:
        position_data = df_sorted[df_sorted['position'] == position]
        
        fig.add_trace(
            go.Bar(
                y=position_data['name'],
                x=position_data['points_per_min'],
                orientation='h',
                name=f"{position} (avg: {position_averages[position]:.1f})"
            )
        )
    
    # Update layout
    fig.update_layout(
        title=plot_title,
        xaxis=dict(
            title='Points per Minute (x100)',
            showgrid=True,
            zeroline=True
        ),
        yaxis=dict(
            title='Player Name',
            autorange='reversed'
        ),
        margin=dict(l=200),
        barmode='group',
        showlegend=True,
        legend_title='Position'
    )
    
    # Save visualization
    html_name = f'ppm_per_position_bar_chart_{filter_text}.html'
    fig.write_html(html_name)
    print(f"Written to {html_name}")
    return fig


def plot_player_points_breakdown(df, min_points=10, max_players_per_position=20, filter_text="None", is_ppm=False, subtitle='', year=None):
    """
    Creates a horizontal bar chart showing the breakdown of fantasy points or points per minute for rugby players,
    sorted by their total points or overall PPM (total points / total minutes) within each position.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing player stats and points. Must include columns:
        
    min_points : int, optional (default=10)
        Minimum total points threshold for including a player in the visualization
        
    max_players_per_position : int, optional (default=20)
        Maximum number of players to show per position
        
    filter_text : str, optional (default="None")
        Text to append to output filename for identification
        
    is_ppm : bool, optional (default=False)
        If True, displays points per minute instead of total points

    Notes:
    ------
    - Players are filtered based on their total points across all rounds
    - PPM is calculated as (total points across all rounds) / (total minutes across all rounds)
    - The bars show the breakdown for the current round
    """
    fantasy_value_round = 3
    # Calculate total points and overall PPM for each player across all rounds
    player_totals = df.groupby('name').agg({
        'points': 'sum',
        'minutes': 'sum'
    }).reset_index()
    
    # Calculate overall PPM using total points / total minutes
    player_totals['overall_ppm'] = (player_totals['points'] / player_totals['minutes']).round(1)
    player_totals = player_totals.rename(columns={'points': 'total_player_points'})
    
    # Aggregate points for unique players for current round
    df_aggregated = df.groupby(['name', 'club', 'position','tries','assists','conversions','penalties',
                            'drop_goals','defenders_beaten','meters_carried','kick_50_22','tackles',
                            'breakdown_steals','penalties_conceded','man_of_match','lineout_steals',
                            'yellow_cards','red_cards','offloads','scrum_wins']).agg({
        'points': 'sum',
        'minutes': 'sum',
        'fantasy_value': 'max',
        'round': 'max'  # Include round to ensure we use the most recent round's value
    }).reset_index()
    
    # Calculate current round PPM
    df_aggregated['points_per_min'] = (df_aggregated['points'] / df_aggregated['minutes']).round(1)
    
    # Add total stats to the aggregated dataframe
    df_aggregated = df_aggregated.merge(player_totals, on='name', how='left')

    fantasy_points = calculate_fantasy_points(df_aggregated, year=year)

    # Filter based on total points
    df_filtered = df_aggregated[df_aggregated['total_player_points'] > min_points]
   
    # Sort by overall PPM or total points depending on mode
    sort_column = 'overall_ppm' if is_ppm else 'total_player_points'

    # Get top N unique players per position (not rows)
    top_players = (df_filtered
        .sort_values(['position', sort_column], ascending=[True, False])
        .drop_duplicates(subset='name')
        .groupby('position')
        .head(max_players_per_position)
    )['name']

    # Keep all rows for those players, sorted for display
    df_sorted = (df_filtered[df_filtered['name'].isin(top_players)]
        .sort_values(['position', sort_column, 'round'],
                    ascending=[True, False, True])
    )

    #print(df_sorted.head(20))
    # Create a mapping of players to their most recent fantasy value
    fantasy_value_map = df_sorted.sort_values('round', ascending=False).groupby('name')['fantasy_value'].first()

    # Player labels - include relevant total stat and fantasy value
    stat_value = df_sorted[sort_column].round(1 if is_ppm else 0).astype(str)
    has_fantasy_values = fantasy_value_map.sum() > 0
    if has_fantasy_values:
        all_labels = df_sorted.apply(
            lambda row: f"{row['name']} ({row['position']}) [{stat_value.loc[row.name]}, ${fantasy_value_map.loc[row['name']]}]",
            axis=1
        )
    else:
        all_labels = df_sorted.apply(
            lambda row: f"{row['name']} ({row['position']}) [{stat_value.loc[row.name]}]",
            axis=1
        )

    round_points = df_sorted.points.sum()
    fantasy_values_rnd = math.ceil(df[df['round'] == fantasy_value_round]['fantasy_value'].sum())

    # --- Helper to compute per-category bar data for a subset ---
    def build_breakdown_data(df_sub, fp_sub):
        """Return (labels_list, category_dict) for a filtered subset."""
        sub_sorted = df_sub.copy()
        fv_map = sub_sorted.sort_values('round', ascending=False).groupby('name')['fantasy_value'].first()
        sv = sub_sorted[sort_column].round(1 if is_ppm else 0).astype(str)
        has_fv = fv_map.sum() > 0
        if has_fv:
            lbls = sub_sorted.apply(
                lambda r: f"{r['name']} ({r['position']}) [{sv.loc[r.name]}, ${fv_map.loc[r['name']]}]", axis=1)
        else:
            lbls = sub_sorted.apply(
                lambda r: f"{r['name']} ({r['position']}) [{sv.loc[r.name]}]", axis=1)

        cat_data = {}
        for category, pts in fp_sub.items():
            ps = pd.Series(pts).reindex(sub_sorted.index).fillna(0)
            if is_ppm:
                ps = ps * sub_sorted['points_per_min'] / sub_sorted['points']
            cat_data[category] = ps
        return list(lbls), cat_data

    # Pre-compute for "All"
    all_categories = list(fantasy_points.keys())
    labels_all_bd, cat_data_all = build_breakdown_data(df_sorted, fantasy_points)

    # Pre-compute per country
    countries = sorted(df_sorted['club'].unique())
    country_bd = {}
    for country in countries:
        mask_c = df_sorted['club'] == country
        df_c = df_sorted[mask_c]
        fp_c = {cat: pd.Series(pts).reindex(df_c.index).fillna(0) for cat, pts in fantasy_points.items()}
        lbls_c, cd_c = build_breakdown_data(df_c, fp_c)
        country_bd[country] = (lbls_c, cd_c)

    # Pre-compute per position
    positions = sorted(df_sorted['position'].unique())
    position_bd = {}
    for pos in positions:
        mask_p = df_sorted['position'] == pos
        df_p = df_sorted[mask_p]
        fp_p = {cat: pd.Series(pts).reindex(df_p.index).fillna(0) for cat, pts in fantasy_points.items()}
        lbls_p, cd_p = build_breakdown_data(df_p, fp_p)
        position_bd[pos] = (lbls_p, cd_p)

    # Create figure with "All" data - always create a trace for every category
    fig = go.Figure()
    colors = px.colors.qualitative.Set3
    color_cycle = cycle(colors)
    row_height = 16

    for category in all_categories:
        ps = cat_data_all[category]
        fig.add_trace(
            go.Bar(
                y=labels_all_bd,
                x=ps,
                orientation='h',
                name=category.replace('_', ' ').title(),
                marker_color=next(color_cycle),
                text=ps.round(1 if is_ppm else 0),
                textposition='inside',
                visible=True if ps.sum() > 0 else 'legendonly',
            )
        )

    # Build dropdown buttons
    def make_button_args(lbls, cat_data_dict):
        """Build restyle args dict for all traces."""
        x_vals = [list(cat_data_dict[cat]) for cat in all_categories]
        y_vals = [lbls for _ in all_categories]
        txt_vals = [list(cat_data_dict[cat].round(1 if is_ppm else 0)) for cat in all_categories]
        return x_vals, y_vals, txt_vals

    x_all, y_all, t_all = make_button_args(labels_all_bd, cat_data_all)
    buttons = [dict(
        label='All Countries',
        method='update',
        args=[{'x': x_all, 'y': y_all, 'text': t_all},
              {'height': max(600, len(set(labels_all_bd)) * row_height),
               'yaxis.categoryarray': list(dict.fromkeys(labels_all_bd))}]
    )]
    for country in countries:
        lbls_c, cd_c = country_bd[country]
        x_c, y_c, t_c = make_button_args(lbls_c, cd_c)
        buttons.append(dict(
            label=country,
            method='update',
            args=[{'x': x_c, 'y': y_c, 'text': t_c},
                  {'height': max(600, len(set(lbls_c)) * row_height),
                   'yaxis.categoryarray': list(dict.fromkeys(lbls_c))}]
        ))

    # Position dropdown buttons
    x_all_p, y_all_p, t_all_p = make_button_args(labels_all_bd, cat_data_all)
    pos_buttons = [dict(
        label='All Positions',
        method='update',
        args=[{'x': x_all_p, 'y': y_all_p, 'text': t_all_p},
              {'height': max(600, len(set(labels_all_bd)) * row_height),
               'yaxis.categoryarray': list(dict.fromkeys(labels_all_bd))}]
    )]
    for pos in positions:
        lbls_p, cd_p = position_bd[pos]
        x_p, y_p, t_p = make_button_args(lbls_p, cd_p)
        pos_buttons.append(dict(
            label=pos,
            method='update',
            args=[{'x': x_p, 'y': y_p, 'text': t_p},
                  {'height': max(600, len(set(lbls_p)) * row_height),
                   'yaxis.categoryarray': list(dict.fromkeys(lbls_p))}]
        ))

    # Update title based on display mode
    display_type = 'PPM' if is_ppm else 'Points'
    fig.update_layout(
        title=f'{display_type} Breakdown by Player and Position ({subtitle})<br>Round points = {round_points}. Total fantasy value = {fantasy_values_rnd}',
        xaxis_title=display_type,
        yaxis_title='Player',
        barmode='stack',
        showlegend=True,
        legend_title='Scoring Categories',
        height=max(600, len(set(labels_all_bd)) * row_height),
        width=None,
        margin=dict(l=250, r=200),
        autosize=True,
        yaxis={
            'autorange': 'reversed',
            'categoryorder': 'array',
            'categoryarray': list(dict.fromkeys(labels_all_bd)),
            'dtick': 1
        },
        updatemenus=[
            dict(
                buttons=buttons,
                direction='down',
                showactive=True,
                x=1.35,
                xanchor='left',
                y=1.0,
                yanchor='top',
            ),
            dict(
                buttons=pos_buttons,
                direction='down',
                showactive=True,
                x=1.35,
                xanchor='left',
                y=0.85,
                yanchor='top',
            ),
        ],
    )

    # Update filename based on display mode
    type_text = 'ppm' if is_ppm else 'points'
    html_name = f'{type_text}_breakdown_{filter_text}.html'
    fig.write_html(html_name)
    print(f"Written to {html_name}")
    return fig

def _load_betting_odds(year):
    """Load betting odds for a given year, returns {round: {(home, away): match_odds}}."""
    odds_path = Path('data/output') / f'betting_odds_{year}.json'
    if not odds_path.exists():
        return {}
    try:
        with open(odds_path) as f:
            data = json.load(f)
        lookup = {}
        for rnd_data in data.get('rounds', []):
            rnd = rnd_data['round']
            lookup[rnd] = {}
            for m in rnd_data.get('matches', []):
                lookup[rnd][(m['home'], m['away'])] = m
        return lookup
    except (json.JSONDecodeError, KeyError):
        return {}


def plot_match_fantasy_comparison(df, fixtures=None, subtitle=''):
    """
    Grouped bar chart showing total fantasy points per team in each match,
    so you can see which team 'won' the fantasy battle per fixture.
    Betting odds are displayed beneath each fixture.
    """
    if fixtures is None:
        fixtures = get_fixtures()

    # Try to determine year from data or subtitle
    year = None
    if subtitle and any(c.isdigit() for c in subtitle):
        import re
        year_match = re.search(r'20\d{2}', subtitle)
        if year_match:
            year = int(year_match.group())
    if year is None:
        from config.settings import YEAR
        year = YEAR

    odds_lookup = _load_betting_odds(year)

    # Sum fantasy points per club per round
    club_round_pts = df.groupby(['round', 'club'])['points'].sum().reset_index()

    rounds_available = sorted(df['round'].unique())
    fig = go.Figure()

    x_labels = []
    home_pts = []
    away_pts = []
    home_names = []
    away_names = []

    for rnd in rounds_available:
        if rnd not in fixtures:
            continue
        round_data = club_round_pts[club_round_pts['round'] == rnd]
        for home, away, *_ in fixtures[rnd]:
            # Build x-axis label with odds embedded
            odds_line = ''
            if rnd in odds_lookup and (home, away) in odds_lookup[rnd]:
                m = odds_lookup[rnd][(home, away)]
                h_prob = m.get('home_win_prob', 0)
                a_prob = m.get('away_win_prob', 0)
                handicap = m.get('handicap', 0)
                total = m.get('total_points', 0)
                odds_line = f"<br><span style='font-size:0.8em;color:#666'>{h_prob*100:.0f}% v {a_prob*100:.0f}% | Hcap {handicap:+.1f} | O/U {total:.0f}</span>"

            label = f"R{rnd}: {home} v {away}{odds_line}"
            x_labels.append(label)
            h_pts = round_data.loc[round_data['club'] == home, 'points'].sum()
            a_pts = round_data.loc[round_data['club'] == away, 'points'].sum()
            home_pts.append(h_pts)
            away_pts.append(a_pts)
            home_names.append(home)
            away_names.append(away)

    fig.add_trace(go.Bar(
        x=x_labels,
        y=home_pts,
        name='Home',
        text=[f"{n}<br>{p:.0f}" for n, p in zip(home_names, home_pts)],
        textposition='inside',
        marker_color='#1f77b4',
    ))
    fig.add_trace(go.Bar(
        x=x_labels,
        y=away_pts,
        name='Away',
        text=[f"{n}<br>{p:.0f}" for n, p in zip(away_names, away_pts)],
        textposition='inside',
        marker_color='#ff7f0e',
    ))

    fig.update_layout(
        title=f'Fantasy Points: Head-to-Head per Match<br><sup>{subtitle}</sup>' if subtitle else 'Fantasy Points: Head-to-Head per Match',
        xaxis_title='Match',
        yaxis_title='Total Fantasy Points',
        barmode='group',
        height=600,
        width=1100,
        margin=dict(b=180),
        legend_title='Team',
    )
    fig.update_xaxes(tickangle=30)

    html_name = 'match_fantasy_comparison.html'
    fig.write_html(html_name)
    print(f"Written to {html_name}")
    return fig


def plot_player_round_heatmap(df, max_players=50, subtitle=''):
    """
    Heatmap with players on the Y axis and rounds on the X axis.
    Colour intensity = fantasy points scored that round.
    Players are grouped by position and sorted by total points descending.
    Quickly reveals who is trending up, down, or staying consistent.
    Includes a dropdown filter for country.
    """
    rounds = sorted(df['round'].unique())
    x_labels = [f"Round {r}" for r in rounds]
    countries = sorted(df['club'].unique())

    def build_heatmap_data(df_subset):
        """Build pivot, labels and z-matrix for a subset of data."""
        pivot = df_subset.pivot_table(index=['name', 'club', 'position'],
                                      columns='round', values='points',
                                      aggfunc='sum', fill_value=0)
        pivot['total'] = pivot.sum(axis=1)
        pivot = pivot[pivot['total'] > 0]
        pivot = pivot.sort_values(['position', 'total'], ascending=[True, False])
        pivot = pivot.groupby('position').head(max_players)

        labels = [f"{name} ({club}, {pos}) [{int(row['total'])}]"
                  for (name, club, pos), row in pivot.iterrows()]
        # Reindex to ensure all rounds are present (some countries may not have data for all rounds)
        for r in rounds:
            if r not in pivot.columns:
                pivot[r] = 0
        z = pivot[rounds].values
        return z, labels

    # Pre-compute heatmap data for "All" and each country
    z_all, labels_all = build_heatmap_data(df)
    country_data = {}
    for country in countries:
        country_data[country] = build_heatmap_data(df[df['club'] == country])

    # Create figure with "All" data
    fig = go.Figure(data=go.Heatmap(
        z=z_all,
        x=x_labels,
        y=labels_all,
        colorscale='YlOrRd',
        text=z_all.astype(int).astype(str),
        texttemplate='%{text}',
        textfont=dict(size=10),
        colorbar=dict(title='Points'),
        hoverongaps=False,
    ))

    # Build dropdown buttons (update data + chart height)
    row_height = 18
    buttons = [dict(
        label='All Countries',
        method='update',
        args=[{'z': [z_all], 'y': [labels_all], 'text': [z_all.astype(int).astype(str)]},
              {'height': max(600, len(labels_all) * row_height)}]
    )]
    for country in countries:
        z_c, labels_c = country_data[country]
        buttons.append(dict(
            label=country,
            method='update',
            args=[{'z': [z_c], 'y': [labels_c], 'text': [z_c.astype(int).astype(str)]},
                  {'height': max(600, len(labels_c) * row_height)}]
        ))

    fig.update_layout(
        title=f'Player Fantasy Points per Round<br><sup>{subtitle}</sup>' if subtitle else 'Player Fantasy Points per Round',
        xaxis_title='Round',
        yaxis_title='Player',
        height=max(600, len(labels_all) * 18),
        width=None,
        margin=dict(l=300, r=200),
        autosize=True,
        xaxis=dict(side='top'),
        yaxis=dict(autorange='reversed', dtick=1),
        updatemenus=[dict(
            buttons=buttons,
            direction='down',
            showactive=True,
            x=1.22,
            xanchor='left',
            y=1.0,
            yanchor='top',
        )],
    )

    html_name = 'player_round_heatmap.html'
    fig.write_html(html_name)
    print(f"Written to {html_name}")
    return fig


def plot_scatter_matrix(df, min_total_points=30, subtitle=''):
    """
    Scatter plot matrix showing relationships between key stats.
    Each point is a player, coloured by position.
    Hover shows player name, club, and all stat values.
    """
    stat_cols = ['tries', 'tackles', 'meters_carried', 'defenders_beaten',
                 'breakdown_steals', 'points']

    agg_dict = {col: 'sum' for col in stat_cols}
    agg_dict['fantasy_value'] = 'max'
    agg_dict['club'] = 'first'

    df_agg = df.groupby(['name', 'position']).agg(agg_dict).reset_index()
    df_agg = df_agg[df_agg['points'] >= min_total_points]

    # Readable axis labels
    label_map = {
        'tries': 'Tries',
        'tackles': 'Tackles',
        'meters_carried': 'Meters',
        'defenders_beaten': 'Def Beaten',
        'breakdown_steals': 'Turnovers',
        'points': 'Fantasy Pts',
    }
    # Only include fantasy_value as a dimension if data is available
    if df_agg['fantasy_value'].sum() > 0:
        label_map['fantasy_value'] = 'Fantasy Value'
    df_plot = df_agg.rename(columns=label_map)
    display_cols = list(label_map.values())

    # Hover text: player name + club
    df_plot['hover'] = df_agg['name'] + ' (' + df_agg['club'] + ')'

    fig = px.scatter_matrix(
        df_plot,
        dimensions=display_cols,
        color='position',
        hover_name='hover',
        opacity=0.7,
        size_max=8,
    )

    fig.update_traces(
        diagonal_visible=False,
        marker=dict(size=6),
    )

    fig.update_layout(
        title=f'Player Stat Profiles — colour = position, hover for name<br><sup>{subtitle}</sup>' if subtitle else 'Player Stat Profiles — colour = position, hover for name',
        height=1000,
        width=1100,
        margin=dict(l=60, r=40, t=60, b=40),
    )

    html_name = 'scatter_matrix.html'
    fig.write_html(html_name)
    print(f"Written to {html_name}")
    return fig


def plot_value_profile(df, max_players_per_position=15, subtitle=''):
    """
    Horizontal stacked bar chart showing each player's fantasy points split into
    'high-value' (tries, turnovers, MOTM, 50/22, lineout steals, drop goals, assists)
    vs 'base' (tackles, meters, defenders beaten, offloads, conversions, penalties, scrum wins).
    Sorted by points-per-dollar (total points / fantasy_value) to highlight efficient picks.
    """
    backs = ['Back Three', 'Centre', 'Fly-Half', 'Scrum-Half']

    # Aggregate per player
    stat_cols = ['tries', 'assists', 'conversions', 'penalties', 'drop_goals',
                 'defenders_beaten', 'meters_carried', 'kick_50_22', 'tackles',
                 'breakdown_steals', 'lineout_steals', 'penalties_conceded',
                 'man_of_match', 'yellow_cards', 'red_cards', 'offloads', 'scrum_wins']

    agg_dict = {col: 'sum' for col in stat_cols}
    agg_dict['points'] = 'sum'
    agg_dict['fantasy_value'] = 'max'
    agg_dict['club'] = 'first'

    df_agg = df.groupby(['name', 'position']).agg(agg_dict).reset_index()
    df_agg = df_agg[df_agg['points'] > 0]

    # Calculate fantasy point contributions per category
    is_back = df_agg['position'].isin(backs)
    high_value = (
        df_agg['tries'].where(is_back, 0) * 10 +
        df_agg['tries'].where(~is_back, 0) * 15 +
        df_agg['breakdown_steals'] * 5 +
        df_agg['lineout_steals'] * 7 +
        df_agg['man_of_match'] * 15 +
        df_agg['kick_50_22'] * 7 +
        df_agg['drop_goals'] * 5 +
        df_agg['assists'] * 4
    )

    base = (
        df_agg['tackles'] * 1 +
        (df_agg['meters_carried'] / 10).apply(np.floor) * 1 +
        df_agg['defenders_beaten'] * 2 +
        df_agg['offloads'] * 2 +
        df_agg['conversions'] * 2 +
        df_agg['penalties'] * 3 +
        df_agg['scrum_wins'] * 1
    )

    negative = (
        df_agg['penalties_conceded'] * -1 +
        df_agg['yellow_cards'] * -5 +
        df_agg['red_cards'] * -8
    )

    df_agg['high_value_pts'] = high_value
    df_agg['base_pts'] = base
    df_agg['negative_pts'] = negative

    # Points per dollar
    has_fantasy_values = df_agg['fantasy_value'].sum() > 0
    if has_fantasy_values:
        df_agg['pts_per_dollar'] = (df_agg['points'] / df_agg['fantasy_value']).round(1)
        sort_col = 'pts_per_dollar'
    else:
        sort_col = 'points'

    # Top N per position
    top_players = (df_agg
        .sort_values(['position', sort_col], ascending=[True, False])
        .drop_duplicates(subset='name')
        .groupby('position')
        .head(max_players_per_position)
    )['name']

    df_plot = df_agg[df_agg['name'].isin(top_players)].copy()
    df_plot = df_plot.sort_values(['position', sort_col], ascending=[True, False])

    # Labels
    if has_fantasy_values:
        labels = df_plot.apply(
            lambda r: f"{r['name']} ({r['position']}) [${r['fantasy_value']}, {r[sort_col]} pts/$]",
            axis=1
        )
    else:
        labels = df_plot.apply(
            lambda r: f"{r['name']} ({r['position']}) [{int(r['points'])} pts]",
            axis=1
        )

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=labels, x=df_plot['high_value_pts'], orientation='h',
        name='High-Value (tries, turnovers, MOTM, 50/22, assists)',
        marker_color='#2ca02c',
        text=df_plot['high_value_pts'].round(0).astype(int),
        textposition='inside',
    ))
    fig.add_trace(go.Bar(
        y=labels, x=df_plot['base_pts'], orientation='h',
        name='Base (tackles, meters, beaten, offloads, kicking, scrums)',
        marker_color='#1f77b4',
        text=df_plot['base_pts'].round(0).astype(int),
        textposition='inside',
    ))
    if (df_plot['negative_pts'] < 0).any():
        fig.add_trace(go.Bar(
            y=labels, x=df_plot['negative_pts'], orientation='h',
            name='Negative (pens conceded, cards)',
            marker_color='#d62728',
            text=df_plot['negative_pts'].round(0).astype(int),
            textposition='inside',
        ))

    sort_label = 'Points per $' if has_fantasy_values else 'Total Points'
    fig.update_layout(
        title=f'Player Value Profile — High-Value vs Base Points (sorted by {sort_label})<br><sup>{subtitle}</sup>' if subtitle else f'Player Value Profile — High-Value vs Base Points (sorted by {sort_label})',
        xaxis_title='Fantasy Points',
        yaxis_title='Player',
        barmode='stack',
        showlegend=True,
        height=max(600, len(df_plot) * 22),
        margin=dict(l=300),
        yaxis={
            'autorange': 'reversed',
            'categoryorder': 'array',
            'categoryarray': list(labels),
            'dtick': 1
        }
    )

    html_name = 'value_profile.html'
    fig.write_html(html_name)
    print(f"Written to {html_name}")
    return fig


def plot_points_distribution_by_category(df, subtitle='', year=None):
    """
    Creates a stacked bar chart showing the points distribution across different scoring categories,
    with each bar segment representing a different round.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing player stats and points with a 'round' column
    year : int, optional
        Year for scoring rules (uses default if None)
    """

    # Calculate points for each category
    fantasy_points = calculate_fantasy_points(df, year=year)
    
    # Get all unique rounds
    rounds = sorted(df['round'].unique())
    
    # Create a dictionary to store points by category and round
    category_round_points = {}
    
    # For each category, calculate points by round
    for category, points_series in fantasy_points.items():
        # Create a DataFrame with round and points
        round_points_df = pd.DataFrame({
            'round': df['round'],
            'points': points_series
        })
        
        # Group by round and sum points
        round_totals = round_points_df.groupby('round')['points'].sum()
        
        # Store in our dictionary
        category_round_points[category] = round_totals
    
    # Calculate total points for each category (for sorting)
    category_totals = {category: points.sum() for category, points in fantasy_points.items()}
    
    # Sort categories by total points
    sorted_categories = sorted(category_totals.items(), key=lambda x: x[1], reverse=True)
    categories = [category for category, _ in sorted_categories]
    
    # Format category names for display
    display_categories = [category.replace('_', ' ').title() for category in categories]
    
    # Create a stacked bar chart
    fig = go.Figure()
    
    # Color scale for rounds
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Add a trace for each round
    for i, round_num in enumerate(rounds):
        round_values = []
        
        # Get points for this round for each category
        for category in categories:
            if round_num in category_round_points[category].index:
                round_values.append(category_round_points[category][round_num])
            else:
                round_values.append(0)
        
        # Add the bar for this round
        fig.add_trace(go.Bar(
            name=f'Round {round_num}',
            x=display_categories,
            y=round_values,
            marker_color=colors[i % len(colors)]
        ))
    
    # Add total point values as text annotations
    total_annotations = []
    for i, category in enumerate(display_categories):
        original_category = categories[i]
        total = category_totals[original_category]
        total_annotations.append(
            dict(
                x=category,
                y=total,
                text=f'{total:.1f}',
                font=dict(family="Arial", size=12),
                showarrow=False,
                yshift=10
            )
        )
    
    # Customize layout
    fig.update_layout(
        title=f'Points Distribution by Scoring Category and Round<br><sup>{subtitle}</sup>' if subtitle else 'Points Distribution by Scoring Category and Round',
        xaxis_title='Scoring Category',
        yaxis_title='Points',
        height=600,
        width=1000,
        margin=dict(b=100, t=50),  # Adjust bottom margin for rotated labels
        barmode='stack',  # Stack the bars
        annotations=total_annotations,  # Add total annotations
        legend_title_text='Round',
        hovermode='x'
    )
    
    # Rotate x-axis labels for better readability
    fig.update_xaxes(tickangle=45)
    
    # Save and display
    html_name = 'points_distribution_by_category_and_round.html'
    fig.write_html(html_name)
    print(f"Written to {html_name}")

    return fig


def optimize_squad(df, year=None, subtitle=''):
    """
    Uses Integer Linear Programming to find the optimal 15-player squad
    within the budget constraint, maximizing total expected points.

    Squad structure (standard rugby):
    - 2 Props, 1 Hooker, 2 Second Row, 3 Back Row (forwards = 8)
    - 1 Scrum-Half, 1 Fly-Half, 2 Centres, 3 Back Three (backs = 7)

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with player stats including 'fantasy_value' and 'points'
    year : int
        Year for scoring rules (determines budget). If None, uses default from settings.
    subtitle : str
        Subtitle for the chart
    """
    # Get budget from year-specific scoring rules
    rules = get_scoring_rules(year)
    budget = rules.get('budget', 230)  # Fallback to 230 if not specified

    # Position requirements for a rugby squad
    POSITION_REQUIREMENTS = {
        'Prop': 2,
        'Hooker': 1,
        'Second Row': 2,
        'Back Row': 3,
        'Scrum-Half': 1,
        'Fly-Half': 1,
        'Centre': 2,
        'Back Three': 3
    }

    # Aggregate player stats - use most recent round's data
    df_latest = df.sort_values('round', ascending=False).drop_duplicates(subset=['name'], keep='first')

    # Filter to players with fantasy values
    df_valid = df_latest[df_latest['fantasy_value'] > 0].copy()

    if df_valid.empty:
        print("No players with fantasy values found. Cannot optimize.")
        return None

    # Calculate average points per round for each player
    player_avg_points = df.groupby('name')['points'].mean()
    df_valid['avg_points'] = df_valid['name'].map(player_avg_points)

    # Create the optimization problem
    prob = LpProblem("Fantasy_Squad_Optimizer", LpMaximize)

    # Create binary decision variables for each player
    players = df_valid['name'].tolist()
    player_vars = LpVariable.dicts("player", players, cat='Binary')

    # Objective: Maximize total average points
    prob += lpSum([player_vars[p] * df_valid[df_valid['name'] == p]['avg_points'].values[0]
                   for p in players]), "Total_Points"

    # Budget constraint (max only - let optimizer find best value)
    prob += lpSum([player_vars[p] * df_valid[df_valid['name'] == p]['fantasy_value'].values[0]
                   for p in players]) <= budget, "Budget"

    # Position constraints
    for position, required in POSITION_REQUIREMENTS.items():
        position_players = df_valid[df_valid['position'] == position]['name'].tolist()
        prob += lpSum([player_vars[p] for p in position_players]) == required, f"Position_{position}"

    # Country constraints - max 4 players per country (fantasy rule)
    COUNTRIES = ['France', 'England', 'Ireland', 'Scotland', 'Wales', 'Italy']
    MAX_PER_COUNTRY = 4
    for country in COUNTRIES:
        country_players = df_valid[df_valid['club'] == country]['name'].tolist()
        if country_players:
            prob += lpSum([player_vars[p] for p in country_players]) <= MAX_PER_COUNTRY, f"Country_{country}"

    # Solve the problem (suppress solver output)
    prob.solve(PULP_CBC_CMD(msg=0))

    if LpStatus[prob.status] != 'Optimal':
        print(f"Optimization failed: {LpStatus[prob.status]}")
        return None

    # Extract selected players
    selected = []
    for p in players:
        if player_vars[p].value() == 1:
            player_data = df_valid[df_valid['name'] == p].iloc[0]
            selected.append({
                'name': p,
                'position': player_data['position'],
                'club': player_data['club'],
                'value': player_data['fantasy_value'],
                'avg_points': player_data['avg_points'],
                'pts_per_dollar': player_data['avg_points'] / player_data['fantasy_value']
            })

    # Sort by position order for display
    position_order = ['Prop', 'Hooker', 'Second Row', 'Back Row', 'Scrum-Half', 'Fly-Half', 'Centre', 'Back Three']
    selected_df = pd.DataFrame(selected)
    selected_df['pos_order'] = selected_df['position'].map({p: i for i, p in enumerate(position_order)})
    selected_df = selected_df.sort_values('pos_order')

    total_value = selected_df['value'].sum()
    total_points = selected_df['avg_points'].sum()

    # Identify captain (highest avg points - doubling them gives most benefit)
    captain_idx = selected_df['avg_points'].idxmax()
    captain = selected_df.loc[captain_idx]
    captain_bonus = captain['avg_points']  # Captain doubles, so bonus = their base points
    total_with_captain = total_points + captain_bonus

    # Mark captain in display
    selected_df['is_captain'] = selected_df.index == captain_idx
    display_names = selected_df.apply(
        lambda r: f"(C) {r['name']}" if r['is_captain'] else r['name'], axis=1
    )

    # Create visualization - table with squad
    fig = go.Figure()

    # Add table
    fig.add_trace(go.Table(
        header=dict(
            values=['Position', 'Player', 'Team', 'Value ($)', 'Avg Pts', 'Pts/$'],
            fill_color='#2c3e50',
            font=dict(color='white', size=12),
            align='left'
        ),
        cells=dict(
            values=[
                selected_df['position'],
                display_names,
                selected_df['club'],
                selected_df['value'].round(1),
                selected_df['avg_points'].round(1),
                selected_df['pts_per_dollar'].round(2)
            ],
            fill_color=[['#ffd700' if selected_df.iloc[i]['is_captain'] else ('#ecf0f1' if i % 2 == 0 else 'white') for i in range(len(selected_df))]],
            align='left',
            height=25
        )
    ))

    fig.update_layout(
        title=f'Optimal Squad (Budget: ${budget}, Used: ${total_value:.1f}, Projected Pts: {total_with_captain:.1f} with Captain)<br><sup>{subtitle}</sup>' if subtitle else f'Optimal Squad (Budget: ${budget}, Used: ${total_value:.1f}, Projected Pts: {total_with_captain:.1f} with Captain)',
        height=500,
        width=800,
        margin=dict(t=80, b=20)
    )

    html_name = 'optimal_squad.html'
    fig.write_html(html_name)
    print(f"Written to {html_name}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"OPTIMAL SQUAD (Budget: ${budget})")
    print(f"{'='*60}")
    print(f"Total Value: ${total_value:.1f} / ${budget}")
    print(f"Base Points: {total_points:.1f}")
    print(f"Captain: {captain['name']} (+{captain_bonus:.1f} pts)")
    print(f"Total with Captain: {total_with_captain:.1f}")
    print(f"{'='*60}")
    for _, row in selected_df.iterrows():
        marker = "(C)" if row['is_captain'] else "   "
        print(f"{marker} {row['position']:12} {row['name']:22} {row['club']:10} ${row['value']:5.1f}  {row['avg_points']:5.1f} pts")
    print(f"{'='*60}")

    return fig