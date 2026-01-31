import json
from pathlib import Path
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from itertools import cycle
import plotly.express as px
import math

FIXTURES = {
    1: [('France', 'Wales'), ('Scotland', 'Italy'), ('Ireland', 'England')],
    2: [('Italy', 'Wales'), ('England', 'France'), ('Scotland', 'Ireland')],
    3: [('Wales', 'Ireland'), ('England', 'Scotland'), ('Italy', 'France')],
    4: [('Ireland', 'France'), ('Scotland', 'Wales'), ('England', 'Italy')],
    5: [('Ireland', 'Italy'), ('England', 'Wales'), ('France', 'Scotland')],
}

def run_analysis(round=None, match=None, max_players_per_position=15):
    """
    Run all analysis visualizations for all available data
    """
    print("\nRunning analysis...")
    
    # Load data
    env = 'ci' if os.environ.get('GITHUB_ACTIONS') else 'local'
    data_dir = Path('stats-output' if env == 'ci' else 'data/output')
    df = load_json_data(data_dir)
    
    if df.empty:
        print("No data available for analysis")
        return

    # clean the data
    df_all_rounds = df_clean(df)

    # round filtering
    if round != None:
        df_cleaned = df_all_rounds[df_all_rounds['round'] == round]
    else:
        df_cleaned = df_all_rounds

    # match filtering (requires round)
    if match is not None and round is not None and round in FIXTURES:
        fixture = FIXTURES[round][match - 1]
        match_clubs = list(fixture)
        df_cleaned = df_cleaned[df_cleaned['club'].isin(match_clubs)]
        print(f"Filtered to match {match}: {fixture[0]} v {fixture[1]}")

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
        # Run visualizations
        #ppm_vs_player_bar_chart(df_cleaned)
        
        df_supersub, filter_text = filter_by_supersubs(df_cleaned)
        ppm_vs_player_bar_chart(df_supersub, filter_text)
        
        ppm_per_position_bar_chart(df_supersub, filter_text)
        
        #df_filt = filter_out_tries(df_cleaned)
        #df_filt = filter_by_round(df_cleaned, [1,2,3])
        #df_filt = filter_by_team(df_cleaned, ['Wales','England','Scotland'])
        #df_filt = filter_by_value(df_filt, 14.7)
        #df_filt = filter_by_players(df_filt, sub_list)
        
        df_filt = df_cleaned
        plot_player_points_breakdown(df_filt, min_points=10, max_players_per_position=max_players_per_position, is_ppm=True)
        plot_player_points_breakdown(df_filt, min_points=10, max_players_per_position=max_players_per_position, is_ppm=False)
        
        plot_points_distribution_by_category(df_cleaned)
        plot_match_fantasy_comparison(df_cleaned)
        plot_player_round_heatmap(df_cleaned, max_players=max_players_per_position)
        plot_scatter_matrix(df_cleaned)
        plot_value_profile(df_cleaned, max_players_per_position=max_players_per_position)

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
        
def load_json_data(data_dir):
    """
    Load all JSON files and flatten into a single DataFrame
    """
    files = list(data_dir.glob('*.json'))
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
    
def calculate_fantasy_points(df):
    # Define position groups
    backs = ['Back Three', 'Centre', 'Fly-Half', 'Scrum-Half']
    forwards = ['Prop', 'Hooker', 'Second Row', 'Back Row']
    
    # Initialize try points based on position
    tries_back = df.apply(lambda row: row['tries'] * 10 if row['position'] in backs else 0, axis=1)
    tries_forward = df.apply(lambda row: row['tries'] * 15 if row['position'] in forwards else 0, axis=1)
    
    fantasy_points = {
        # attacking
        'tries_back': tries_back,
        'tries_forward': tries_forward,
        'assists': df['assists'] * 4,
        'conversions': df['conversions'] * 2,
        'penalties': df['penalties'] * 3,
        'drop_goals': df['drop_goals'] * 5,
        'defenders_beaten': df['defenders_beaten'] * 2,
        'meters': (df['meters_carried'] / 10).apply(np.floor) * 1,  # Floor to nearest 10m
        'kick_50_22': df['kick_50_22'] * 7,
        'offloads': df['offloads'] * 2,
        'scrum_wins': df['scrum_wins'] * 1, # 2 if offload
        # defensive
        'tackles': df['tackles'] * 1,
        'breakdown_steals': df['breakdown_steals'] * 5,
        'lineout_steals': df['lineout_steals'] * 7,
        'penalties_conceded': df['penalties_conceded'] * -1,
        # general
        'man_of_match': df['man_of_match'] * 15,
        'yellow_cards': df['yellow_cards'] * -5,
        'red_cards': df['red_cards'] * -8
    }
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

def ppm_vs_player_bar_chart(df, filter_text='all_players'):
    # Create a horizontal bar chart comparing points per minute and minutes played for substitutes.
    # Points per minute on one set of bars on primary axis
    # Minutes played on another set of bars on secondary axis
    # The bars should not be stacked on top of each other, but should be offset
    # The x axis 0 point should line up with the seconday axis 0 point
    
    plot_title = f'Points per Minute and Minutes Played. Filterered by {filter_text}'
    
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

def ppm_per_position_bar_chart(df, filter_text='all_players'):
    """
    Create a horizontal bar chart showing points per minute for substitutes, grouped by position.
    Positions are ordered by their average points.
    """
    plot_title = f'Points per Minute and Minutes Played. Grouped by position. Filtered by {filter_text}'
    
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


def plot_player_points_breakdown(df, min_points=10, max_players_per_position=20, filter_text="None", is_ppm=False):
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
   
    fantasy_points = calculate_fantasy_points(df_aggregated)
   
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
    
    # Create figure
    fig = go.Figure()
   
    # Get plotly's qualitative color sequence
    colors = px.colors.qualitative.Set3
    color_cycle = cycle(colors)
    
    for category, points in fantasy_points.items():
        # Align points with sorted DataFrame
        points_subset = pd.Series(points).reindex(df_sorted.index).fillna(0)
        
        if is_ppm:
            # Convert fantasy points to PPM for current round
            points_subset = points_subset * df_sorted['points_per_min'] / df_sorted['points']
            
        mask = points_subset > 0
       
        if mask.any():
            fig.add_trace(
                go.Bar(
                    y=all_labels,
                    x=points_subset,
                    orientation='h',
                    name=category.replace('_', ' ').title(),
                    marker_color=next(color_cycle),
                    text=points_subset.round(1 if is_ppm else 0),
                    textposition='inside',
                )
            )
   
    # Update title based on display mode
    display_type = 'PPM' if is_ppm else 'Points'
    fig.update_layout(
        title=f'{display_type} Breakdown by Player and Position. Round points = {round_points}. Total fantasy value = {fantasy_values_rnd}',
        xaxis_title=display_type,
        yaxis_title='Player',
        barmode='stack',
        showlegend=True,
        legend_title='Scoring Categories',
        height=max(600, df_sorted['name'].nunique() * 20),
        margin=dict(l=250),
        yaxis={
            'autorange': 'reversed',
            'categoryorder': 'array',
            'categoryarray': list(dict.fromkeys(all_labels)),
            'dtick': 1
        }
    )
   
    # Update filename based on display mode
    type_text = 'ppm' if is_ppm else 'points'
    html_name = f'{type_text}_breakdown_{filter_text}.html'
    fig.write_html(html_name)
    print(f"Written to {html_name}")
    return fig

def plot_match_fantasy_comparison(df):
    """
    Grouped bar chart showing total fantasy points per team in each match,
    so you can see which team 'won' the fantasy battle per fixture.
    """
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
        if rnd not in FIXTURES:
            continue
        round_data = club_round_pts[club_round_pts['round'] == rnd]
        for home, away in FIXTURES[rnd]:
            label = f"R{rnd}: {home} v {away}"
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
        title='Fantasy Points: Head-to-Head per Match',
        xaxis_title='Match',
        yaxis_title='Total Fantasy Points',
        barmode='group',
        height=500,
        width=1100,
        margin=dict(b=120),
        legend_title='Team',
    )
    fig.update_xaxes(tickangle=30)

    html_name = 'match_fantasy_comparison.html'
    fig.write_html(html_name)
    print(f"Written to {html_name}")
    return fig


def plot_player_round_heatmap(df, max_players=50):
    """
    Heatmap with players on the Y axis and rounds on the X axis.
    Colour intensity = fantasy points scored that round.
    Players are grouped by position and sorted by total points descending.
    Quickly reveals who is trending up, down, or staying consistent.
    """
    rounds = sorted(df['round'].unique())

    # Pivot: one row per player, one column per round
    pivot = df.pivot_table(index=['name', 'club', 'position'],
                           columns='round', values='points',
                           aggfunc='sum', fill_value=0)

    pivot['total'] = pivot.sum(axis=1)
    pivot = pivot[pivot['total'] > 0]
    pivot = pivot.sort_values(['position', 'total'], ascending=[True, False])

    # Limit per position to keep the chart readable
    pivot = pivot.groupby('position').head(max_players)

    # Build labels: "Player (Country, Position) [total]"
    labels = [f"{name} ({club}, {pos}) [{int(row['total'])}]"
              for (name, club, pos), row in pivot.iterrows()]

    z = pivot[rounds].values
    x_labels = [f"Round {r}" for r in rounds]

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=x_labels,
        y=labels,
        colorscale='YlOrRd',
        text=z.astype(int).astype(str),
        texttemplate='%{text}',
        textfont=dict(size=10),
        colorbar=dict(title='Points'),
        hoverongaps=False,
    ))

    fig.update_layout(
        title='Player Fantasy Points per Round',
        xaxis_title='Round',
        yaxis_title='Player',
        height=max(600, len(labels) * 18),
        width=800,
        margin=dict(l=300),
        yaxis=dict(autorange='reversed', dtick=1),
    )

    html_name = 'player_round_heatmap.html'
    fig.write_html(html_name)
    print(f"Written to {html_name}")
    return fig


def plot_scatter_matrix(df, min_total_points=30):
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
        title='Player Stat Profiles — colour = position, hover for name',
        height=1000,
        width=1100,
        margin=dict(l=60, r=40, t=60, b=40),
    )

    html_name = 'scatter_matrix.html'
    fig.write_html(html_name)
    print(f"Written to {html_name}")
    return fig


def plot_value_profile(df, max_players_per_position=15):
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
        title=f'Player Value Profile — High-Value vs Base Points (sorted by {sort_label})',
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


def plot_points_distribution_by_category(df):
    """
    Creates a stacked bar chart showing the points distribution across different scoring categories,
    with each bar segment representing a different round.
   
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing player stats and points with a 'round' column
    """

    # Calculate points for each category
    fantasy_points = calculate_fantasy_points(df)
    
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
        title='Points Distribution by Scoring Category and Round',
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