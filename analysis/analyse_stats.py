import json
from pathlib import Path
import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def run_analysis(round=1):
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

    # Clean and convert points and matches_played to numeric
    df['points'] = pd.to_numeric(df['points'].replace('', '0'), errors='coerce').fillna(0)
    df['minutes'] = pd.to_numeric(df['matches_played'].replace('', '0'), errors='coerce').fillna(0)
    # Calculate points per minute and convert to integer
    df['points_per_min'] = df.apply(
        lambda row: int((row['points'] / row['minutes']) * 100) if row['minutes'] > 0 else 0,
        axis=1
    ) 

    # round filtering
    if round != None:
        df= df[df['round'] == round]
    
    # save df to csv
    output_file = data_dir / '6N_combined_stats.csv'
    df.to_csv(output_file)
    print(f"Saved combined stats to: {output_file}")
    
    try:
        # Run visualizations
        visualise_supersub(df)
        print("Created super sub analysis visualization")
        
        visualise_supersub_by_position(df)
        print("Created super sub by position visualization")
        
        #visualise_supersub_by_position(df, False)
        
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
            'fantasy_value': player.get('fantasy_value', ''),
            'fantasy_position_group': player.get('fantasy_position_group', ''),
            **player['stats']  # Unpack all stats fields
        }
        for f in files
        for json_data in [json.load(f.open())]
        for player in json_data['players']
    ]
    
    return pd.DataFrame(data_list)


def visualise_supersub(df):
    # Create a horizontal bar chart comparing points per minute and minutes played for substitutes.
    # Points per minute on one set of bars on primary axis
    # Minutes played on another set of bars on secondary axis
    # The bars should not be stacked on top of each other, but should be offset
    # The x axis 0 point should line up with the seconday axis 0 point
    
    # Filter for substitutes and create explicit copy
    df = df[df['fantasy_position_group'] == 'Subs'].copy()

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
        title='Super Sub Analysis - Points per Minute and Minutes Played',
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
    fig.write_html('super_sub_analysis.html')
    
    return fig  # Optional: return the figure for further manipulation

def visualise_supersub_by_position(df, supersub=True):
    """
    Create a horizontal bar chart showing points per minute for substitutes, grouped by position.
    
    Args:
        data (list): List containing player data with stats dictionaries
        
    Returns:
        plotly.graph_objects.Figure: The generated visualization
    """    
    # Filter for substitutes only
    if supersub:    
        df = df[df['fantasy_position_group'] == 'Subs'].copy()
    
    # Sort by points per minute within each position
  
    df_sorted = df.sort_values(['position', 'points_per_min'], ascending=[True, False])
    
    # Create figure
    fig = go.Figure()
    
    # Create a bar trace for each position
    for position in df_sorted['position'].unique():
        position_data = df_sorted[df_sorted['position'] == position]
        
        fig.add_trace(
            go.Bar(
                y=position_data['name'],
                x=position_data['points_per_min'],
                orientation='h',
                name=position
            )
        )
    
    # Update layout
    fig.update_layout(
        title='Super Sub Analysis - Points per Minute by Position',
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
    fig.write_html('super_sub_position_analysis.html')
    
    return fig