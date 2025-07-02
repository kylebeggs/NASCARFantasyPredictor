#!/usr/bin/env python3
"""Convert Lap Raptor data to NASCAR predictor format."""

import pandas as pd
from datetime import datetime

def convert_lapraptor_to_nascar_format(input_file, output_file):
    """Convert Lap Raptor CSV format to NASCAR predictor format."""
    
    # Read the data
    df = pd.read_csv(input_file)
    
    print(f"Original data: {len(df)} records")
    print(f"Date range: {df['race_date'].min()} to {df['race_date'].max()}")
    print(f"Unique tracks: {df['track_name'].nunique()}")
    print(f"Unique drivers: {df['driver_name'].nunique()}")
    
    # Convert to NASCAR predictor format
    converted_df = pd.DataFrame({
        'date': pd.to_datetime(df['race_date']).dt.strftime('%Y-%m-%d'),
        'race_name': df['track_name'] + ' Race',  # Create race name from track
        'track_name': df['track_name'],
        'driver_name': df['driver_name'],
        'car_number': None,  # Not available in Lap Raptor data
        'finish_position': df['finish'],
        'start_position': df['start'],
        'laps_led': df['laps_led'].fillna(0),
        'total_laps': df['laps'],
        'team': None,  # Not available in Lap Raptor data
        'manufacturer': None,  # Not available in Lap Raptor data
        'status': df['status'],
        'series': df['series'],
        # Additional Lap Raptor specific data
        'mid_position': df['mid'],
        'best_position': df['best'],
        'worst_position': df['worst'],
        'avg_position': df['avg'],
        'green_flag_passing_diff': df['green_flag_passing_diff'],
        'green_flag_passes': df['green_flag_passes'],
        'rating': df['rating']
    })
    
    # Filter for Cup series only (most relevant for predictions)
    cup_data = converted_df[converted_df['series'] == 'Cup'].copy()
    
    # Remove rows with missing essential data
    cup_data = cup_data.dropna(subset=['driver_name', 'finish_position'])
    
    print(f"\\nFiltered Cup series data: {len(cup_data)} records")
    print(f"Date range: {cup_data['date'].min()} to {cup_data['date'].max()}")
    print(f"Unique races: {cup_data['date'].nunique()}")
    
    # Save the converted data
    cup_data.to_csv(output_file, index=False)
    print(f"\\nConverted data saved to: {output_file}")
    
    return cup_data

if __name__ == "__main__":
    convert_lapraptor_to_nascar_format('combined_lapraptor_data.csv', 'nascar_formatted_data.csv')