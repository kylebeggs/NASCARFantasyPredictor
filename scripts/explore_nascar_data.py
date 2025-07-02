#!/usr/bin/env python3
"""Properly explore NASCAR API data structure."""

import requests
import json

def explore_nascar_data():
    """Properly explore the NASCAR API data."""
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })
    
    url = "https://cf.nascar.com/cacher/2024/race_list_basic.json"
    
    try:
        response = session.get(url)
        data = response.json()
        
        print("🏁 NASCAR API Data Structure 🏁\n")
        print(f"Top level type: {type(data)}")
        print(f"Top level keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
        
        # Print the actual structure
        print(f"\nFull data structure:")
        print(json.dumps(data, indent=2)[:2000] + "..." if len(json.dumps(data)) > 2000 else json.dumps(data, indent=2))
        
        # If it's a dict, explore each series
        if isinstance(data, dict):
            for series_key, series_data in data.items():
                print(f"\n📊 Series: {series_key}")
                print(f"Type: {type(series_data)}")
                
                if isinstance(series_data, list) and len(series_data) > 0:
                    print(f"Length: {len(series_data)}")
                    print(f"First item: {json.dumps(series_data[0], indent=2)}")
                    
                    # Look for race information
                    for item in series_data[:3]:  # Check first 3 items
                        if isinstance(item, dict):
                            print(f"Item keys: {list(item.keys())}")
                            
                            # Check if this looks like race data
                            if any(key in item for key in ['race_id', 'race_name', 'track', 'date']):
                                print(f"🏁 Found race data in {series_key}!")
                                break
                
                elif isinstance(series_data, dict):
                    print(f"Keys: {list(series_data.keys())}")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    explore_nascar_data()