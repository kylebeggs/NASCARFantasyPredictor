#!/usr/bin/env python3
"""Check for race details and results in the main NASCAR API."""

import requests
import json

def check_race_details():
    """Check detailed race information from NASCAR API."""
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })
    
    # Get 2024 race data
    url = "https://cf.nascar.com/cacher/2024/race_list_basic.json"
    
    try:
        response = session.get(url)
        data = response.json()
        
        # Look at Cup Series races
        cup_races = data['series_1']
        
        print(f"🏁 Analyzing {len(cup_races)} Cup Series races\\n")
        
        # Check if winner info is already in the race data
        races_with_winners = 0
        races_with_stages = 0
        
        for i, race in enumerate(cup_races[:10]):  # Check first 10 races
            race_name = race.get('race_name')
            winner_id = race.get('winner_driver_id')
            has_stages = bool(race.get('stage_results'))
            
            print(f"{i+1}. {race_name}")
            print(f"   Winner ID: {winner_id}")
            print(f"   Has stage results: {has_stages}")
            
            if winner_id:
                races_with_winners += 1
            if has_stages:
                races_with_stages += 1
                
                # Show stage results
                stage_results = race.get('stage_results', [])
                print(f"   Stages: {len(stage_results)}")
                
                for stage in stage_results[:1]:  # Show first stage
                    stage_num = stage.get('stage_number')
                    results = stage.get('results', [])
                    print(f"     Stage {stage_num}: {len(results)} drivers")
                    
                    # Show top 3 from stage
                    for j, result in enumerate(results[:3]):
                        driver_name = result.get('driver_fullname')
                        car_num = result.get('car_number')
                        points = result.get('stage_points')
                        print(f"       {j+1}. {driver_name} #{car_num} ({points} pts)")
            
            print()
        
        print(f"Summary:")
        print(f"Races with winner data: {races_with_winners}/10")
        print(f"Races with stage results: {races_with_stages}/10")
        
        # Try to find detailed results for a specific race with stage data
        race_with_stages = None
        for race in cup_races:
            if race.get('stage_results'):
                race_with_stages = race
                break
        
        if race_with_stages:
            race_id = race_with_stages['race_id']
            print(f"\\n🔍 Trying alternative result URLs for race {race_id}...")
            
            # Try many different URL patterns
            result_urls = [
                f"https://cf.nascar.com/cacher/{race_id}/results.json",
                f"https://cf.nascar.com/cacher/2024/{race_id}/results.json",
                f"https://cf.nascar.com/live/{race_id}/results.json",
                f"https://cf.nascar.com/live/race-center/{race_id}/results.json",
                f"https://cf.nascar.com/live/race/{race_id}/results.json",
                f"https://cf.nascar.com/cacher/live/{race_id}/results.json",
                f"https://cf.nascar.com/cacher/{race_id}/race_results.json",
                f"https://cf.nascar.com/cacher/race/{race_id}/results.json",
                f"https://feed.nascar.com/live/{race_id}/results.json",
                f"https://api.nascar.com/race/{race_id}/results"
            ]
            
            for url in result_urls:
                try:
                    response = session.get(url, timeout=5)
                    print(f"  {url}: {response.status_code}")
                    
                    if response.status_code == 200:
                        try:
                            result_data = response.json()
                            print(f"    ✅ JSON data found! Keys: {list(result_data.keys()) if isinstance(result_data, dict) else 'Not a dict'}")
                            
                            if isinstance(result_data, dict):
                                for key, value in result_data.items():
                                    if isinstance(value, list) and len(value) > 0:
                                        print(f"      {key}: {len(value)} items")
                                        if isinstance(value[0], dict):
                                            print(f"        Sample keys: {list(value[0].keys())[:5]}")
                        except:
                            print(f"    Response preview: {response.text[:100]}...")
                            
                except Exception as e:
                    print(f"  {url}: Error - {e}")
        
        else:
            print("\\nNo races found with stage results to test detailed results")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_race_details()