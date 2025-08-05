"""LapRaptor.com data scraper for NASCAR race results and loop data."""

import logging
import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class LapRaptorScraper:
    """Scraper for LapRaptor.com NASCAR data."""
    
    def __init__(self, delay: float = 1.0):
        """Initialize the scraper with rate limiting."""
        self.base_url = "https://lapraptor.com"
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def _get_page(self, url: str) -> Optional[BeautifulSoup]:
        """Get and parse a web page with error handling."""
        try:
            time.sleep(self.delay)
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return BeautifulSoup(response.content, 'html.parser')
        except requests.RequestException as e:
            logger.error(f"Error fetching {url}: {e}")
            return None
    
    def get_2025_race_ids(self) -> List[Dict]:
        """Get all 2025 NASCAR Cup Series race IDs and basic info."""
        # Known 2025 race IDs from our earlier research
        known_race_ids = [
            '5543', '5546', '5544', '5545', '5551', '5549', '5548', '5583',
            '5558', '5550', '5555', '5554', '5557', '5562', '5563', '5573',
            '5552', '5569'  # Adding Grant Park 165 mentioned in the first fetch
        ]
        
        races = []
        
        # Test each known race ID
        for race_id in known_race_ids:
            race_url = f"{self.base_url}/races/{race_id}/"
            soup = self._get_page(race_url)
            
            if soup:
                # Check if this is a 2025 race
                page_text = soup.get_text()
                if '2025' in page_text:
                    # Extract race name
                    title_elem = soup.find('h1') or soup.find('title')
                    race_name = title_elem.get_text(strip=True) if title_elem else f"Race {race_id}"
                    
                    # Extract date
                    date_match = re.search(r'(\d{1,2}/\d{1,2}/2025)', page_text)
                    if date_match:
                        race_date = date_match.group(1)
                        races.append({
                            'race_id': race_id,
                            'race_name': race_name,
                            'date': race_date,
                            'url': race_url
                        })
                        logger.info(f"Found 2025 race: {race_id} - {race_name} on {race_date}")
        
        return races
    
    def get_race_results(self, race_id: str) -> Optional[pd.DataFrame]:
        """Get race results for a specific race ID."""
        race_url = f"{self.base_url}/races/{race_id}/"
        soup = self._get_page(race_url)
        
        if not soup:
            return None
        
        try:
            # Find the results table
            results_table = soup.find('table', class_='table')
            if not results_table:
                logger.warning(f"No results table found for race {race_id}")
                return None
            
            # Parse table headers
            headers = []
            header_row = results_table.find('thead')
            if header_row:
                for th in header_row.find_all('th'):
                    headers.append(th.get_text(strip=True))
            
            # Parse table data
            data = []
            tbody = results_table.find('tbody')
            if tbody:
                for row in tbody.find_all('tr'):
                    row_data = []
                    for td in row.find_all('td'):
                        row_data.append(td.get_text(strip=True))
                    if row_data:
                        data.append(row_data)
            
            if not data:
                logger.warning(f"No data found for race {race_id}")
                return None
            
            # Create DataFrame
            df = pd.DataFrame(data, columns=headers)
            
            # Get race metadata
            race_info = self._extract_race_info(soup)
            
            # Add race metadata to each row
            for key, value in race_info.items():
                df[key] = value
            
            return df
            
        except Exception as e:
            logger.error(f"Error parsing race results for {race_id}: {e}")
            return None
    
    def get_loop_data(self, race_id: str) -> Optional[pd.DataFrame]:
        """Get loop data for a specific race ID."""
        loop_url = f"{self.base_url}/races/{race_id}/loop-data/"
        soup = self._get_page(loop_url)
        
        if not soup:
            return None
        
        try:
            # Find the loop data table
            loop_table = soup.find('table', class_='table')
            if not loop_table:
                logger.warning(f"No loop data table found for race {race_id}")
                return None
            
            # Parse table headers
            headers = []
            header_row = loop_table.find('thead')
            if header_row:
                for th in header_row.find_all('th'):
                    headers.append(th.get_text(strip=True))
            
            # Parse table data
            data = []
            tbody = loop_table.find('tbody')
            if tbody:
                for row in tbody.find_all('tr'):
                    row_data = []
                    for td in row.find_all('td'):
                        row_data.append(td.get_text(strip=True))
                    if row_data:
                        data.append(row_data)
            
            if not data:
                logger.warning(f"No loop data found for race {race_id}")
                return None
            
            # Create DataFrame
            df = pd.DataFrame(data, columns=headers)
            
            # Get race metadata
            race_info = self._extract_race_info(soup)
            
            # Add race metadata to each row
            for key, value in race_info.items():
                df[key] = value
            
            return df
            
        except Exception as e:
            logger.error(f"Error parsing loop data for {race_id}: {e}")
            return None
    
    def _extract_race_info(self, soup: BeautifulSoup) -> Dict:
        """Extract race metadata from the page."""
        race_info = {}
        
        try:
            # Look for race title
            title_elem = soup.find('h1') or soup.find('title')
            if title_elem:
                race_info['race_name'] = title_elem.get_text(strip=True)
            
            # Look for race date
            date_elem = soup.find(text=re.compile(r'\d{2}/\d{2}/2025'))
            if date_elem:
                race_info['date'] = date_elem.strip()
            
            # Look for track information
            track_elem = soup.find(text=re.compile(r'miles?'))
            if track_elem:
                race_info['track_info'] = track_elem.strip()
            
        except Exception as e:
            logger.warning(f"Error extracting race info: {e}")
        
        return race_info
    
    def standardize_race_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize race data to match our schema."""
        if df is None or df.empty:
            return pd.DataFrame()
        
        # Create mapping for common column names
        column_mapping = {
            'Driver': 'driver_name',
            'Car': 'car_number',
            'Start': 'start_position',
            'Finish': 'finish_position',
            'Laps': 'total_laps',
            'Laps Led': 'laps_led',
            'Status': 'status',
            'Finish Status': 'status',
            'ARP': 'avg_position',
            'High': 'best_position',
            'Low': 'worst_position',
            'Rating': 'rating',
            'Fastest': 'fastest_lap',
            'Fastest Laps': 'fastest_lap',
            'Top 15': 'top_15_laps',
            'T15 Laps': 'top_15_laps',
            '% Top 15': 'pct_top_15_laps',
            '%T15': 'pct_top_15_laps',
            '% Led': 'pct_laps_led',
            'Manu.': 'manufacturer'
        }
        
        # Rename columns
        df = df.rename(columns=column_mapping)
        
        # Extract date from race_name if it contains date info
        if 'race_name' in df.columns and 'date' not in df.columns:
            # Try to extract date from race_name
            first_race_name = df['race_name'].iloc[0] if not df['race_name'].empty else ""
            date_match = re.search(r'(\d{1,2}/\d{1,2}/2025)', first_race_name)
            if date_match:
                race_date = date_match.group(1)
                df['date'] = pd.to_datetime(race_date, format='%m/%d/%Y', errors='coerce')
            else:
                # Try to infer date from race context or use placeholder
                df['date'] = pd.to_datetime('2025-01-01')  # Placeholder, will be updated
        
        # Convert date format if it exists
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Convert numeric columns
        numeric_cols = [
            'start_position', 'finish_position', 'total_laps', 'laps_led',
            'avg_position', 'best_position', 'worst_position', 'rating',
            'fastest_lap', 'top_15_laps', 'pct_top_15_laps', 'pct_laps_led',
            'car_number'
        ]
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Add series information
        df['series'] = 'Cup'
        
        # Add track_name if available
        if 'race_name' in df.columns and 'track_name' not in df.columns:
            df['track_name'] = df['race_name']
        
        return df
    
    def get_completed_2025_races(self) -> List[Dict]:
        """Get list of completed 2025 races."""
        all_races = self.get_2025_race_ids()
        completed_races = []
        
        for race in all_races:
            # Check if race has results by trying to fetch them
            results = self.get_race_results(race['race_id'])
            if results is not None and not results.empty:
                completed_races.append(race)
        
        return completed_races
    
    def scrape_all_2025_data(self) -> pd.DataFrame:
        """Scrape all available 2025 NASCAR Cup Series data."""
        logger.info("Starting to scrape 2025 NASCAR data from LapRaptor...")
        
        # Use the known race IDs directly
        known_race_ids = [
            '5543', '5546', '5544', '5545', '5551', '5549', '5548', '5583',
            '5558', '5550', '5555', '5554', '5557', '5562', '5563', '5573',
            '5552', '5569'
        ]
        
        all_data = []
        
        for race_id in known_race_ids:
            logger.info(f"Scraping race {race_id}")
            
            # Get race results
            results = self.get_race_results(race_id)
            if results is not None and not results.empty:
                # Standardize the data
                standardized_data = self.standardize_race_data(results)
                all_data.append(standardized_data)
                
                logger.info(f"Successfully scraped {len(standardized_data)} driver results")
            else:
                logger.warning(f"No results found for race {race_id}")
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            logger.info(f"Successfully scraped {len(combined_df)} total records from {len(all_data)} races")
            return combined_df
        else:
            logger.warning("No data scraped")
            return pd.DataFrame()


def main():
    """Test the scraper."""
    scraper = LapRaptorScraper()
    
    # Test getting race IDs
    print("Getting 2025 race IDs...")
    races = scraper.get_2025_race_ids()
    print(f"Found {len(races)} races")
    
    # Test getting results for first race
    if races:
        first_race = races[0]
        print(f"\nTesting race {first_race['race_id']}: {first_race['race_name']}")
        results = scraper.get_race_results(first_race['race_id'])
        if results is not None:
            print(f"Results shape: {results.shape}")
            print(f"Columns: {results.columns.tolist()}")
        else:
            print("No results found")


if __name__ == "__main__":
    main()