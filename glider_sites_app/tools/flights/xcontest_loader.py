# xcontest_loader.py

import asyncio
import logging
from typing import Optional, List, Dict
from urllib.parse import urlencode
import pandas as pd
from playwright.async_api import async_playwright, Page, Browser
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


RADIUS_M_DEFAULT = 250

class XContestLoader:
    """
    XContest flight data loader using Playwright for authenticated access
    """
    
    def __init__(self, username: Optional[str] = None, password: Optional[str] = None):
        """
        Initialize XContest loader
        
        Args:
            username: XContest login username (optional, for authenticated access)
            password: XContest login password (optional)
        """
        self.username = username
        self.password = password
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None
        self.playwright = None
        self.is_authenticated = False
    
    async def __aenter__(self):
        """Context manager entry"""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        await self.close()
    
    async def start(self):
        """Start the browser and create a page"""
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=True)
        self.page = await self.browser.new_page()
        
        # Set a reasonable user agent
        await self.page.set_extra_http_headers({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Login if credentials provided
        if self.username and self.password:
            await self.login()
    
    async def close(self):
        """Close the browser"""
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
    
    async def login(self):
        """Login to XContest"""
        try:
            logger.info("Logging in to XContest...")
            await self.page.goto('https://www.xcontest.org/world/en/')
            
            logger.info("Waiting for login form to appear...")
            await self.page.wait_for_selector('form#login', timeout=5000)
            
            logger.info("Filling in login credentials...")
            await self.page.fill('input#login-username', self.username)
            await self.page.fill('input#login-password', self.password)
            
            logger.info("Submitting login form...")
            
            # Submit form
            await self.page.click('input.submit[type="submit"]')
            
            # Wait for the response
            await asyncio.sleep(2)
            await self.page.wait_for_load_state('networkidle', timeout=10000)
            
            # Check for error dialog/message first
            error_dialog = await self.page.query_selector('#login-error-dialog[open]')
            if error_dialog:
                error_text = await error_dialog.inner_text()
                logger.error(f"Login error dialog: {error_text}")
                await self.page.screenshot(path='xcontest_login_failed.png')
                self.is_authenticated = False
                return
            
            await self.page.screenshot(path='xcontest_after_login.png')
            logger.info("Post-login screenshot saved to xcontest_after_login.png")
            
            self.is_authenticated = True
            logger.info("Successfully logged in to XContest")
            
        except Exception as e:
            logger.error(f"Login failed with exception: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.is_authenticated = False
    
    def build_search_url(
        self,
        lat: float,
        lon: float,
        radius_m: int = RADIUS_M_DEFAULT,
        date_filter: Optional[str] = None,
        mode: str = 'START',
        sort: str = 'pts',
        sort_dir: str = 'down'
    ) -> str:
        """
        Build XContest search URL
        
        Args:
            lat: Latitude of search center
            lon: Longitude of search center
            radius_m: Search radius in meters (default RADIUS_M_DEFAULTm)
            date_filter: Date filter in YYYY-MM-DD (specific day), YYYY-MM (month), or YYYY (year) format
            mode: Search mode (START, LINE, etc.)
            sort: Sort field (pts, date, dst)
            sort_dir: Sort direction (up, down)
            
        Returns:
            Full search URL
        """
        params = {
            'filter[point]': f"{lon} {lat}",
            'filter[radius]': str(radius_m),
            'filter[mode]': mode,
            'filter[date_mode]': 'dmy',
            'filter[value_mode]': 'dst',
            'list[sort]': sort,
            'list[dir]': sort_dir
        }
        
        # Add date filter if provided (can be YYYY, YYYY-MM, or YYYY-MM-DD)
        if date_filter:
            params['filter[date]'] = date_filter
        
        base_url = 'https://www.xcontest.org/world/en/flights-search/'
        return f"{base_url}?{urlencode(params)}"
    
    async def extract_flights_from_page(self) -> List[Dict]:
        """
        Extract flight data from the current page
        
        Returns:
            List of flight dictionaries with fields:
            - flight_id: XContest flight ID (from row id attribute)
            - flight_url: URL to flight detail page
            - date: Flight date (DD.MM.YY format)
            - time: Flight time (HH:MM format)
            - utc_offset: UTC offset (e.g., UTC+02:00)
            - pilot_name: Pilot's name
            - pilot_country: Pilot's country code (e.g., DE)
            - takeoff_name: Takeoff location name
            - takeoff_country: Takeoff country code
            - takeoff_registered: Whether takeoff is registered (✔ if yes)
            - flight_type: Flight type (e.g., free flight)
            - distance_km: Distance in km
            - points: XContest points
            - glider_class: Glider class (A, B, C, D, etc.)
            - glider_name: Glider model name
        """
        flights = []
        
        try:
            logger.info("Waiting for flight table to load...")
            
            # Wait for table to be present
            await self.page.wait_for_selector('table.flights', timeout=5000)
            
            # Get all flight rows
            rows = await self.page.query_selector_all('table.flights tbody tr')
            logger.info(f"Found {len(rows)} flight rows")
            
            if not rows:
                logger.warning("No flight rows found")
                return flights
            
            for idx, row in enumerate(rows):
                try:
                    flight = {}
                    
                    # Extract flight ID from row id attribute (e.g., "flight-3161854" -> "3161854")
                    row_id = await row.get_attribute('id')
                    if row_id and row_id.startswith('flight-'):
                        flight['flight_id'] = row_id.replace('flight-', '')
                    
                    # Get all cells
                    cells = await row.query_selector_all('td')
                    
                    if len(cells) < 8:
                        logger.warning(f"Row {idx}: Expected at least 8 cells, got {len(cells)}")
                        continue
                    
                    # Cell 0: Row number (skip)
                    
                    # Cell 1: Date and time
                    date_cell = cells[1]
                    date_full_div = await date_cell.query_selector('div.full')
                    if date_full_div:
                        date_text = await date_full_div.inner_text()
                        # Parse "02.06.22 15:31UTC+02:00" format
                        lines = date_text.strip().split('\n')
                        if lines:
                            # First line has date and time
                            date_time_parts = lines[0].strip().split()
                            if len(date_time_parts) >= 2:
                                # Convert from DD.MM.YY to YYYY-MM-DD
                                date_str = date_time_parts[0]  # e.g., "02.06.22"
                                try:
                                    parsed_date = datetime.strptime(date_str, '%d.%m.%y')
                                    flight['date'] = parsed_date.strftime('%Y-%m-%d')
                                except ValueError:
                                    flight['date'] = date_str  # Keep original if parsing fails
                                flight['time'] = date_time_parts[1]  # e.g., "15:31"
                            # UTC offset might be in second line or in first
                            #utc_span = await date_full_div.query_selector('span.XCutcOffset')
                            #if utc_span:
                            #    flight['utc_offset'] = await utc_span.inner_text()
                    
                    # Cell 2: Pilot (country + name)
                    pilot_cell = cells[2]
                    pilot_full_div = await pilot_cell.query_selector('div.full')
                    if pilot_full_div:
                        # Country flag
                        #country_span = await pilot_full_div.query_selector('span.cic')
                        #if country_span:
                        #    flight['pilot_country'] = await country_span.get_attribute('title')
                        # Pilot name and ID
                        pilot_link = await pilot_full_div.query_selector('a.plt')
                        if pilot_link:
                            flight['pilot_name'] = await pilot_link.inner_text()
                            # Extract pilot ID from href (e.g., "/world/en/pilots/detail:pogacsa" -> "pogacsa")
                            pilot_href = await pilot_link.get_attribute('href')
                            if pilot_href and 'detail:' in pilot_href:
                                flight['pilot_id'] = pilot_href.split('detail:')[1].split('/')[0]
                    
                    # Cell 3: Takeoff (country + location + registered marker)
                    takeoff_cell = cells[3]
                    takeoff_full_div = await takeoff_cell.query_selector('div.full')
                    if takeoff_full_div:
                        # Country flag
                        #country_span = await takeoff_full_div.query_selector('span.cic')
                        #if country_span:
                        #    flight['takeoff_country'] = await country_span.get_attribute('title')
                        # Takeoff name
                        takeoff_link = await takeoff_full_div.query_selector('a.lau')
                        if takeoff_link:
                            flight['takeoff_name'] = await takeoff_link.inner_text()
                        # Registered marker (✔)
                        #registered_span = await takeoff_full_div.query_selector('span.lau[style*="green"]')
                        #if registered_span:
                        #    flight['takeoff_registered'] = await registered_span.inner_text()
                    
                    # Cell 4: Flight type (check for any disc-* div and get title attribute)
                    type_cell = cells[4]
                    # Look for any div with class starting with "disc-"
                    type_div = await type_cell.query_selector('div[class*="disc-"]')
                    if type_div:
                        flight['flight_type'] = await type_div.get_attribute('title')
                    
                    # Cell 5: Distance (km)
                    distance_cell = cells[5]
                    distance_strong = await distance_cell.query_selector('strong')
                    if distance_strong:
                        distance_text = await distance_strong.inner_text()
                        flight['distance_km'] = distance_text.replace(' km', '').strip()
                    
                    # Cell 6: Points
                    points_cell = cells[6]
                    points_strong = await points_cell.query_selector('strong')
                    if points_strong:
                        points_text = await points_strong.inner_text()
                        flight['points'] = points_text.replace(' p.', '').strip()
                    
                    # Cell 7: Glider class and name (e.g., "cat-A" class, "PHI Beat Light" in title)
                    glider_cell = cells[7]
                    # Extract class from cell class attribute (e.g., "cat-A")
                    glider_class_attr = await glider_cell.get_attribute('class')
                    if glider_class_attr:
                        for cls in glider_class_attr.split():
                            if cls.startswith('cat-'):
                                # xcontest category is not entirely same as EN class, but we use it here
                                flight['glider_category'] = cls.replace('cat-', '')
                    # Glider name from div title
                    glider_div = await glider_cell.query_selector('div')
                    if glider_div:
                        glider_name = await glider_div.get_attribute('title')
                        if glider_name:
                            flight['glider_name'] = glider_name
                    
                    # Cell 9 (or last): Flight detail link
                    detail_link = await row.query_selector('a.detail')
                    if detail_link:
                        href = await detail_link.get_attribute('href')
                        if href:
                            flight['flight_details'] = href
                    
                    if flight:
                        flights.append(flight)
                    
                    # Log first flight for debugging
                    if idx == 0:
                        logger.debug(f"First flight sample: {flight}")
                    
                except Exception as e:
                    logger.warning(f"Failed to extract row {idx}: {e}")
                    import traceback
                    logger.debug(traceback.format_exc())
                    continue
            
            logger.info(f"Extracted {len(flights)} flights from page")
            
        except Exception as e:
            logger.error(f"Failed to extract flights: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Save debug screenshot
            await self.page.screenshot(path='xcontest_extraction_error.png')
            logger.info("Error screenshot saved to xcontest_extraction_error.png")
        
        return flights
    
    async def extract_available_flight_dates(self, date_from: Optional[str] = None) -> List[str]:
        """
        Extract available flight dates from the date filter dropdown
        
        Args:
            date_from: Only return dates > this date (YYYY-MM-DD format)
            
        Returns:
            List of flight dates in YYYY-MM-DD format, sorted ascending
        """
        try:
            # Get all options from the date filter dropdown
            date_options = await self.page.query_selector_all('select#filter-date option')
            
            flight_dates = []
            for option in date_options:
                value = await option.get_attribute('value')
                # Only include options with full dates (YYYY-MM-DD format)
                # Skip empty, year-only (YYYY), or month-only (YYYY-MM) options
                if value and len(value) == 10 and value.count('-') == 2:
                    # Filter by date_from if provided
                    if date_from is None or value > date_from:
                        flight_dates.append(value)
            
            # Sort dates ascending (oldest first)
            flight_dates.sort()
            
            logger.info(f"Found {len(flight_dates)} available flight dates")
            if flight_dates:
                logger.info(f"Date range: {flight_dates[0]} to {flight_dates[-1]}")
            
            return flight_dates
            
        except Exception as e:
            logger.error(f"Failed to extract flight dates: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    async def search_flights(
        self,
        lat: float,
        lon: float,
        radius_m: int = RADIUS_M_DEFAULT,
        date_from: Optional[str] = None,
        get_next_date_only: bool = True
    ) -> pd.DataFrame:
        """
        Search for flights and return as DataFrame
        
        Args:
            lat: Latitude of search center
            lon: Longitude of search center
            radius_m: Search radius in meters
            date_from: Start date (YYYY-MM-DD) - only dates > this will be fetched
            get_next_date_only: If True, only fetch the next available flight date (default: True)
                               If False, fetch all available dates from date_from onwards
            
        Returns:
            DataFrame with flight data
        """
        # Check if authentication is required and successful
        if self.username and self.password and not self.is_authenticated:
            logger.error("Cannot search flights - login failed or not authenticated")
            return pd.DataFrame()
        
        all_flights = []
        
        # First, navigate to search page WITHOUT date filter to load ALL available dates for this location
        # The dropdown is populated based on the location, not the date filter
        url = self.build_search_url(lat, lon, radius_m, date_filter=None)
        logger.info(f"Loading search page to get available dates: {url}")
        await self.page.goto(url)
        await asyncio.sleep(1)  # Wait for dropdown to populate with location-specific dates
        
        # Extract available flight dates
        flight_dates = await self.extract_available_flight_dates(date_from)
        
        if not flight_dates:
            logger.warning(f"No flight dates found from {date_from}")
            return pd.DataFrame()
        
        # Determine which dates to process
        dates_to_fetch = [flight_dates[0]] if get_next_date_only else flight_dates[:10]
        
        logger.info(f"Will fetch {len(dates_to_fetch)} date(s): {dates_to_fetch}")
        
        # Fetch flights for each date
        for flight_date in dates_to_fetch:
            logger.info(f"Fetching flights for {flight_date}...")
            
            # Build URL for specific date
            url = self.build_search_url(lat, lon, radius_m, flight_date)
            await self.page.goto(url)
            await asyncio.sleep(1)  # Be polite to server
            
            # Extract flights from this date
            flights = await self.extract_flights_from_page()
            logger.info(f"  Found {len(flights)} flights on {flight_date}")
            all_flights.extend(flights)
        
        logger.info(f"Total flights extracted: {len(all_flights)}")
        
        return pd.DataFrame(all_flights)


async def load_xcontest_flights(
    lat: float,
    lon: float,
    radius_m: int = RADIUS_M_DEFAULT,
    date_from: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    get_next_date_only: bool = True
) -> pd.DataFrame:
    """
    Convenience function to load XContest flights
    
    Args:
        lat: Latitude of search center
        lon: Longitude of search center  
        radius_m: Search radius in meters (default 250m)
        date_from: Start date (YYYY-MM-DD) - only dates > this will be fetched
        username: XContest username for authentication
        password: XContest password
        get_next_date_only: If True, only fetch the next available flight date (default: True)
                           Allows flights_service to incrementally load data
        
    Returns:
        DataFrame with flight data
    """
    async with XContestLoader(username, password) as loader:
        return await loader.search_flights(
            lat, lon, radius_m, date_from, get_next_date_only
        )


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Example: Rammelsberg NW
    lat = 51.888626
    lon = 10.430664
    
    # Get credentials from environment
    username = os.getenv('XCONTEST_USERNAME')
    password = os.getenv('XCONTEST_PASSWORD')
    
    # Load flights
    flights_df = asyncio.run(
        load_xcontest_flights(
            lat=lat,
            lon=lon,
            radius_m=RADIUS_M_DEFAULT,
            date_from='2022-04-30',  # Will get next available flight date > this
            username=username,
            password=password,
            get_next_date_only=True  # Only get next date (flights_service handles incremental loading)
        )
    )
    
    print(f"\nFound {len(flights_df)} flights")
    print(flights_df.head())
    
    # Show summary
    if not flights_df.empty:
        print(f"\nDate range: {flights_df['date'].min()} to {flights_df['date'].max()}")
        print(f"\nUnique pilots: {flights_df['pilot_name'].nunique()}")