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
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
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
            date_from: Start date in YYYY-MM-DD format
            date_to: End date in YYYY-MM-DD format
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
        
        # Add date range if provided
        if date_from and date_to:
            # Convert YYYY-MM-DD to YYYY-MM format for monthly search
            date_start = datetime.strptime(date_from, '%Y-%m-%d')
            date_end = datetime.strptime(date_to, '%Y-%m-%d')
            
            # For now, use the start month
            params['filter[date]'] = date_start.strftime('%Y-%m')
        
        base_url = 'https://www.xcontest.org/world/en/flights-search/'
        return f"{base_url}?{urlencode(params)}"
    
    async def extract_flights_from_page(self) -> List[Dict]:
        """
        Extract flight data from the current page
        
        Returns:
            List of flight dictionaries
        """
        flights = []
        
        try:
            logger.info("Waiting for flight table to load...")
            
            # First, try to find any table
            #await self.page.wait_for_load_state('domcontentloaded', timeout=10000)
            
            # Debug: Save screenshot
            await self.page.screenshot(path='xcontest_debug.png')
            logger.info("Screenshot saved to xcontest_debug.png")
            
            # Debug: Print page content structure
            tables = await self.page.query_selector_all('table')
            logger.info(f"Found {len(tables)} tables on page")
            
            # Try different possible selectors ?
            possible_selectors = [
                'table.flights'
            ]
            
            rows = []
            for selector in possible_selectors:
                try:
                    await self.page.wait_for_selector(selector, timeout=2000)
                    rows = await self.page.query_selector_all(f'{selector} tbody tr')
                    if rows:
                        logger.info(f"Found {len(rows)} rows using selector: {selector}")
                        break
                except:
                    continue
            
            if not rows:
                logger.warning("No flight rows found with any selector")
                # Print HTML for debugging
                html = await self.page.content()
                logger.debug(f"Page HTML length: {len(html)}")
                return flights
            
            for idx, row in enumerate(rows):
                try:
                    flight = {}
                    
                    # Extract all cells first
                    cells = await row.query_selector_all('td')
                    logger.info(f"Row {idx}: Found {len(cells)} cells")
                    
                    if len(cells) == 0:
                        continue
                    
                    # Debug: print cell contents for first row
                    if idx == 0:
                        for i, cell in enumerate(cells):
                            text = (await cell.inner_text()).strip()[:50]
                            logger.info(f"  Cell {i}: {text}")
                    
                    # Extract flight ID from row or links
                    flight_link = await row.query_selector('a[href*="/flights/detail"]')
                    if flight_link:
                        href = await flight_link.get_attribute('href')
                        flight['flight_id'] = href
                    
                    # Try to extract based on cell count and content
                    # This is a best-guess extraction - adjust based on debug output
                    if len(cells) >= 6:
                        flight['date'] = (await cells[0].inner_text()).strip()
                        flight['pilot'] = (await cells[1].inner_text()).strip()
                        flight['takeoff'] = (await cells[2].inner_text()).strip()
                        flight['distance_km'] = (await cells[3].inner_text()).strip()
                        flight['points'] = (await cells[4].inner_text()).strip()
                        flight['glider'] = (await cells[5].inner_text()).strip()
                    elif len(cells) >= 4:
                        # Minimal extraction if fewer columns
                        flight['date'] = (await cells[0].inner_text()).strip()
                        flight['pilot'] = (await cells[1].inner_text()).strip()
                        flight['points'] = (await cells[-1].inner_text()).strip()
                    
                    if flight:
                        flights.append(flight)
                    
                except Exception as e:
                    logger.warning(f"Failed to extract row {idx}: {e}")
                    continue
            
            logger.info(f"Extracted {len(flights)} flights from page")
            
        except Exception as e:
            logger.error(f"Failed to extract flights: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        return flights
    
    async def has_next_page(self) -> bool:
        """
        Check if there's a next page available
        
        Returns:
            True if next page exists, False otherwise
        """
        try:
            next_button = await self.page.query_selector('a.next:not(.disabled)')
            return next_button is not None
        except:
            return False
    
    async def goto_next_page(self) -> bool:
        """
        Navigate to the next page
        
        Returns:
            True if navigation successful, False otherwise
        """
        try:
            next_button = await self.page.query_selector('a.next:not(.disabled)')
            if next_button:
                await next_button.click()
                await self.page.wait_for_load_state('networkidle')
                return True
        except Exception as e:
            logger.error(f"Failed to navigate to next page: {e}")
        
        return False
    
    async def search_flights(
        self,
        lat: float,
        lon: float,
        radius_m: int = RADIUS_M_DEFAULT,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        max_pages: int = 10
    ) -> pd.DataFrame:
        """
        Search for flights and return as DataFrame
        
        Args:
            lat: Latitude of search center
            lon: Longitude of search center
            radius_m: Search radius in meters
            date_from: Start date (YYYY-MM-DD)
            date_to: End date (YYYY-MM-DD)
            max_pages: Maximum number of pages to scrape
            
        Returns:
            DataFrame with flight data
        """
        # Check if authentication is required and successful
        if self.username and self.password and not self.is_authenticated:
            logger.error("Cannot search flights - login failed or not authenticated")
            return pd.DataFrame()
        
        all_flights = []
        
        # Build and navigate to search URL
        url = self.build_search_url(lat, lon, radius_m, date_from, date_to)
        logger.info(f"Searching flights: {url}")
        
        await self.page.goto(url)
       # await self.page.wait_for_load_state('networkidle')
        
        # Extract flights from pages
        page_num = 1
        while page_num <= max_pages:
            logger.info(f"Processing page {page_num}/{max_pages}")
            
            flights = await self.extract_flights_from_page()
            all_flights.extend(flights)
            
            # Check for next page
            if not await self.has_next_page():
                logger.info("No more pages available")
                break
            
            # Navigate to next page
            if not await self.goto_next_page():
                logger.warning("Failed to navigate to next page")
                break
            
            page_num += 1
            
            # Small delay to be polite
            await asyncio.sleep(1)
        
        logger.info(f"Total flights extracted: {len(all_flights)}")
        
        return pd.DataFrame(all_flights)


async def load_xcontest_flights(
    lat: float,
    lon: float,
    radius_m: int = RADIUS_M_DEFAULT,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    max_pages: int = 10
) -> pd.DataFrame:
    """
    Convenience function to load XContest flights
    
    Args:
        lat: Latitude of search center
        lon: Longitude of search center  
        radius_m: Search radius in meters (default 250m)
        date_from: Start date (YYYY-MM-DD)
        date_to: End date (YYYY-MM-DD)
        username: XContest username for authentication
        password: XContest password
        max_pages: Maximum pages to scrape
        
    Returns:
        DataFrame with flight data
    """
    async with XContestLoader(username, password) as loader:
        return await loader.search_flights(
            lat, lon, radius_m, date_from, date_to, max_pages
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
            date_from='2022-06-01',
            date_to='2022-06-10',
            username=username,
            password=password,
            max_pages=3
        )
    )
    
    print(f"\nFound {len(flights_df)} flights")
    print(flights_df.head(10))
    
    # Show summary
    if not flights_df.empty:
        print(f"\nDate range: {flights_df['date'].min()} to {flights_df['date'].max()}")
        print(f"\nUnique pilots: {flights_df['pilot'].nunique()}")