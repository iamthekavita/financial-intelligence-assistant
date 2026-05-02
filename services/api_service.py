import asyncio
from typing import Any, Dict, List, Tuple

import httpx

# API configuration (Financial Modeling Prep - stable endpoints)
API_KEY = "YX82fp16OttIOtMaO08PvxLMi9Pu7aN9"
BASE_URL = "https://financialmodelingprep.com/stable"

# List of company tickers to fetch
COMPANIES = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]


async def fetch_company_data(company: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Fetch profile data for a single company ticker from Financial Modeling Prep."""
    # Build request URL with query params
    url = f"{BASE_URL}/profile?symbol={company}&apikey={API_KEY}"

    # Async HTTP client for non-blocking requests
    async with httpx.AsyncClient() as client:
        response = await client.get(url, timeout=15)

        # Basic error handling for failed requests
        if response.status_code != 200:
            return company, []

        # Parse JSON safely
        try:
            data = response.json()
        except ValueError:
            data = []

        # Return tuple → (ticker, data)
        return company, data


async def fetch_all_companies() -> Dict[str, List[Dict[str, Any]]]:
    """Fetch company profiles for all configured tickers in parallel."""
    # Create async tasks for each company
    tasks = [fetch_company_data(company) for company in COMPANIES]

    # Run all tasks in parallel
    results = await asyncio.gather(*tasks)

    # Convert list of tuples → dictionary {ticker: data}
    return dict(results)