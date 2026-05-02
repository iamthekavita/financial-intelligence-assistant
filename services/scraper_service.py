import httpx
from bs4 import BeautifulSoup
from typing import List, Tuple

# Source page containing list of companies
UPSTOX_URL = "https://upstox.com/stocks-market/share-market-listed-company-in-india/"


def fetch_upstox_companies() -> List[Tuple[str, str]]:
    """Scrape the Upstox company listing page and return a list of company names and URLs."""

    # Make HTTP request to fetch page HTML
    response = httpx.get(UPSTOX_URL, timeout=15)
    response.raise_for_status()

    # Parse HTML content
    soup = BeautifulSoup(response.text, "html.parser")

    # Extract all anchor tags (<a>)
    links = soup.find_all("a")

    companies: List[Tuple[str, str]] = []
    seen_urls = set()

    for link in links:
        name = link.get_text(strip=True) # visible text
        href = link.get("href")         # link URL

        # Filter only valid company links
        if not name or not href or "/stocks/" not in href:
            continue

        # Convert relative URL → absolute URL
        full_url = f"https://upstox.com{href}"

        # Skip duplicates
        if full_url in seen_urls:
            continue

        seen_urls.add(full_url)
        companies.append((name, full_url))

    return companies