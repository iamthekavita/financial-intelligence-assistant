from typing import Dict, List, Tuple

import pandas as pd


def api_to_dataframe(raw_data: Dict[str, List[dict]]) -> pd.DataFrame:
    """Convert raw API profile data into a pandas DataFrame for retrieval chunk creation."""
    records: List[Dict[str, object]] = []

    for company, data in raw_data.items():
        # Skip if no data or unexpected format
        if not data or not isinstance(data, list):
            continue

        # API returns list → take first item as primary record
        item = data[0]

        # API returns list → take first item as primary record
        records.append(
            {
                "company": company,
                "price": item.get("price", 0),
                "marketCap": item.get("mktCap", 0),
            }
        )

    # Convert list of dicts → DataFrame
    return pd.DataFrame(records)


def scraped_to_dataframe(companies: List[Tuple[str, str]]) -> pd.DataFrame:
    """Convert scraped company tuples into a DataFrame."""
    return pd.DataFrame(companies, columns=["company", "url"])


def dataframe_to_text_chunks(df: pd.DataFrame, source: str = "api") -> List[str]:
    """Create human-readable text chunks from a DataFrame for semantic retrieval."""
    texts: List[str] = []

    for _, row in df.iterrows():
        # Format depends on source (API vs Web)
        if source == "api":
            text = (
                f"Company: {row['company']} | "
                f"Price: {row['price']} | "
                f"MarketCap: {row['marketCap']}"
            )
        else:
            text = f"Company: {row['company']} | Stock Page: {row['url']}"

        texts.append(text)

    return texts