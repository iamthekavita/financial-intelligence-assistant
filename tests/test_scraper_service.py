from services.scraper_service import fetch_upstox_companies


def test_fetch_upstox_companies_live_returns_company_links():
    companies = fetch_upstox_companies()

    assert isinstance(companies, list)
    assert len(companies) > 0

    name, url = companies[0]
    assert isinstance(name, str)
    assert isinstance(url, str)
    assert name.strip() != ""
    assert url.startswith("https://upstox.com")
