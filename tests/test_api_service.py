import pytest

from services.api_service import COMPANIES, fetch_all_companies, fetch_company_data


@pytest.mark.asyncio
async def test_fetch_company_data_live_for_aapl():
    company, data = await fetch_company_data("AAPL")

    assert company == "AAPL"
    assert isinstance(data, list)

    if data:
        assert isinstance(data[0], dict)


@pytest.mark.asyncio
async def test_fetch_company_data_live_for_unknown_symbol():
    company, data = await fetch_company_data("ZZZZINVALID")

    assert company == "ZZZZINVALID"
    assert isinstance(data, list)


@pytest.mark.asyncio
async def test_fetch_all_companies_live():
    result = await fetch_all_companies()

    assert isinstance(result, dict)
    assert set(COMPANIES).issubset(set(result.keys()))

    # Each value should always be a list, even if the API has no data for a ticker.
    for value in result.values():
        assert isinstance(value, list)