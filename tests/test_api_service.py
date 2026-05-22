import pytest
from unittest.mock import Mock, patch

from services.api_service import (
    fetch_company_data,
    fetch_all_companies
)


# -------------------------------
# Test: Successful API response
# -------------------------------
@pytest.mark.asyncio
async def test_fetch_company_data_success():

    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = [
        {
            "symbol": "AAPL",
            "price": 271.35
        }
    ]

    with patch("httpx.AsyncClient.get", return_value=mock_response):

        company, data = await fetch_company_data("AAPL")

        assert company == "AAPL"
        assert isinstance(data, list)
        assert data[0]["symbol"] == "AAPL"


# -------------------------------
# Test: API failure response
# -------------------------------
@pytest.mark.asyncio
async def test_fetch_company_data_failure():

    mock_response = Mock()
    mock_response.status_code = 500

    with patch("httpx.AsyncClient.get", return_value=mock_response):

        company, data = await fetch_company_data("AAPL")

        assert company == "AAPL"
        assert data == []


# -------------------------------
# Test: Invalid JSON response
# -------------------------------
@pytest.mark.asyncio
async def test_fetch_company_data_invalid_json():

    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.side_effect = ValueError

    with patch("httpx.AsyncClient.get", return_value=mock_response):

        company, data = await fetch_company_data("AAPL")

        assert company == "AAPL"
        assert data == []


# -------------------------------
# Test: Fetch all companies
# -------------------------------
@pytest.mark.asyncio
async def test_fetch_all_companies():

    mock_result = (
        "AAPL",
        [{"symbol": "AAPL"}]
    )

    with patch(
        "services.api_service.fetch_company_data",
        Mock(return_value=mock_result)
    ):

        result = await fetch_all_companies()

        assert isinstance(result, dict)
        assert "AAPL" in result
        assert result["AAPL"][0]["symbol"] == "AAPL"


# -------------------------------
# Test: Empty response
# -------------------------------
@pytest.mark.asyncio
async def test_fetch_company_data_empty_response():

    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = []

    with patch("httpx.AsyncClient.get", return_value=mock_response):

        company, data = await fetch_company_data("AAPL")

        assert company == "AAPL"
        assert data == []