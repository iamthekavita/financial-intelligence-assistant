"""Simple MCP-style tool registry."""

from services.api_service import fetch_all_companies
from services.retrieval_agent import retrieve_context
from services.scraper_service import fetch_upstox_companies


TOOLS = {

    "API Agent": fetch_all_companies,

    "Web Agent": fetch_upstox_companies,

    "RAG Agent": retrieve_context,
}