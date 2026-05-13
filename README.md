# Financial Intelligence Assistant
A simple RAG-based application built with Streamlit to explore financial data. It pulls company info from an API, scrapes stock pages, converts everything into embeddings, and uses FAISS for semantic search. A local LLM is used to generate answers from retrieved context.

## Project Overview
- Fetches company data from a financial API
- Scrapes company stock pages from a web source
- Converts API and scraped data into retrieval-ready text chunks
- Generates embeddings using sentence-transformers
- Stores and searches data in FAISS
- Streams answers from a local LLM endpoint

## Folder Structure
- app.py – Main app (UI + flow control)
- services/ – Backend logic
- services/api_service.py – API calls
- services/scraper_service.py – Web scraping
- services/processing_service.py – Data → text chunks
- services/embedding_service.py – Embeddings
- services/vector_store.py – FAISS index
- services/llm_service.py – Prompt building + LLM calls

## Setup
### Create a virtual environment:
python -m venv .venv
.\.venv\Scripts\activate      # Windows
# source .venv/bin/activate   # Mac/Linux

### Install dependencies:
pip install streamlit pandas httpx beautifulsoup4 sentence-transformers faiss-cpu requests

## Run the App
- Make sure your local LLM server is running
- Start the app:

streamlit run app.py

## How to Use
- Click `Prepare RAG Data` to fetch API and web data
- Click `Create Vector DB` to generate embeddings and build the FAISS index
- Enter a question and click `Search`
- The app streams the LLM answer and shows retrieved context

## Notes
- The app uses the endpoint: `http://localhost:11434/api/generate`
- The default model is `llama3.1:latest`
- Embeddings are normalized and indexed with FAISS `IndexFlatIP`
- Search uses the top relevant chunks from the vector store

## Troubleshooting

### If the LLM endpoint returns 404
- Verify Ollama (or your local model server) is running
- Confirm the endpoint URL is `http://localhost:11434/api/generate`
- Ensure the model is available locally and is accepting requests

### If search returns no answer
- Confirm `Prepare RAG Data` completed successfully
- Confirm `Create Vector DB` completed successfully
- Check that the vector store is available in the Streamlit session

# Financial Intelligence RAG System
- FAISS vector DB
- Streamlit UI
- Local LLM streaming
- API + web scraping

## Run
streamlit run app.py