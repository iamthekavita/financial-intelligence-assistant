##### Data → Chunks → Embeddings → FAISS → Retrieval → LLM #####
import streamlit as st
import asyncio

# Service layer imports (each handles a specific responsibility)
from services.api_service import fetch_all_companies
from services.embedding_service import get_embedding
from services.llm_service import generate_answer
from services.processing_service import (
    api_to_dataframe,
    dataframe_to_text_chunks,
    scraped_to_dataframe,
)
from services.scraper_service import fetch_upstox_companies
from services.vector_store import VectorStore
from services.router_agent import route_query
from services.retrieval_agent import retrieve_context


# Streamlit page setup
st.set_page_config(page_title="Financial Intelligence RAG System")
st.title("Financial Intelligence RAG System")


def prepare_rag_data():
    """Fetch API and web data, convert it into retrieval chunks, and store them in the Streamlit session."""
    st.info("Fetching data from the financial API and web scraping source...")

    # -------- API DATA --------
    api_data = asyncio.run(fetch_all_companies()) # async API call
    api_df = api_to_dataframe(api_data)           # structured format
    api_chunks = dataframe_to_text_chunks(api_df, source="api")  # RAG-ready text

    # -------- WEB SCRAPED DATA --------
    scraped_companies = fetch_upstox_companies()
    scraped_df = scraped_to_dataframe(scraped_companies)
    scraped_chunks = dataframe_to_text_chunks(scraped_df, source="web")

    # Store chunks for later steps (vector DB + search)
    st.session_state.api_chunks = api_chunks
    st.session_state.scraped_chunks = scraped_chunks

    st.success("Data prepared successfully.")
    st.write("API sample:", api_chunks[:5])
    st.write("Web sample:", scraped_chunks[:5])
    st.write(f"API chunks: {len(api_chunks)}")
    st.write(f"Web chunks: {len(scraped_chunks)}")


def create_vector_db():
    """Generate embeddings for prepared chunks and build a FAISS similarity index."""
    if "api_chunks" not in st.session_state:
        st.error("Please prepare the RAG data before creating the vector database.")
        return

    st.info("Generating embeddings and building the vector database...")

    # Combine both data sources (limit web data to avoid too many embeddings)
    all_chunks = st.session_state.api_chunks + st.session_state.scraped_chunks[:20]

    # Convert each chunk into vector representation
    embeddings = [get_embedding(text) for text in all_chunks]

    if not embeddings:
        st.error("No text chunks available to index.")
        return

    # Initialize FAISS index
    vector_dim = len(embeddings[0])
    vector_store = VectorStore(vector_dim)
    vector_store.add(embeddings, all_chunks)

    # Save vector DB in session
    st.session_state.vector_store = vector_store
    st.success("Vector DB created successfully.")

def search_and_answer(query: str):

    if "vector_store" not in st.session_state:
        st.error("Please create the vector database before searching.")
        return

    if not query.strip():
        st.warning("Enter a question before clicking Search.")
        return

    # -------------------------------
    # Router Agent decides workflow
    # -------------------------------
    selected_agent = route_query(query)

    # Show selected agent in UI
    st.info(f"Selected Agent: {selected_agent}")

    # -------------------------------
    # Retrieval Agent performs search
    # -------------------------------
    results = retrieve_context(
        query,
        st.session_state.vector_store
    )

    # Build context for LLM
    context = "\n".join(results)

    # Generate final response
    with st.spinner("Searching and generating answer..."):

        answer = generate_answer(
            query,
            context
        )

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Retrieved Context")

    for idx, chunk in enumerate(results, start=1):
        st.write(f"{idx}. {chunk}")
# ---------------- User Actions ----------------
if st.button("Prepare RAG Data"):
    prepare_rag_data()

if st.button("Create Vector DB"):
    create_vector_db()

query = st.text_input("Ask your question")
if st.button("Search"):
    search_and_answer(query)