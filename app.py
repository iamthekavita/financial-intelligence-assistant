##### Data → Chunks → Embeddings → FAISS → Retrieval → LLM #####

import asyncio

import streamlit as st

# ---------------- Service Layer Imports ----------------
from services.api_service import fetch_all_companies
from services.embedding_service import get_embedding
from services.llm_service import generate_answer, generate_answer_stream
from services.mcp_registry import TOOLS
from services.processing_service import (
    api_to_dataframe,
    dataframe_to_text_chunks,
    scraped_to_dataframe,
)
from services.finance_supervisor_agent import select_agents
from services.scraper_service import fetch_upstox_companies
from services.vector_store import VectorStore

# ---------------- Streamlit Setup ----------------
st.set_page_config(page_title="Financial Intelligence RAG System")
st.title("Financial Intelligence RAG System")

MAX_RETRIEVED_CONTEXT = 5


# -------------------------------------------------------
# STEP 1 → Fetch and prepare RAG data
# -------------------------------------------------------
def prepare_rag_data():
    """
    Fetch API + web data and convert into
    semantic retrieval chunks.
    """

    # ---------------------------------------------------
    # Avoid regenerating data in same session
    # ---------------------------------------------------
    if "api_chunks" in st.session_state:

        st.warning(
            "RAG data already prepared for this session."
        )

        return

    st.info("Fetching API and web data...")

    # ---------------- API DATA ----------------
    api_data = asyncio.run(
        fetch_all_companies()
    )

    api_df = api_to_dataframe(
        api_data
    )

    api_chunks = dataframe_to_text_chunks(
        api_df,
        source="api"
    )

    # ---------------- WEB DATA ----------------
    scraped_companies = (
        fetch_upstox_companies()
    )

    scraped_df = scraped_to_dataframe(
        scraped_companies
    )

    scraped_chunks = dataframe_to_text_chunks(
        scraped_df,
        source="web"
    )

    # ---------------------------------------------------
    # Save chunks in Streamlit session state
    # ---------------------------------------------------
    st.session_state.api_chunks = api_chunks

    st.session_state.scraped_chunks = scraped_chunks

    st.success(
        "RAG data prepared successfully."
    )

    # ---------------- Preview ----------------
    st.write(
        "API sample:",
        api_chunks[:5]
    )

    st.write(
        "Web sample:",
        scraped_chunks[:5]
    )

    st.write(
        f"API chunks: {len(api_chunks)}"
    )

    st.write(
        f"Web chunks: {len(scraped_chunks)}"
    )


# -------------------------------------------------------
# STEP 2 → Create Vector Database
# -------------------------------------------------------
def create_vector_db():
    """
    Convert chunks into embeddings
    and store them inside FAISS.
    """

    if "api_chunks" not in st.session_state:

        st.error(
            "Please prepare RAG data first."
        )

        return

    st.info(
        "Generating embeddings and building vector DB..."
    )

    # Limit noisy web chunks
    all_chunks = (
        st.session_state.api_chunks
        + st.session_state.scraped_chunks[:20]
    )

    # ---------------- Embedding Generation ----------------
    embeddings = [
        get_embedding(text)
        for text in all_chunks
    ]

    if not embeddings:

        st.error(
            "No embeddings generated."
        )

        return

    # ---------------- FAISS Vector Store ----------------
    vector_dim = len(embeddings[0])

    vector_store = VectorStore(vector_dim)

    vector_store.add(
        embeddings,
        all_chunks
    )

    # Save vector DB
    st.session_state.vector_store = vector_store

    st.success("Vector DB created successfully.")


# -------------------------------------------------------
# STEP 3 → MCP Routing + Tool Execution + LLM
# -------------------------------------------------------
def search_and_answer(query: str):

    if "vector_store" not in st.session_state:

        st.error(
            "Please create the vector database first."
        )

        return

    if not query.strip():

        st.warning(
            "Please enter a question."
        )

        return

    # ---------------------------------------------------
    # Finance Supervisor Agent
    # ---------------------------------------------------
    selected_agents = select_agents(query)

    st.info(
        f"Selected Agents: {', '.join(selected_agents)}"
    )

    results = []

    # ---------------------------------------------------
    # Execute all selected agents
    # ---------------------------------------------------
    for selected_agent in selected_agents:

        if selected_agent not in TOOLS:

            st.warning(
                f"Unknown agent: {selected_agent}"
            )

            continue

        tool = TOOLS[selected_agent]

        # ---------------- API Agent ----------------
        if selected_agent == "API Agent":

            api_results = asyncio.run(tool())

            for company, data in api_results.items():

                if not data:
                    continue

                item = data[0]

                results.append(
                    f"Company: {company} | "
                    f"Price: {item.get('price', 'N/A')} | "
                    f"Market Cap: {item.get('mktCap', 'N/A')}"
                )

        # ---------------- Web Agent ----------------
        elif selected_agent == "Web Agent":

            web_results = tool()

            for name, url in web_results[:5]:

                results.append(
                    f"Company: {name} | URL: {url}"
                )

        # ---------------- RAG Agent ----------------
        elif selected_agent == "RAG Agent":

            rag_results = tool(
                query,
                st.session_state.vector_store
            )

            results.extend(rag_results)

    # ---------------------------------------------------
    # Build a clean, capped context set for the final LLM
    # ---------------------------------------------------
    unique_results = []
    seen = set()

    for item in results:
        if item not in seen:
            seen.add(item)
            unique_results.append(item)

    retrieved_context = unique_results[:MAX_RETRIEVED_CONTEXT]
    context = "\n".join(retrieved_context)

    # ---------------------------------------------------
    # Final LLM Response
    # ---------------------------------------------------
    if enable_stream:

        thinking_area = st.empty()
        answer_area = st.empty()

        thinking_text = ""
        answer_text = ""

        thinking_area.markdown("**Thinking:**\n\n")

        for chunk in generate_answer_stream(
            query,
            context,
            think=True
        ):

            if chunk.get("thinking"):

                thinking_text += chunk["thinking"]

                thinking_area.text(
                    thinking_text
                )

            elif chunk.get("content"):

                answer_text += chunk["content"]

                answer_area.markdown(
                    "**Answer:**\n\n"
                    + answer_text
                )

    else:

        with st.spinner(
            "Generating response..."
        ):

            result = generate_answer(
                query,
                context,
                think=True,
            )

        if (
            enable_nonstream_think
            and isinstance(result, dict)
        ):

            st.subheader("Thinking")

            st.write(
                result.get("thinking")
            )

            st.subheader("Answer")

            st.write(
                result.get("response")
            )

        else:

            st.subheader("Answer")

            st.write(result)

    st.subheader("Retrieved Context")

    for idx, chunk in enumerate(
        retrieved_context,
        start=1
    ):

        st.write(
            f"{idx}. {chunk}"
        )
# -------------------------------------------------------
# Streamlit UI Actions
# -------------------------------------------------------

if st.button("Prepare RAG Data"):
    prepare_rag_data()

if st.button("Create Vector DB"):
    create_vector_db()

query = st.text_input("Ask your question")

# Toggle streaming thought trace
enable_stream = st.checkbox("Enable thinking stream", value=True)
# When not streaming, optionally request thinking from the model
enable_nonstream_think = st.checkbox("Enable thinking (non-stream)", value=False)

if st.button("Search"):
    # Pass the user's stream preference into the search handler
    search_and_answer(query)  