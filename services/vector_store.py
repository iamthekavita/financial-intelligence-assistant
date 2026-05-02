"""A simple FAISS-backed vector store for nearest neighbor search."""

from typing import List

import faiss
import numpy as np


class VectorStore:
    def __init__(self, dim: int):
        """Initialize the vector store with an inner-product index."""
         # Using IndexFlatIP (Inner Product) → works with cosine similarity when vectors are normalized
        self.index = faiss.IndexFlatIP(dim)  #getting saved in RAM
        self.texts: List[str] = []

    def add(self, embeddings: List[List[float]], texts: List[str]) -> None:
        """Normalize embeddings and add them to the FAISS index."""
        # Convert list of embeddings into numpy array (required by FAISS)
        vectors = np.array(embeddings, dtype="float32")

        # Normalize vectors → enables cosine similarity using inner product
        faiss.normalize_L2(vectors)

        # Add vectors to FAISS index
        self.index.add(vectors)

        # Keep corresponding text for retrieval
        self.texts.extend(texts)

    def search(self, query_embedding: List[float], k: int = 5) -> List[str]:
        """Search the index for the top-k most similar text chunks."""
        # Convert query into numpy array and normalize (same as stored vectors)
        query_vector = np.array([query_embedding], dtype="float32")
        faiss.normalize_L2(query_vector)

        # Perform similarity search → returns distances and indices
        _, indices = self.index.search(query_vector, k)

        # Map indices back to original text chunks
        return [self.texts[i] for i in indices[0] if i != -1]
    

    # Normalize
#     🎯 Simple analogy

# Socho:

# Ek banda zor se bol raha hai 📢
# Ek banda dheere bol raha hai 🤏

# 👉 Dono same baat bol rahe hain

# Normalization kya karta hai?

# 👉 Volume hata deta hai
# 👉 Sirf baat (meaning) rakhta hai

# 🧠 RAG mein kyu zaroori hai?

# Tumhare system mein:

# "high price" ≈ "expensive stock"

# 👉 Words alag
# 👉 Meaning same

# Normalization help karta hai:

# Meaning match karo, size nahi

# 🔥 Ek line mein

# 👉 normalize_L2 vector ka size hata ke sirf uska meaning (direction) compare karne deta hai