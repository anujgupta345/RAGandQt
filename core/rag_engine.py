from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document


class RAGEngine:
    def __init__(self):
        # Local embedding model (NO API needed)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.vector_db = None

    def build_index(self, chunks):
        """Convert chunks → FAISS index"""
        self.vector_db = FAISS.from_documents(chunks, self.embeddings)

    def retrieve(self, query, k=3):
        """Semantic search"""
        if not self.vector_db:
            return []

        docs = self.vector_db.similarity_search(query, k=k)
        return docs