from PySide6.QtCore import QThread, Signal
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class ChunkWorker(QThread):
    statusChanged = Signal(str)
    chunkCountChanged = Signal(int)
    completed = Signal(object)

    def __init__(self, files, chunk_size, chunk_overlap):
        super().__init__()
        self.files = files
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def run(self):
        try:
            docs = []
            for p in self.files:
                self.statusChanged.emit(f'Loading: {p}')
                loader = PyPDFLoader(p) if p.lower().endswith('.pdf') else TextLoader(p, encoding='utf-8')
                docs.extend(loader.load())

            self.statusChanged.emit('Chunking text...')
            splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
            chunks = splitter.split_documents(docs)
            self.chunkCountChanged.emit(len(chunks))

            self.statusChanged.emit('Building FAISS Vector Database (Local Python)...')

            # --- THE FIX: Bypassing Ollama entirely for the database ---
            from langchain_huggingface import HuggingFaceEmbeddings
            from langchain_community.vectorstores import FAISS

            # This downloads a tiny, lightning-fast math model directly into Python
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vectorstore = FAISS.from_documents(chunks, embeddings)

            self.statusChanged.emit('Ready')
            self.completed.emit(vectorstore)

        except Exception as e:
            print(f"\n--- CRITICAL WORKER CRASH ---\n{str(e)}\n-----------------------------\n")
            self.statusChanged.emit(f"Error: {str(e)}")
            self.completed.emit(None)