# AI Assistant - Versatile Pro RAG

A powerful, fully local, and cloud-capable Retrieval-Augmented Generation (RAG) application built with Python and PySide6. This desktop application allows users to upload multiple documents, build a persistent vector database, and query the text using top-tier LLMs like Llama 3.1, Google Gemini, or OpenAI's GPT-4o.

## ✨ Features

* **Multi-Model Routing:** Seamlessly switch between local models (via Ollama) and cloud models (Google Gemini, OpenAI) directly from the UI.
* **Persistent Vector Database:** Uses FAISS and HuggingFace Embeddings (`all-MiniLM-L6-v2`) to build a lightning-fast mathematical database that saves to your hard drive for instant loading.
* **Smart Security:** Never hardcode API keys. The app dynamically prompts for cloud API keys only when necessary and stores them securely for the session.
* **Multi-Threaded GUI:** Built with PySide6 (`QThread`), ensuring the UI remains perfectly smooth and responsive while processing massive document batches in the background.
* **Precision Control:** Adjustable UI temperature dial to switch between strict, to-the-point text extraction and creative synthesis.

## 🛠️ Prerequisites

Before you begin, ensure you have the following installed on your machine:
* **Python 3.9+**
* **Ollama** (for running local models)

### Required Local Models
If you plan to run queries entirely offline, you must pull a local model via Ollama. Open your terminal and run:
```bash
ollama pull llama3.1

🚀 Installation & Setup
1. Clone the repository:

Bash
git clone [https://github.com/yourusername/your-repo-name.git](https://github.com/yourusername/your-repo-name.git)
cd your-repo-name
2. Create and activate a virtual environment (Recommended):

Bash
# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
3. Install dependencies:

Bash
pip install -r requirements.txt
(Note: This includes LangChain, PySide6, FAISS-cpu, HuggingFace tools, and PyPDF).

💻 Usage
Start the application by running the main entry point:

Bash
python main.py
Workflow:

Upload Documents: Click "Upload Files" (hold Cmd/Ctrl to select multiple PDFs or TXT files).

Process or Load: Wait for the chunks to generate, or click "Load DB from Disk" if you have previously saved a database.

Select a Model: Type your preferred model into the text box (e.g., llama3.1, gemini-1.5-flash, gpt-4o).

Query: Type your question and click "Generate".

⚠️ Troubleshooting & Notes
HuggingFace Warning: On your first run, you may see a terminal warning about unauthenticated requests to the HF Hub. This is completely normal and harmless. The app is simply downloading the 90MB embedding model, which will be cached permanently on your machine.

API Keys: If you request a cloud model, a secure PySide6 dialog will prompt you for the necessary API key.
