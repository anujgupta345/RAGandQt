from PySide6.QtCore import QThread, Signal


class GenerateWorker(QThread):
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, query, vectorstore, model_name, temperature):
        super().__init__()
        self.query = query
        self.vectorstore = vectorstore
        self.model_name = model_name
        self.temperature = temperature

    def run(self):
        try:
            from langchain_classic.chains.combine_documents import create_stuff_documents_chain
            from langchain_core.prompts import ChatPromptTemplate

            model_lower = self.model_name.lower()

            # Route client based on model name
            if "gemini" in model_lower:
                from langchain_google_genai import ChatGoogleGenerativeAI
                llm = ChatGoogleGenerativeAI(model=self.model_name, temperature=self.temperature)
            elif "gpt" in model_lower or "o1" in model_lower:
                from langchain_openai import ChatOpenAI
                llm = ChatOpenAI(model=self.model_name, temperature=self.temperature)
            else:
                from langchain_ollama import ChatOllama
                llm = ChatOllama(model=self.model_name, temperature=self.temperature)

            # Safety Gate 1: If no database exists at all
            if self.vectorstore is None:
                self.finished.emit("I am sorry, but no documents have been uploaded to search from.")
                return

            # Fetch the top 5 chunks without filtering by HuggingFace's math scores
            relevant_docs = self.vectorstore.similarity_search(self.query, k=20)

            # --- DEBUGGING: PRINT CHUNKS TO PYCHARM CONSOLE ---
            print("\n" + "=" * 40)
            print(f"QUESTION: {self.query}")
            print("--- WHAT THE AI IS READING: ---")
            for i, doc in enumerate(relevant_docs):
                print(f"CHUNK {i + 1}: {doc.page_content}\n")
            print("=" * 40 + "\n")
            # --------------------------------------------------
            # Safety Gate 2: If the database is somehow totally empty
            if not relevant_docs:
                self.finished.emit(
                    "I am sorry, but the provided documents do not contain any information regarding this question.")
                return

            # ADVERSARIAL PROMPT DESIGN FOR LOCAL MODELS
            prompt = ChatPromptTemplate.from_messages([
                ("system", (
                    "SYSTEM OPERATING DIRECTIVE: You are a strict, locked-down text extraction tool.\n"
                    "Your ONLY source of truth is the text inside the <context></context> tags below.\n\n"
                    "CRITICAL CONSTRAINTS:\n"
                    "1. Scan the <context> tags for facts that answer the user's input.\n"
                    "2. You are allowed to synthesize, count, or summarize information if it is present across multiple chunks (e.g., counting items in a list).\n"
                    "3. If the context does not explicitly mention the topic or provide enough info to answer, you MUST say exactly: "
                    "'I am sorry, but the provided documents do not contain information regarding this question.'\n"
                    "4. DO NOT use your own background weights, training data, coding knowledge, or general memory.\n\n"
                    "--- SAFE SOURCE CONTEXT ---\n"
                    "<context>\n{context}\n</context>"
                )),
                ("human", (
                    "Remember: If the answer cannot be logically derived from the <context> tags above, output the refusal sentence exactly.\n\n"
                    "Question: {input}"
                ))
            ])

            chain = create_stuff_documents_chain(llm, prompt)
            result = chain.invoke({"input": self.query, "context": relevant_docs})

            self.finished.emit(result)

        except ImportError as e:
            self.error.emit(f"Missing package. Please run: pip install {str(e).split()[-1].replace('_', '-')}")
        except Exception as e:
            if "Connection error" in str(e) or "Connection refused" in str(e):
                self.error.emit("Could not connect to Ollama. Make sure Ollama is active on your Mac.")
            else:
                self.error.emit(str(e))