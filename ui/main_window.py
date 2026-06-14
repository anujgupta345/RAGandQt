import os
from PySide6.QtWidgets import *
from PySide6.QtCore import Qt
from workers.chunk_worker import ChunkWorker
from workers.generate_worker import GenerateWorker
from ui.loading_overlay import LoadingOverlay


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('AI Assistant - Versatile Pro RAG')
        self.resize(1300, 850)
        self.current_model = 'llama3.1'
        self.vectorstore = None

        c = QWidget();
        self.setCentralWidget(c);
        r = QVBoxLayout(c)
        r.addWidget(QLabel('AI Assistant'))

        # --- MODEL & TEMPERATURE SETTINGS ---
        self.model_edit = QLineEdit()
        self.model_edit.setPlaceholderText("Enter model (e.g., llama3.1, gemini-1.5-flash, gpt-4o)")
        self.model_label = QLabel('Current Model: llama3.1')

        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(0.0, 1.0)
        self.temp_spin.setSingleStep(0.1)
        self.temp_spin.setValue(0.0)  # 0.0 is strict/to-the-point

        mg = QGroupBox('Model Settings');
        mv = QVBoxLayout(mg);
        mh = QHBoxLayout()
        mh.addWidget(QLabel("Model:"));
        mh.addWidget(self.model_edit)
        mh.addWidget(QLabel("Temperature (0.0=Strict, 1.0=Creative):"));
        mh.addWidget(self.temp_spin)
        mv.addLayout(mh);
        mv.addWidget(self.model_label);
        r.addWidget(mg)

        # --- CHUNK SETTINGS ---
        self.chunk_size = QSpinBox();
        self.chunk_size.setValue(1000);
        self.chunk_size.setMaximum(100000)
        self.chunk_overlap = QSpinBox();
        self.chunk_overlap.setValue(200);
        self.chunk_overlap.setMaximum(100000)
        cg = QGroupBox('Chunk Settings');
        ch = QHBoxLayout(cg)
        ch.addWidget(QLabel('Chunk Size'));
        ch.addWidget(self.chunk_size)
        ch.addWidget(QLabel('Chunk Overlap'));
        ch.addWidget(self.chunk_overlap);
        r.addWidget(cg)

        # --- DOCUMENTS & FAISS DB ---
        self.doc_list = QListWidget()
        self.btn_up = QPushButton('Upload Files')
        self.btn_del = QPushButton('Remove Selected')
        self.btn_save_db = QPushButton('Save DB to Disk')
        self.btn_load_db = QPushButton('Load DB from Disk')

        self.btn_up.clicked.connect(self.upload_docs)
        self.btn_del.clicked.connect(self.delete_doc)
        self.btn_save_db.clicked.connect(self.save_database)
        self.btn_load_db.clicked.connect(self.load_database)

        dg = QGroupBox('Documents & Database');
        dv = QVBoxLayout(dg);
        dv.addWidget(self.doc_list)
        bh = QHBoxLayout()
        bh.addWidget(self.btn_up);
        bh.addWidget(self.btn_del)
        bh.addWidget(self.btn_save_db);
        bh.addWidget(self.btn_load_db)
        dv.addLayout(bh);
        r.addWidget(dg)

        self.doc_label = QLabel('Documents: 0');
        self.chunk_label = QLabel('Chunks: 0');
        self.status = QLabel('Ready')
        r.addWidget(self.doc_label);
        r.addWidget(self.chunk_label);
        r.addWidget(self.status)

        # --- PROMPT & OUTPUT ---
        self.prompt = QTextEdit();
        self.prompt.setPlaceholderText("Type your question here...")
        r.addWidget(self.prompt)

        self.btn_generate = QPushButton('Generate')
        self.btn_generate.clicked.connect(self.generate_answer)
        r.addWidget(self.btn_generate)

        self.output = QTextEdit();
        self.output.setReadOnly(True)
        r.addWidget(self.output)

        self.overlay = LoadingOverlay(self)

    def set_ui_enabled(self, enabled):
        self.btn_up.setEnabled(enabled)
        self.btn_del.setEnabled(enabled)
        self.btn_save_db.setEnabled(enabled)
        self.btn_load_db.setEnabled(enabled)
        self.btn_generate.setEnabled(enabled)
        self.prompt.setEnabled(enabled)

    def upload_docs(self):
        # Allow multi-file selection natively
        files, _ = QFileDialog.getOpenFileNames(self, 'Select Documents (Hold Cmd/Ctrl for multiple)', '',
                                                'Documents (*.pdf *.txt)')
        if not files: return
        for f in files: self.doc_list.addItem(f)
        self.doc_label.setText(f'Documents: {self.doc_list.count()}')
        self.start_chunking()

    def start_chunking(self):
        docs = [self.doc_list.item(i).text() for i in range(self.doc_list.count())]
        if not docs:
            self.vectorstore = None
            self.chunk_label.setText('Chunks: 0')
            return

        self.set_ui_enabled(False)
        self.overlay.set_message('Processing Documents...')
        self.overlay.show()
        self.overlay.raise_()

        self.worker = ChunkWorker(docs, self.chunk_size.value(), self.chunk_overlap.value())
        self.worker.statusChanged.connect(self.status.setText)
        self.worker.chunkCountChanged.connect(lambda c: self.chunk_label.setText(f'Chunks: {c}'))
        self.worker.completed.connect(self.done_chunking)
        self.worker.start()

    def done_chunking(self, vectorstore):
        self.vectorstore = vectorstore
        self.overlay.hide()
        self.status.setText("Ready")
        self.set_ui_enabled(True)

    def delete_doc(self):
        row = self.doc_list.currentRow()
        if row >= 0: self.doc_list.takeItem(row)
        self.doc_label.setText(f'Documents: {self.doc_list.count()}')
        self.start_chunking()

    # --- DB PERSISTENCE ---
    def save_database(self):
        if not self.vectorstore:
            QMessageBox.warning(self, "Error", "No database to save! Upload files first.")
            return
        self.status.setText("Saving database to disk...")
        self.vectorstore.save_local("saved_vector_db")
        self.status.setText("Database saved successfully.")
        QMessageBox.information(self, "Success", "Database saved! You can load it instantly next time.")

    def load_database(self):
        if not os.path.exists("saved_vector_db"):
            QMessageBox.warning(self, "Error", "No saved database found!")
            return

        self.set_ui_enabled(False)
        self.overlay.set_message('Loading Database...')
        self.overlay.show()
        self.overlay.raise_()

        try:
            from langchain_community.vectorstores import FAISS
            from langchain_huggingface import HuggingFaceEmbeddings  # Updated!

            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # Updated!
            self.vectorstore = FAISS.load_local("saved_vector_db", embeddings, allow_dangerous_deserialization=True)
            self.status.setText("Database loaded from disk.")
            self.chunk_label.setText("Chunks: (Loaded from Disk)")
            self.doc_list.clear()
            self.doc_label.setText("Documents: (Loaded from Disk)")
        except Exception as e:
            self.status.setText(f"Error loading DB: {e}")
        finally:
            self.overlay.hide()
            self.set_ui_enabled(True)

    # --- API KEY & GENERATION ---
    def ensure_api_key(self, model_name):
        model_lower = model_name.lower()
        if "gemini" in model_lower:
            env_var = "GOOGLE_API_KEY"; provider = "Google Gemini"
        elif "gpt" in model_lower or "o1" in model_lower:
            env_var = "OPENAI_API_KEY"; provider = "OpenAI"
        else:
            return True

        if not os.getenv(env_var):
            key, ok = QInputDialog.getText(self, f"{provider} API Key Required",
                                           f"Enter {provider} key:\n(Saved for session)", QLineEdit.Password)
            if ok and key.strip():
                os.environ[env_var] = key.strip(); return True
            else:
                return False
        return True

    def generate_answer(self):
        query = self.prompt.toPlainText().strip()
        if not query:
            self.status.setText("Please enter a prompt first.")
            return

        typed_model = self.model_edit.text().strip()
        if typed_model:
            self.current_model = typed_model
            self.model_label.setText(f'Current Model: {typed_model}')

        if not self.ensure_api_key(self.current_model):
            self.status.setText("Cancelled: API key required.")
            return

        self.set_ui_enabled(False)
        self.output.clear()
        self.overlay.set_message('Generating Answer...')
        self.overlay.show()
        self.overlay.raise_()
        self.status.setText("Generating response...")

        temp = self.temp_spin.value()
        self.gen_worker = GenerateWorker(query, self.vectorstore, self.current_model, temp)
        self.gen_worker.finished.connect(self.done_generation)
        self.gen_worker.error.connect(self.error_generation)
        self.gen_worker.start()

    def done_generation(self, result):
        self.output.setPlainText(result)
        self.overlay.hide()
        self.status.setText("Ready")
        self.set_ui_enabled(True)

    def error_generation(self, err_msg):
        self.output.setPlainText(f"Error: {err_msg}")
        self.overlay.hide()
        self.status.setText("Error generating response.")
        self.set_ui_enabled(True)

    def closeEvent(self, event):
        if hasattr(self, 'worker') and self.worker.isRunning(): self.worker.wait()
        if hasattr(self, 'gen_worker') and self.gen_worker.isRunning(): self.gen_worker.wait()
        super().closeEvent(event)