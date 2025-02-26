# 🗂️ Chat-With-PDF

## 📌 About the Project
**Chat-With-PDF** is an AI-powered application that allows users to interact with PDF documents using natural language. Built with **LangChain, RAG, Qdrant, and Mistral (Groq)**, this tool enables seamless document search, question-answering, and retrieval-augmented generation (RAG) for a more efficient and intuitive user experience.

---

## 🚀 Features
- **Conversational AI** – Chat with PDFs and get relevant answers instantly.
- **Qdrant Vector Database** – Efficiently stores and retrieves document embeddings.
- **RAG (Retrieval-Augmented Generation)** – Enhances the accuracy of responses by retrieving relevant document sections.
- **Fast Processing** – Leverages Mistral (Groq) for optimized AI interactions.
- **Streamlit Interface** – Easy-to-use web-based application for seamless user experience.

---

## 🛠️ Installation & Setup
### 1️⃣ Run Qdrant using Docker
Ensure Qdrant is running locally before proceeding:
```bash
docker run -p 6333:6333 -v .:/qdrant/storage/ qdrant/qdrant
```

### 2️⃣ Set Up the Qdrant Database
Run the ingestion script to process and store PDF embeddings:
```bash
python ingest_bot.py
```

### 3️⃣ Start the Application
Launch the Streamlit web interface to interact with PDFs:
```bash
streamlit run app_bot.py
```
