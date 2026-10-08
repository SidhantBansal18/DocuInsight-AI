# DocuInsight-AI

A sophisticated Retrieval-Augmented Generation (RAG) system built to index local documents and provide context-aware analysis using a local LLM via Ollama.

## 🚀 Features
- **Local-First Architecture**: Uses FAISS for vector storage and Ollama for embeddings and LLM generation, ensuring complete data privacy.
- **Document Intelligence**: Supports PDF and Text files via LangChain's directory loaders, turning static files into an interactive knowledge base.
- **Optimized Chunking**: Implements `RecursiveCharacterTextSplitter` with tuned chunk sizes and overlaps for high-fidelity retrieval.
- **Comparative Analysis**: Capable of identifying differences between multiple documents (e.g., comparing policy changes across different years).
- **Pure Python Implementation**: Custom wrappers for Ollama to ensure compatibility across different Python versions (tested on Python 3.14).

## 🛠️ Tech Stack
- **Language**: Python 3.14
- **Vector Store**: FAISS
- **Orchestration**: LangChain
- **LLM/Embeddings**: Ollama (`gemma4:31b-cloud`)

## 📋 Prerequisites
1. **Ollama**: Install and run [Ollama](https://ollama.ai/).
2. **Model**: Pull the required model:
   ```bash
   ollama pull gemma4:31b-cloud
   ```
3. **Dependencies**:
   ```bash
   pip install langchain faiss-cpu pypdf requests
   ```

## ⚙️ Setup & Usage
1. **Clone the repository**:
   ```bash
   git clone https://github.com/SidhantBansal18/DocuInsight-AI.git
   cd DocuInsight-AI
   ```
2. **Add Knowledge**:
   Place your `.pdf` or `.txt` files in the `knowledge_base/` directory.
3. **Run the Agent**:
   ```bash
   python rag_agent.py
   ```

## 📂 Project Structure
- `rag_agent.py`: The main agent logic including ingestion, indexing, and chat loop.
- `knowledge_base/`: Directory containing documents for the RAG pipeline.
- `faiss_index/`: Local storage for the generated vector embeddings.

## 📝 License
MIT
