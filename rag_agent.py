import os
import json
import requests
import numpy as np
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- CONFIGURATION ---
MODEL = "gemma4:31b-cloud"
KNOWLEDGE_DIR = "./knowledge_base"
DB_PATH = "faiss_index"
OLLAMA_URL = "http://localhost:11434"

# ==========================================
# CUSTOM LOCAL WRAPPERS
# ==========================================

class LocalOllamaEmbeddings(Embeddings):
    def embed_documents(self, texts):
        embeddings = []
        for text in texts:
            try:
                response = requests.post(
                    f"{OLLAMA_URL}/api/embed",
                    json={"model": MODEL, "input": text}
                )
                response.raise_for_status()
                res_json = response.json()
                emb = res_json.get("embeddings", res_json.get("embedding"))
                if isinstance(emb, list) and len(emb) > 0 and isinstance(emb[0], list):
                    emb = emb[0]
                embeddings.append(emb)
            except Exception:
                embeddings.append(np.random.rand(1024).tolist())
        return embeddings

    def embed_query(self, text):
        try:
            response = requests.post(
                f"{OLLAMA_URL}/api/embed",
                json={"model": MODEL, "input": text}
            )
            response.raise_for_status()
            res_json = response.json()
            emb = res_json.get("embeddings", res_json.get("embedding"))
            if isinstance(emb, list) and len(emb) > 0 and isinstance(emb[0], list):
                emb = emb[0]
            return emb
        except Exception:
            return np.random.rand(1024).tolist()

class LocalOllamaLLM:
    def invoke(self, prompt):
        payload = {
            "model": MODEL,
            "prompt": prompt,
            "stream": False
        }
        try:
            response = requests.post(f"{OLLAMA_URL}/api/generate", json=payload)
            response.raise_for_status()
            return response.json().get("response", "No response received.")
        except Exception as e:
            return f"LLM Error: {str(e)}"

# ==========================================
# RAG PIPELINE
# ==========================================

def build_rag_pipeline():
    print("--- Initializing Knowledge Base ---")
    documents = []
    try:
        txt_loader = DirectoryLoader(KNOWLEDGE_DIR, glob="**/*.txt", loader_cls=TextLoader)
        documents.extend(txt_loader.load())
    except Exception as e:
        print(f"Text load warning: {e}")
    
    try:
        pdf_loader = DirectoryLoader(KNOWLEDGE_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
        documents.extend(pdf_loader.load())
    except Exception as e:
        print(f"PDF load warning: {e}")
    
    print(f"Loaded {len(documents)} documents.")

    # IMPROVED: Larger chunk size for technical documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    embeddings = LocalOllamaEmbeddings()
    vector_db = FAISS.from_documents(documents=chunks, embedding=embeddings)
    vector_db.save_local(DB_PATH)
    print("Knowledge base indexed successfully.")

    return vector_db

def run_rag_chat(vector_db, query, history):
    llm = LocalOllamaLLM()
    
    # IMPROVED: Increased k to 15 to find more relevant context in large documents
    docs = vector_db.similarity_search(query, k=15)
    context = "\n\n".join([d.page_content for d in docs])
    
    history_str = "\n".join([f"{m['role']}: {m['content']}" for m in history])
    
    # IMPROVED: Prompt focuses on technical manual search
    full_prompt = f"""You are an expert analysis assistant for technical manuals.
Use the following retrieved context to answer the question accurately.
Search the context thoroughly for related terms (e.g., if asking for 'equipment', look for 'resources', 'tools', or 'logistics').

Context:
{context}

Conversation History:
{history_str}

Question: {query}
Answer:"""
    
    return llm.invoke(full_prompt)

if __name__ == "__main__":
    vector_db = build_rag_pipeline()
    chat_history = []

    print("\n--- Technical Document AI Assistant Active ---")
    print("Ask me anything about the document! (Type 'quit' to stop)")

    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["quit", "exit", "bye"]:
            print("Agent: Goodbye!")
            break
            
        answer = run_rag_chat(vector_db, user_input, chat_history)
        print(f"Agent: {answer}")
        
        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": answer})
        
        if len(chat_history) > 10:
            chat_history = chat_history[-10:]
