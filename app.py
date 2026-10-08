import streamlit as st
import os
from rag_agent import build_rag_pipeline, run_rag_chat, KNOWLEDGE_DIR

# --- PAGE CONFIG ---
st.set_page_config(page_title="DocuInsight-AI", page_icon="📚", layout="wide")

# Custom CSS for a consistent Dark Theme and better aesthetics
st.markdown("""
    <style>
    /* Force Dark Theme Colors */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    [data-testid="stSidebar"] {
        background-color: #161B22 !important;
        border-right: 1px solid #30363D;
    }
    .stChatMessage {
        border-radius: 15px;
        padding: 10px;
        margin-bottom: 10px;
    }
    div[data-testid="stMetricValue"] {
        color: #58A6FF;
    }
    /* Style the buttons in sidebar to be dark */
    .stButton>button {
        background-color: #21262D;
        color: #C9D1D9;
        border: 1px solid #30363D;
        width: 100%;
    }
    .stButton>button:hover {
        border-color: #8B949E;
        color: #F0F6FB;
    }
    </style>
    """, unsafe_allow_html=True)

# --- SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

# --- SIDEBAR: FILE MANAGEMENT ---
with st.sidebar:
    st.title("📚 DocuInsight-AI")
    st.subheader("Knowledge Base")
    
    # 1. Upload Section
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT", 
        type=["pdf", "txt"], 
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    
    if uploaded_files:
        if not os.path.exists(KNOWLEDGE_DIR):
            os.makedirs(KNOWLEDGE_DIR)
            
        for uploaded_file in uploaded_files:
            file_path = os.path.join(KNOWLEDGE_DIR, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.success(f"Uploaded {len(uploaded_files)} files!")

    # 2. Document List & Deletion Section
    st.markdown("---")
    st.markdown("**Current Documents**")
    
    if os.path.exists(KNOWLEDGE_DIR):
        files = [f for f in os.listdir(KNOWLEDGE_DIR) if f.endswith(('.pdf', '.txt'))]
        if files:
            for file in files:
                col1, col2 = st.columns([0.8, 0.2])
                col1.markdown(f"📄 {file}")
                if col2.button("🗑️", key=f"del_{file}"):
                    os.remove(os.path.join(KNOWLEDGE_DIR, file))
                    st.toast(f"Deleted {file}")
                    st.rerun()
        else:
            st.info("No documents found.")
    else:
        st.info("Knowledge base empty.")

    st.markdown("---")
    
    # 3. Indexing Action
    if st.button("🔄 Re-index Knowledge Base"):
        with st.spinner("Indexing documents... please wait..."):
            try:
                st.session_state.vector_db = build_rag_pipeline()
                st.success("Indexing complete!")
            except Exception as e:
                st.error(f"Indexing failed: {e}")

    st.divider()
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# --- MAIN UI: CHAT INTERFACE ---
st.title("🤖 Document Analysis Agent")
st.markdown("Ask questions about your uploaded documents. I'll analyze the content and provide a detailed answer.")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("What would you like to know about your documents?"):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Add to session state
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Generate AI Response
    with st.chat_message("assistant"):
        if st.session_state.vector_db is None:
            with st.spinner("Please index the knowledge base first using the sidebar!"):
                # Try to auto-initialize if no DB is loaded
                try:
                    st.session_state.vector_db = build_rag_pipeline()
                except:
                    st.error("No documents found. Please upload files and click 'Re-index' in the sidebar.")
                    st.stop()

        with st.spinner("Analyzing documents..."):
            # we pass only the last 10 messages to avoid context overflow
            history = st.session_state.messages[-10:]
            response = run_rag_chat(st.session_state.vector_db, prompt, history)
            st.markdown(response)
            
    # Add AI response to session state
    st.session_state.messages.append({"role": "assistant", "content": response})
