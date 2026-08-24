import os
import streamlit as st
from groq import PermissionDeniedError

from chain import load_vectorstore, build_chain, answer_question
import config

st.set_page_config(page_title="RAG Q&A Chatbot", layout="wide")

st.title("RAG Q&A Chatbot")
st.caption("PDF ingestion • FAISS retrieval • LLaMA 3 (Groq) generation")

@st.cache_resource(show_spinner="Loading vector store and model...")

def get_chain():
    if not os.path.exists(config.VECTORSTORE_DIR):
        return None, None, None
    vectorstore = load_vectorstore()
    chain, retriever = build_chain(vectorstore)
    return vectorstore, chain, retriever

vectorstore, chain, retriever= get_chain()

if vectorstore is None:
    st.warning(
        f"No vectorstore found at '{config.VECTORSTORE_DIR}'. "
        "Add PDFs to the 'data/' folder and run 'python ingest.py' first." 
    )
    st.stop()

if not config.GROQ_API_KEY:
    st.error("GROQ_API_KEY is not set. Add it to your .env file.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

#chat input
if question := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role":"user", "content":question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating answer..."):
            try:
                result = answer_question(question, chain, retriever)
            except PermissionDeniedError:
                st.error(
                    "Groq denied the request (HTTP 403). Check your network "
                    "firewall, VPN, or IP access settings in the Groq console, "
                    "then try again."
                )
                st.stop()
        st.markdown(result["answer"])
        with st.expander(
            f"Sources({len(result['sources'])} chunks retrieved) ·"
            f"retrieval {result['retrieval_latency_ms']:.0f}ms ·"
            f"generation {result['generation_latency_ms']:.0f}ms"
        ):
            for i, doc in enumerate(result["sources"], start=1):
                source_name= doc.metadata.get("source", "unknown")
                page=doc.metadata.get("page", "?")
                st.markdown(f"**chunk {i}** - '{source_name}' (page {page})")
                st.text(doc.page_content[:500]+("..." if len(doc.page_content)>500 else ""))
    st.session_state.messages.append({"role":"assistant", "content": result["answer"]})

with st.sidebar:
    st.header("Settings")
    st.markdown(f"**Embedding model:** '{config.EMBEDDING_MODEL_NAME}'")
    st.markdown(f"**LLM:** '{config.LLM_MODEL_NAME}' (Groq)")
    st.markdown(f"**Chunk size / overlap:** {config.CHUNK_SIZE} / {config.CHUNK_OVERLAP}")
    st.markdown(f"**Top-k retrieved:** {config.RETRIEVER_TOP_K}")
    if st.button("Clear chat"):
        st.session_state.messages=[]
        st.rerun()
    st.divider()
    st.caption("Open 'pages/Dashboard.py' (or run dashboard.py) to view query monitoring metrics. ")