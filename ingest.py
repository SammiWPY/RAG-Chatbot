from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import argparse
import os
import time
import config

def load_pdf(data_dir: str):
    loader=DirectoryLoader(
        data_dir,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True
    )
    docs=loader.load()
    if not docs:
        raise FileNotFoundError(
            f"No PDFs found in {data_dir}'. Add at least one .pdf file and retry."
        )
    print(f"Loaded {len(docs)} pages from {data_dir}")
    return docs

def chunk_doc(docs, chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    print(f"{len(docs)} pages became {len(chunks)} chunks")
    return chunks

def embed_chunk(embedding_model_name="sentence_transformers/all-MiniLM-L12-v2"):
    embeddings= HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={"device":"cpu"},
        encode_kwargs={"normalize_embeddings":True}
    )
    return embeddings

def index(chunks, embeddings, persist_dir: str):
    start= time.time()
    vectorstore= FAISS.from_documents(chunks, embeddings)
    elapsed= time.time()-start
    print(f"Embedded {len(chunks)} chunks in {elapsed: .2f}s")

    os.makedirs(persist_dir, exist_ok=True)
    vectorstore.save_local(persist_dir)
    print(f"FAISS index saved to '{persist_dir}'")
    return vectorstore

def main():
    parser= argparse.ArgumentParser(description="Ingest PDFs into a FAISS vector store.")
    parser.add_argument("--data_dir", default=config.DATA_DIR)
    parser.add_argument("--persist_dir", default=config.VECTORSTORE_DIR)
    parser.add_argument("--chunk_size", type=int, default=config.CHUNK_SIZE)
    parser.add_argument("--chunk_overlap", type=int, default=config.CHUNK_OVERLAP)
    args= parser.parse_args()
    docs= load_pdf(args.data_dir)
    chunks=chunk_doc(docs, args.chunk_size, args.chunk_overlap)
    embeddings=embed_chunk(config.EMBEDDING_MODEL_NAME)
    vectorstore=index(chunks, embeddings, args.persist_dir)


if __name__ == "__main__":
    main()
