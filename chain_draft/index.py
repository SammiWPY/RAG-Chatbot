from langchain_community.vectorstores import FAISS
from chain_draft.embed import embed_chunk
from chain_draft.chunk import chunk_doc

chunks=chunk_doc()
embeddings=embed_chunk()

vectorstore = FAISS.from_documents(chunks, embeddings)

# Try a similarity search directly, no LLM involved yet
results = vectorstore.similarity_search_with_score("garmin", k=3)
for r, score in results:
    print("---")
    print(f"Similarity Score: {score}")
    print(r.page_content[:200])
    print(r.metadata)

vectorstore.save_local("vectorstore")