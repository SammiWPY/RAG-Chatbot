from langchain_text_splitters import RecursiveCharacterTextSplitter
from chain_draft.load import load_pdf

def chunk_doc():
    docs=load_pdf() #path can be input here

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    # Debug: Print Phase 1 related chunks
    #for i, chunk in enumerate(chunks):
    #    if "phase 1" in chunk.page_content.lower():
    #        print(f"\n--- Chunk {i} ---")
     #       print(chunk.page_content[:300])
    return chunks


    
#return chunks
#print(f"{len(docs)} pages became {len(chunks)} chunks")
#print(chunks[0].page_content)
#print(chunks[0].metadata)   # still carries source/page