from langchain_huggingface import HuggingFaceEmbeddings

def embed_chunk():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": False},
    )
    return embeddings
#print(len(vector))     # 384
#print(vector[:5])      # first 5 numbers, just to see they're real floats

