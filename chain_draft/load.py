# step1_load.py
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("data/Project Description_26S.pdf")
docs = loader.load()

print(f"Loaded {len(docs)} page-documents")
print(docs[0].page_content[:500])   # first 300 characters of page 1
print(docs[0].metadata)             # {'source': ..., 'page': 0}

def load_pdf(path="data/Project Description_26S.pdf"):
    loader=PyPDFLoader(path)
    docs=loader.load()
    return docs
    