# RAG Q&A Chatbot — Full Implementation Guide

A complete, working reference implementation of a Retrieval-Augmented Generation
(RAG) chatbot: PDF ingestion → recursive chunking → local embeddings (FAISS) →
LLaMA 3 generation via Groq → SQLite query logging → Streamlit dashboard.

```
rag-chatbot/
├── data/                 # put your source PDFs here
├── vectorstore/          # FAISS index gets saved here (generated)
├── logs/                 # SQLite query log (generated)
├── config.py             # all tunable parameters live here
├── ingest.py             # PDF -> chunks -> embeddings -> FAISS index
├── chain.py              # retrieval + LLaMA 3 (Groq) generation chain
├── logger.py              # SQLite logging + latency timer
├── interface.py                 # Streamlit chat UI
├── dashboard.py            # Streamlit monitoring dashboard
├── requirements.txt
└── .env.example
```

---

## 1. Architecture overview

```
                ┌─────────────┐
   PDFs  ─────► │   ingest.py │
                └──────┬──────┘
                       │ RecursiveCharacterTextSplitter
                       ▼
              chunks (LangChain Documents)
                       │ all-MiniLM-L6-v2 (HuggingFace, local, CPU)
                       ▼
                FAISS vector index  ──► saved to vectorstore/
                       │
                       ▼ (at query time)
   User Q ──► retriever.invoke() ──► top-k chunks ──► prompt ──► Groq LLaMA 3
                       │                                            │
                       ▼                                            ▼
                latency timer                                   answer
                       │                                            │
                       └────────────► logger.py → SQLite ◄──────────┘
                                             │
                                             ▼
                                  dashboard.py (Streamlit)
```

Two Streamlit apps sit on top of the same core: `interface.py` (the chat interface
end users interact with) and `dashboard.py` (the internal monitoring view).
Both import `chain.py`, so there is exactly one retrieval/generation
implementation to maintain.

---

## 2. Prerequisites

- Python 3.10+
- A free [Groq API key](https://console.groq.com/keys) (Groq hosts LLaMA 3
  inference and gives very low latency + a generous free tier)
- ~1–2 GB free disk for the embedding model + FAISS index (varies with corpus size)

---

## 3. Setup

```bash
# 1. Clone/create the project folder and enter it
cd rag-chatbot

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure your API key
cp .env.example .env
# then edit .env and paste your GROQ_API_KEY
```

`requirements.txt` pins the key libraries:

```
langchain, langchain-community, langchain-groq, langchain-huggingface
faiss-cpu
sentence-transformers
pypdf
streamlit
python-dotenv
pandas, plotly
```

---

## 4. Step-by-step implementation

### 4.1 Ingestion pipeline (`ingest.py`)

This is the "AI Solution Prototyping" piece — PDF ingestion, recursive
chunking, and local embeddings into FAISS.

**Load PDFs.** `DirectoryLoader` + `PyPDFLoader` walks `data/` and loads every
PDF page as a LangChain `Document`, preserving `source` and `page` metadata
(used later to cite sources in the UI).

**Recursive chunking.** `RecursiveCharacterTextSplitter` tries to split on
paragraph breaks first (`\n\n`), then sentence breaks, then words — only
falling back to a hard character cut as a last resort. This keeps chunks
semantically coherent, which matters a lot for retrieval quality. The two
knobs you'll tune during evaluation:

- `chunk_size` — larger chunks retain more context per chunk but reduce
  retrieval precision and use more tokens per LLM call
- `chunk_overlap` — higher overlap reduces the chance a fact gets split
  across a chunk boundary, at the cost of redundant storage/embedding time

**Local embeddings.** `all-MiniLM-L6-v2` via `HuggingFaceEmbeddings` runs
entirely on CPU with no external API call — fast, free, and deterministic,
which is why it's a common default for prototyping RAG pipelines.
Embeddings are normalized so FAISS's default L2 index behaves like cosine
similarity.

**FAISS index.** `FAISS.from_documents(...)` builds an in-memory index, then
`save_local()` persists it to disk so you don't have to re-embed on every
app restart.

Run it:

```bash
# Drop PDFs into data/, then:
python ingest.py

# Experiment with chunking parameters:
python ingest.py --chunk_size 500  --chunk_overlap 50
python ingest.py --chunk_size 1500 --chunk_overlap 200
```

Each run overwrites `vectorstore/` with the new index — for a real sweep,
save each run to a differently-named `--persist_dir` and evaluate both (see
§6).

### 4.2 Retrieval + generation chain (`rag_chain.py`)

This is the "AI Tool Evaluation" piece — prompting LLaMA 3 through Groq and
measuring output quality/latency.

- `load_vectorstore()` reloads the persisted FAISS index with the *same*
  embedding model used at ingestion time (embeddings from different models
  aren't compatible with each other).
- `build_chain()` wires up a LangChain **LCEL** pipeline:
  `retriever → format_docs → prompt → ChatGroq(llama-3.1-8b-instant) → StrOutputParser`.
  The system prompt instructs the model to answer only from retrieved
  context and say "I don't know" rather than hallucinate.
- `answer_question()` wraps retrieval and generation each in a `Timer`
  (see `logger.py`) and writes a row to SQLite via `log_query()` — this is
  the automated logging described in your bullet.

Swap `LLM_MODEL_NAME` in `config.py` to `llama-3.1-70b-versatile` for higher
quality at higher latency/cost — a good axis to A/B alongside chunking.

Quick CLI test once you have an index:

```bash
python rag_chain.py
```

### 4.3 Logging (`logger.py`)

A minimal SQLite wrapper. `init_db()` creates `query_logs` if it doesn't
exist (idempotent, safe to call every request). Each row captures:

| column | purpose |
|---|---|
| `timestamp` | when the query happened |
| `question` / `answer` | full text for auditing/QA |
| `retrieved_chunks` | how many chunks the retriever returned |
| `retrieval_latency_ms` / `generation_latency_ms` / `total_latency_ms` | performance monitoring |
| `top_source` | which document the top-ranked chunk came from |
| `chunk_size` / `chunk_overlap` | which ingestion config produced the index in use, so you can compare configs later |

`Timer` is a tiny context manager using `time.perf_counter()` for
millisecond-precision latency measurement.

### 4.4 Chat UI (`interface.py`)

A Streamlit chat app:
- Caches the loaded vectorstore/chain with `@st.cache_resource` so the
  embedding model and FAISS index load once per session, not per message.
- Renders conversation history with `st.chat_message`.
- On each question, calls `answer_question()`, displays the answer, and
  shows retrieved source chunks + per-stage latency in an expander for
  transparency.

Run it:

```bash
streamlit run interface.py
```

### 4.5 Monitoring dashboard (`dashboard.py`)

Reads directly from the SQLite log and renders:
- Headline metrics (query count, average latency by stage)
- A latency-over-time line chart
- A histogram of chunks retrieved per query
- A bar chart of most-frequently-retrieved source documents
- A table comparing average latency across different `chunk_size`/`chunk_overlap`
  configurations you've tested — directly supporting the "identify the
  best-performing configuration" workflow
- A raw recent-queries table for spot-checking answer quality

Run it (in a second terminal, alongside `interface.py`):

```bash
streamlit run dashboard.py
```

---

## 5. End-to-end quick start

```bash
# 1. Install deps + set API key (see §3)

# 2. Add PDFs
cp your_files/*.pdf data/

# 3. Build the vector index
python ingest.py

# 4. Launch the chatbot
streamlit run interface.py

# 5. In a second terminal, launch monitoring
streamlit run dashboard.py
```

Ask a few questions in the chat UI, then flip to the dashboard tab to see
the logged latency/retrieval metrics update.

---

## 6. Running a chunking parameter sweep (evaluation workflow)

To reproduce the "testing accuracy and relevance across varied chunking
parameters" step systematically:

1. Build multiple indices into separate folders:
   ```bash
   python ingest.py --chunk_size 500  --chunk_overlap 50  --persist_dir vectorstore_500_50
   python ingest.py --chunk_size 1000 --chunk_overlap 150 --persist_dir vectorstore_1000_150
   python ingest.py --chunk_size 1500 --chunk_overlap 200 --persist_dir vectorstore_1500_200
   ```
2. Point `VECTORSTORE_DIR` in `config.py` (or an env var override) at each
   in turn, run the same fixed set of evaluation questions through
   `rag_chain.py`, and let `logger.py` capture `chunk_size`/`chunk_overlap`
   alongside latency for each run.
3. Use the dashboard's "Latency by chunking configuration" table, plus
   manual review of the `answer` column in `query_logs`, to pick the
   configuration with the best relevance/latency trade-off.

---

## 7. Extending this project

- **Streaming responses:** swap `StrOutputParser()`'s `.invoke()` for
  `.stream()` in `rag_chain.py` and use `st.write_stream()` in `app.py`.
- **Re-ranking:** add a cross-encoder re-ranker (e.g.
  `sentence-transformers/ms-marco-MiniLM-L-6-v2`) after FAISS retrieval to
  improve precision on larger corpora.
- **Multi-format ingestion:** extend `ingest.py` with additional LangChain
  loaders (`UnstructuredWordDocumentLoader`, `WebBaseLoader`, etc.) beyond PDFs.
- **Answer quality scoring:** log a thumbs-up/down in the Streamlit chat and
  store it as an extra SQLite column to build a feedback loop.
- **Swap vector stores:** the same LCEL chain works with Chroma, Pinecone,
  or Weaviate — only `ingest.py`/`rag_chain.py`'s store construction changes.

---

## 8. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `FileNotFoundError: No PDFs found` | Add at least one `.pdf` to `data/` before running `ingest.py` |
| `GROQ_API_KEY is not set` | Copy `.env.example` to `.env` and paste your key |
| Retrieval returns irrelevant chunks | Try a smaller `chunk_size` (more precise chunks) or increase `RETRIEVER_TOP_K` |
| Answers ignore document content | Check the system prompt is being passed — verify with `python rag_chain.py` directly |
| `allow_dangerous_deserialization` error | Required flag for loading local FAISS pickles you created yourself — safe here since you control the file |
