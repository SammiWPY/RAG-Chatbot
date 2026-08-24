# Learn RAG From Scratch: A Step-by-Step Build Guide

This guide teaches you to build the RAG chatbot yourself, piece by piece,
understanding *why* each part exists before you write it. Don't copy-paste
the whole thing at once — work through it in order. Each stage ends with a
**checkpoint** you should actually run before moving on.

By the end you'll be able to explain (not just run) every line of the
project, and you'll be equipped to modify it confidently.

---

## Part 0: The mental model (read this before touching code)

Before any code, understand the problem RAG solves.

**The problem:** An LLM like LLaMA 3 only knows what it was trained on. It
has never seen your PDFs. If you paste your whole document into the prompt,
you'll blow past context limits and pay for a huge number of tokens on
every single question, most of it irrelevant.

**The RAG idea:** Instead of giving the model the whole document, give it
*only the relevant slice* of the document, fetched fresh for each question.

That means two separate phases that never talk to each other in real time:

1. **Indexing phase (offline, done once per document set):** chop your
   documents into small pieces ("chunks"), convert each chunk into a vector
   of numbers that captures its meaning ("embedding"), and store all those
   vectors in a searchable structure ("vector index").
2. **Query phase (online, done per question):** convert the *question* into
   a vector the same way, find the stored chunks whose vectors are closest
   to it ("retrieval"), and hand only those chunks to the LLM along with the
   question, asking it to answer using just that context ("generation").

That's the whole idea. Everything else in this project — FAISS, MiniLM,
Groq, SQLite, Streamlit — is just tooling in service of those two phases.
Keep this diagram in your head:

```
INDEXING (once):   PDF → chunks → embed each chunk → store vectors (FAISS)
QUERYING (per Q):  question → embed question → find nearest chunk vectors
                    → stuff those chunks + question into a prompt → LLM answers
```

**Checkpoint 0:** Before writing code, say out loud (or write in one
sentence) why you can't just paste the whole PDF into the prompt every time.
If you can't answer that, re-read this section.

---

## Part 1: Environment setup

### 1.1 Why a virtual environment

Python packages installed globally can conflict between projects. A venv
gives this project its own isolated set of installed packages.

```bash
mkdir rag-chatbot && cd rag-chatbot
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

You'll know it worked because your terminal prompt now shows `(venv)`.

### 1.2 Get a Groq API key

Groq hosts LLaMA 3 and gives you fast, low-latency inference through a
simple HTTP API (this is *not* running the model on your machine — Groq's
servers do the generation). Go to console.groq.com, create a free account,
and generate an API key. Keep it somewhere safe — you'll never commit it to
code.

**Why Groq specifically, and not OpenAI?** No deep reason beyond this
project's design — Groq's free tier is generous and its inference is very
fast, which is nice while iterating. The architecture doesn't care; you
could swap in any chat-completion API later.

### 1.3 Install dependencies one layer at a time

Rather than installing everything from a requirements file blindly, install
in the order you'll use them, so you understand what each one is for:

```bash
pip install pypdf langchain-community langchain-text-splitters
```
— PDF loading + text splitting.

```bash
pip install sentence-transformers langchain-huggingface
```
— local embedding model support.

```bash
pip install faiss-cpu
```
— the vector index library.

```bash
pip install langchain-groq langchain-core
```
— the LLM call + LangChain's chain-building primitives.

```bash
pip install python-dotenv
```
— loads your API key from a `.env` file instead of hardcoding it.

```bash
pip install streamlit pandas plotly
```
— the UI and dashboard, later.

**Checkpoint 1:** Run `pip list` and confirm all of the above appear. Create
a `.env` file containing `GROQ_API_KEY=your_key_here` and add `.env` to a
`.gitignore` if you're using git — never commit real API keys.

---

## Part 2: Build the ingestion pipeline, one stage at a time

Create a folder `data/` and drop 1–2 PDFs into it — pick something short
first (a few pages) so you can eyeball the output at each step.

### 2.1 Stage 1 — just load the PDF and look at it

Don't chunk or embed yet. First understand what a "Document" is.

```python
# step1_load.py
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("data/your_file.pdf")
docs = loader.load()

print(f"Loaded {len(docs)} page-documents")
print(docs[0].page_content[:300])   # first 300 characters of page 1
print(docs[0].metadata)             # {'source': ..., 'page': 0}
```

Run it. You should see one `Document` object *per page*, each with
`.page_content` (the text) and `.metadata` (which file/page it came from).
That metadata matters later — it's how you'll cite sources in the UI.

**Why PyPDFLoader specifically?** It's a thin, dependency-light wrapper
around `pypdf` that plugs straight into LangChain's Document format, which
every downstream LangChain component (splitters, vector stores, retrievers)
expects. If you had Word docs instead, you'd swap this one loader for
`UnstructuredWordDocumentLoader` and nothing else in the pipeline would change.

If you have many PDFs, swap in `DirectoryLoader` to loop over a whole folder:

```python
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader("data", glob="**/*.pdf", loader_cls=PyPDFLoader)
docs = loader.load()
```

**Checkpoint 2.1:** Print `len(docs)` and confirm it matches your PDF's page
count. Print one page's content and confirm it's readable text (if it's
garbled or empty, your PDF may be a scanned image — that needs OCR, which
this project doesn't cover).

### 2.2 Stage 2 — chunking, and why "recursive"

A whole page is usually too big and too topically mixed for good retrieval
(a page might cover three unrelated points; you want to retrieve just the
one relevant to the question). So we split each Document into smaller
chunks.

The naive approach: cut every N characters. Problem — that will slice
sentences and words in half at arbitrary points, destroying meaning right
at the chunk boundary.

**Recursive splitting** fixes this by trying a list of separators in order
of preference — paragraph breaks first, then line breaks, then sentence
ends, then spaces, only cutting mid-word as an absolute last resort:

```python
# step2_chunk.py
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""],
)
chunks = splitter.split_documents(docs)

print(f"{len(docs)} pages became {len(chunks)} chunks")
print(chunks[0].page_content)
print(chunks[0].metadata)   # still carries source/page
```

Two parameters to actually understand, not just copy:

- **`chunk_size`** (in characters): how big each piece is. Too small and a
  chunk loses context (a fact split from the sentence that explains it);
  too large and you dilute relevance (a chunk with five topics mixed
  together, only one relevant to the question, still gets embedded as one
  vector — an average that represents none of them well).
- **`chunk_overlap`**: how many characters the end of one chunk repeats at
  the start of the next. This exists so a fact sitting right at a chunk
  boundary doesn't get orphaned — split so that neither chunk alone
  contains the whole idea. The tradeoff is redundant storage and embedding
  cost.

**Checkpoint 2.2:** Try `chunk_size=200` vs `chunk_size=2000` on the same
PDF and print `len(chunks)` for each. Confirm smaller chunk_size produces
more chunks. Read a couple of chunks at the small size — do they feel like
they're cutting off mid-thought? That's the tradeoff you're seeing directly.

### 2.3 Stage 3 — embeddings: turning text into vectors

This is the conceptual core of RAG, so slow down here.

An **embedding model** maps a piece of text to a fixed-length list of
numbers (a vector) such that pieces of text with *similar meaning* end up
with vectors that are *close together* in that numeric space. "The cat sat
on the mat" and "A feline rested on the rug" would land near each other
even though they share almost no words, because a good embedding model
captures meaning, not just vocabulary.

We're using `all-MiniLM-L6-v2`, a small model (~80MB) that runs entirely on
your CPU — no API call, no cost, no network dependency for this step. It
outputs 384-dimensional vectors. It's not the most powerful embedding model
available, but it's fast and good enough to prototype with, which is why
it's a common default.

```python
# step3_embed.py
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

vector = embeddings.embed_query("What is the refund policy?")
print(len(vector))     # 384
print(vector[:5])      # first 5 numbers, just to see they're real floats
```

The first run will download the model (~80MB) from HuggingFace — that's
normal, it's cached afterward.

**Why `normalize_embeddings=True`?** It scales every vector to length 1.
This makes "closeness" comparisons (specifically cosine similarity) behave
consistently with FAISS's default distance metric — without it, a longer
vector could appear "closer" to things just because of its magnitude, not
its actual meaning. This is a subtle but important correctness detail.

**Checkpoint 2.3:** Embed two clearly similar sentences and two clearly
unrelated ones. You don't need to compute similarity by hand yet — just
confirm both embeddings are 384-length lists of floats and that they're
different from each other (proving the model is actually responding to
content, not returning a constant).

### 2.4 Stage 4 — FAISS: storing and searching vectors

Once every chunk has a vector, you need a structure to search: "given this
new query vector, which stored chunk vectors are nearest?" Doing this by
brute-force comparing every vector works fine at small scale (which is what
FAISS does under the hood for small indices) — the value FAISS adds is a
consistent, fast API for this that also scales to millions of vectors later.

```python
# step4_index.py
from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_documents(chunks, embeddings)

# Try a similarity search directly, no LLM involved yet
results = vectorstore.similarity_search("What is the refund policy?", k=3)
for r in results:
    print("---")
    print(r.page_content[:200])
    print(r.metadata)
```

**This is the step to really understand before adding an LLM.** Run this
and manually judge: are the top-3 returned chunks actually relevant to your
query? If retrieval quality is bad, no amount of clever prompting later will
fix it — garbage in, garbage out. This is why RAG debugging almost always
starts here, not at the LLM.

Persist it so you don't have to re-embed every time:

```python
vectorstore.save_local("vectorstore")

# later, reload:
vectorstore = FAISS.load_local(
    "vectorstore", embeddings, allow_dangerous_deserialization=True
)
```

That `allow_dangerous_deserialization=True` flag exists because FAISS's
save format uses Python's `pickle`, which *can* execute arbitrary code if
you load a file from someone untrusted. It's safe here because you're only
ever loading a file your own `ingest.py` created.

**Checkpoint 2.4:** Ask `similarity_search` three questions: one clearly
answerable from your PDF, one about something the PDF doesn't mention at
all, and one ambiguous one. Look at what comes back each time. Notice that
similarity search *always* returns something, even for the irrelevant
question — it returns the "closest available" chunks even if none are
actually relevant. This is a real limitation you should know about (some
production systems add a relevance-score cutoff to handle this).

### 2.5 Assemble `ingest.py`

Now combine 2.1–2.4 into one script with a `main()` and CLI arguments for
`chunk_size`/`chunk_overlap`, so you can re-run the whole pipeline with
different parameters without editing code each time. (This is exactly the
`ingest.py` from the delivered project — go back and read it now; every
line should make sense given what you just built manually.)

---

## Part 3: Build the retrieval + generation chain

### 3.1 What a "retriever" adds over `similarity_search`

`vectorstore.as_retriever(search_kwargs={"k": 4})` wraps the vector store in
LangChain's standard `Retriever` interface. Functionally, for our purposes,
it's the same lookup you already did in 2.4 — the wrapper just lets it plug
into LangChain's chain-composition syntax (next section) instead of you
calling `.similarity_search()` manually every time.

### 3.2 Prompt design: why a system message with a context slot

```python
SYSTEM_PROMPT = """You are a helpful assistant answering questions strictly using the
provided context retrieved from the user's documents. If the answer is not
contained in the context, say you don't know rather than guessing.

Context:
{context}
"""
```

Three deliberate choices here worth understanding:

1. **"strictly using the provided context"** — without this instruction,
   the LLM will happily answer from its own general training knowledge,
   which defeats the purpose of RAG (you want answers grounded in *your*
   documents, not the model's guesses).
2. **"say you don't know rather than guessing"** — this is your main
   defense against hallucination. It won't be perfect, but it meaningfully
   reduces confidently-wrong answers when retrieval comes up empty.
3. **`{context}` as a template slot** — this gets filled in per-request with
   the retrieved chunks, not hardcoded, since it's different every query.

### 3.3 LCEL: chaining pieces together

LangChain Expression Language (LCEL) lets you compose a pipeline with the
`|` operator, similar to Unix pipes:

```python
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.1-8b-instant", temperature=0.2)

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}"),
])

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

answer = chain.invoke("What is the refund policy?")
print(answer)
```

Read this right to left in terms of data flow, but understand the *first*
dict specially: `{"context": ..., "question": ...}` is LCEL's way of
building the two prompt-template variables in parallel from the same input.
`RunnablePassthrough()` for `"question"` just means "pass the original
input through unchanged, use it directly as the question." Then that dict
feeds `prompt` (fills the template) → `llm` (generates a response) →
`StrOutputParser()` (extracts plain text instead of a message object).

**Why `temperature=0.2` and not higher?** Lower temperature makes output
more deterministic/focused — appropriate for a Q&A system where you want
consistent, grounded answers, not creative variation. If you were building
a creative-writing assistant you'd want it higher.

**Checkpoint 3:** Run this chain on the same three questions from Checkpoint
2.4 (clearly answerable / clearly absent / ambiguous). Confirm: for the
absent one, does the model actually say it doesn't know, or does it
hallucinate? If it hallucinates, that's useful — it tells you the prompt
needs strengthening, or the temperature needs lowering further.

---

## Part 4: Add latency + logging instrumentation

### 4.1 Why measure retrieval and generation separately

If a query is slow, you need to know *which stage* is slow to know what to
optimize — a slow embedding lookup has a completely different fix (better
indexing, fewer chunks) than a slow LLM call (smaller model, shorter
context, streaming).

```python
import time

class Timer:
    def __enter__(self):
        self._start = time.perf_counter()
        return self
    def __exit__(self, *exc):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000

with Timer() as t:
    docs = retriever.invoke(question)
print(t.elapsed_ms)
```

`time.perf_counter()` (not `time.time()`) is the right tool for measuring
short durations — it's a monotonic clock unaffected by system clock
adjustments, giving you a clean, always-increasing measurement.

### 4.2 Why SQLite, not just a text file

You want structured, queryable logs — "what's my average latency this
week" is a `GROUP BY` query in SQLite, but a painful regex job on a log
file. SQLite specifically needs zero setup (it's a single file on disk, no
server process), which is ideal for a prototype.

```python
import sqlite3

conn = sqlite3.connect("logs/query_logs.db")
conn.execute("""
CREATE TABLE IF NOT EXISTS query_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT, question TEXT, answer TEXT,
    retrieved_chunks INTEGER,
    retrieval_latency_ms REAL, generation_latency_ms REAL
)
""")
conn.execute(
    "INSERT INTO query_logs (timestamp, question, answer, retrieved_chunks, retrieval_latency_ms, generation_latency_ms) VALUES (?, ?, ?, ?, ?, ?)",
    ("2026-08-10T12:00:00", "What is X?", "X is...", 4, 120.5, 850.2),
)
conn.commit()
```

Note the `?` placeholders instead of f-string interpolation — this is
**parameterized querying**, and it's not a style preference. Building SQL
with string formatting is a real security/correctness risk (SQL injection,
plus it silently breaks on quote characters in your data). Always use
placeholders.

**Checkpoint 4:** Manually insert 3 fake rows, then run
`SELECT AVG(retrieval_latency_ms) FROM query_logs;` in a Python shell (or
the `sqlite3` CLI) and confirm you get a sensible average back.

---

## Part 5: Build the Streamlit chat UI

### 5.1 Streamlit's core mental model

Streamlit re-runs your **entire script top to bottom** every time the user
interacts with the page (types a message, clicks a button). This is
unintuitive at first — it means any "state" you want to persist across
interactions (like chat history) must be explicitly stored in
`st.session_state`, or it'll reset every rerun.

```python
import streamlit as st

if "messages" not in st.session_state:
    st.session_state.messages = []   # only initializes once, first run

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
         st.markdown(msg["content"])

if question := st.chat_input("Ask something"):
    st.session_state.messages.append({"role": "user", "content": question})
    # ... generate answer, then:
    st.session_state.messages.append({"role": "assistant", "content": answer})
```

### 5.2 Why `@st.cache_resource` matters here specifically

Loading the embedding model and FAISS index takes real time (model load +
disk read). Without caching, that reload would happen on *every single
rerun* — including every keystroke-driven rerun — which would make the app
unusably slow.

```python
@st.cache_resource
def get_chain():
    vectorstore = FAISS.load_local(...)
    chain, retriever = build_chain(vectorstore)
    return vectorstore, chain, retriever
```

`cache_resource` (not `cache_data`) is specifically for objects that
shouldn't be copied/serialized — like a loaded model or an open connection
— it caches the actual object reference.

**Checkpoint 5:** Run `streamlit run app.py`, ask a question, then ask a
second one. Watch your terminal — confirm the "Loading vector store..."
spinner only fires once, not on every question. If it fires every time,
you've broken the caching (usually by putting the loader call outside the
`@st.cache_resource`-decorated function).

---

## Part 6: Build the monitoring dashboard

This part is mostly pandas + plotly, not new RAG concepts, but it's where
the loop closes: **the logs you wrote in Part 4 become the evaluation data
you use to decide if a chunking config change actually helped.**

```python
import pandas as pd, sqlite3, plotly.express as px

conn = sqlite3.connect("logs/query_logs.db")
df = pd.read_sql_query("SELECT * FROM query_logs", conn)

fig = px.line(df, x="timestamp", y="total_latency_ms")
st.plotly_chart(fig)
```

The one non-obvious piece: grouping by chunking config to compare
experiments —

```python
df.groupby(["chunk_size", "chunk_overlap"])["total_latency_ms"].mean()
```

This only works if you logged `chunk_size`/`chunk_overlap` alongside every
query (which is why `rag_chain.py` in the delivered project passes those
into `log_query()`).

**Checkpoint 6:** Run the chatbot with two different `vectorstore` builds
(different chunk sizes), ask the same 3 questions against each, then
confirm the dashboard's grouped table shows both configs with different
average latencies.

---

## Part 7: Put it all together and test the whole loop

At this point you should have all six files from the delivered project,
but built up piece by piece rather than copy-pasted. Do one final
end-to-end test:

```bash
python ingest.py                 # build the index
streamlit run app.py             # terminal 1
streamlit run dashboard.py       # terminal 2
```

Ask 5–10 real questions about your PDF in the chat UI. Then:

1. In the dashboard, confirm every question shows up with real latency numbers.
2. In the chat UI's source expander, confirm the retrieved chunks actually
   relate to what you asked.
3. Deliberately ask something *not* in the PDF and confirm the model
   declines gracefully instead of making something up.

If all three hold, you've built (and verified) a working RAG system, not
just run someone else's code.

---

## Part 8: Self-check — can you explain these without looking?

Test your own understanding before calling this "learned":

1. Why can't you just paste the whole document into every prompt instead of using retrieval?
2. What does `chunk_overlap` actually protect against?
3. Why does the embedding model need to be *the same* at indexing time and query time?
4. What does `normalize_embeddings=True` change about how FAISS compares vectors?
5. Why does `similarity_search` always return results, even for irrelevant questions — and why is that a problem worth handling?
6. What's the difference between what `retrieval_latency_ms` measures and what `generation_latency_ms` measures, and which would you look at first if the app felt slow?
7. Why does `st.session_state` exist — what breaks without it?
8. Why are `?` placeholders used in the SQL insert instead of an f-string?

If any of these feel shaky, jump back to that Part and re-run its checkpoint.

---

## Part 9: Where to go next

- Swap the embedding model for a larger one (e.g. `BAAI/bge-base-en-v1.5`)
  and compare retrieval quality on the same test questions.
- Add a relevance-score threshold to retrieval so it can return "no good
  match found" instead of always returning top-k regardless of quality.
- Try `llama-3.1-70b-versatile` vs `llama-3.1-8b-instant` and log both, then
  compare answer quality against latency/cost in the dashboard.
- Add source citations inline in the answer itself (not just the expander)
  by having the prompt reference chunk numbers.

You now have both the working project and the understanding to modify every
piece of it deliberately, rather than treating it as a black box.
