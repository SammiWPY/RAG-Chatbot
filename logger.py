import time
import sqlite3
from datetime import datetime

class Timer:
    def __enter__(self):
        self._start= time.perf_counter()
        return self
    def __exit__(self, *exc):
        self.elapsed_ms=(time.perf_counter()-self._start)*1000

def log_query(question, answer, retrieved_chunks, retrieval_latency_ms, generation_latency_ms):
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
        (datetime.now().isoformat(), question, answer, retrieved_chunks, retrieval_latency_ms, generation_latency_ms),
    )
    conn.commit()
    conn.close()
    