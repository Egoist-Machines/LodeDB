import os
os.environ["RAYON_NUM_THREADS"] = "1"
import time, json, numpy as np
import faiss
from turbovec import TurboQuantIndex

DATA_DIR = os.path.expanduser("~/data/py-turboquant")
DIM, BIT_WIDTH = 3072, 4

def load_openai(dim, seed=42):
    all_vecs = np.load(os.path.join(DATA_DIR, f"openai-{dim}.npy"))
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(all_vecs))
    database = all_vecs[idx[:100_000]]
    queries = all_vecs[idx[100_000:101_000]]
    database /= np.linalg.norm(database, axis=-1, keepdims=True)
    queries /= np.linalg.norm(queries, axis=-1, keepdims=True)
    return database, queries

database, queries = load_openai(DIM)
faiss.omp_set_num_threads(1)

# TurboQuant
tq = TurboQuantIndex(dim=DIM, bit_width=BIT_WIDTH)
tq.add(database)
tq.search(queries[:1], k=64)  # warmup
tq_times = []
for _ in range(5):
    t0 = time.perf_counter()
    tq.search(queries, k=64)
    tq_times.append((time.perf_counter() - t0) / len(queries) * 1000)
tq_ms = sorted(tq_times)[2]

# FAISS PQ
m_pq = DIM
pq = faiss.IndexPQFastScan(DIM, m_pq, 4)
pq.train(database)
pq.add(database)
pq.search(queries[:1], 64)  # warmup
faiss_times = []
for _ in range(5):
    t0 = time.perf_counter()
    pq.search(queries, 64)
    faiss_times.append((time.perf_counter() - t0) / len(queries) * 1000)
faiss_ms = sorted(faiss_times)[2]

result = {"dim": DIM, "bit_width": BIT_WIDTH, "arch": "x86", "threading": "st",
          "tq_ms_per_query": round(tq_ms, 3), "faiss_ms_per_query": round(faiss_ms, 3)}
out = os.path.join(os.path.dirname(__file__), "..", "results", "speed_d3072_4bit_x86_st.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(result, open(out, "w"), indent=2)
print(json.dumps(result, indent=2))
