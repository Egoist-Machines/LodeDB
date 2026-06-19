#!/usr/bin/env python3
"""Compression benchmark: TQ index file size vs FP32."""
import os, json, tempfile
import numpy as np
import h5py

DATA_DIR = os.path.expanduser("~/data/py-turboquant")
GLOVE_PATH = os.path.join(DATA_DIR, "glove-200-angular.hdf5")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
RESULT_FILE = os.path.join(RESULTS_DIR, "compression.json")


def load_openai(dim, seed=42):
    path = os.path.join(DATA_DIR, f"openai-{dim}.npy")
    all_vecs = np.load(path)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(all_vecs))
    database = all_vecs[idx[:100_000]]
    queries = all_vecs[idx[100_000:101_000]]
    database /= np.linalg.norm(database, axis=-1, keepdims=True)
    queries /= np.linalg.norm(queries, axis=-1, keepdims=True)
    return database, queries


def load_glove(seed=42):
    f = h5py.File(GLOVE_PATH, "r")
    all_train = f["train"][:].astype(np.float32)
    queries = f["test"][:].astype(np.float32)
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(all_train), 100_000, replace=False)
    database = all_train[idx]
    database /= np.linalg.norm(database, axis=-1, keepdims=True)
    queries /= np.linalg.norm(queries, axis=-1, keepdims=True)
    return database, queries


def measure_index_size(database, dim, bit_width):
    from turbovec import TurboQuantIndex
    index = TurboQuantIndex(dim, bit_width)
    index.add(database)
    with tempfile.NamedTemporaryFile(suffix=".tv", delete=False) as tmp:
        path = tmp.name
    try:
        index.write(path)
        return os.path.getsize(path)
    finally:
        os.remove(path)


def main():
    datasets = [
        ("glove_d200", 200, load_glove),
        ("openai_d1536", 1536, lambda: load_openai(1536)),
        ("openai_d3072", 3072, lambda: load_openai(3072)),
    ]

    results = {}
    for name, dim, loader in datasets:
        print(f"\nLoading {name}...")
        database, _ = loader()
        n = len(database)
        fp32_bytes = n * dim * 4

        for bit_width in [2, 4]:
            key = f"{name}_{bit_width}bit"
            print(f"  {key}...", end=" ", flush=True)

            index_bytes = measure_index_size(database, dim, bit_width)
            fp32_mb = fp32_bytes / (1024 * 1024)
            index_mb = index_bytes / (1024 * 1024)

            results[key] = {
                "n": n,
                "dim": dim,
                "bit_width": bit_width,
                "fp32_mb": round(fp32_mb, 1),
                "index_mb": round(index_mb, 1),
                "ratio": round(fp32_mb / index_mb, 1),
            }
            print(f"{fp32_mb:.1f} MB -> {index_mb:.1f} MB ({fp32_mb / index_mb:.1f}x)")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(RESULT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULT_FILE}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
