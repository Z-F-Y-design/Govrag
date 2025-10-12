import json, os, numpy as np, faiss, pickle
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from modelscope import snapshot_download

CHUNKS = "project/data/clean/chunks.jsonl"
IDX_DIR = "project/index"
EMB_NAME = "BAAI/bge-m3"

os.makedirs(IDX_DIR, exist_ok=True)
model = SentenceTransformer(EMB_NAME)


texts, metas = [], []
with open(CHUNKS, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        texts.append(obj["text"]); metas.append(obj["meta"])

if not texts:
    raise ValueError("No text chunks loaded. Check if chunks.jsonl contains valid data.")
# Dense vectors


vecs = model.encode(texts, batch_size=8, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True)
if vecs.size == 0:
    raise ValueError("Embedding result is empty. Check input data and model.")

dim = vecs.shape[1]
index = faiss.IndexFlatIP(dim); index.add(vecs)

# BM25
tokenized = [t.split() for t in texts]
bm25 = BM25Okapi(tokenized)

# Save
faiss.write_index(index, os.path.join(IDX_DIR, "faiss.index"))
np.save(os.path.join(IDX_DIR, "dense.npy"), vecs)
with open(os.path.join(IDX_DIR, "meta.pkl"), "wb") as w: pickle.dump({"texts": texts, "metas": metas}, w)
with open(os.path.join(IDX_DIR, "bm25.pkl"), "wb") as w: pickle.dump({"bm25": bm25}, w)

print("Index built at", IDX_DIR)