import os, pickle, numpy as np, faiss, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

IDX_DIR = "project/index"
EMB_NAME = "BAAI/bge-m3"

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Current device: {torch.cuda.get_device_name()}")
    print(f"Memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# 选择一个性能/显存友好的中文指令模型（先不微调）
BASE = "Qwen/Qwen1.5-1.8B-Chat"
#BASE = "Qwen/Qwen2.5-3B-Instruct"   # 显存小；若有更大显存可换 7B
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# ---- Load indices
index = faiss.read_index(os.path.join(IDX_DIR, "faiss.index"))
vecs = np.load(os.path.join(IDX_DIR, "dense.npy"))
# with open(os.path.join(IDX_DIR, "meta.pkl",""), "rb") as r:
#     pass
# ↑ 上面这行是防误触示例。如果你复制这里的代码，请删除这两行“with open ... pass”。
# 正确做法如下：
import pickle as pkl
with open(os.path.join(IDX_DIR, "meta.pkl"), "rb") as r:
    store = pkl.load(r)
texts, metas = store["texts"], store["metas"]
with open(os.path.join(IDX_DIR, "bm25.pkl"), "rb") as r:
    bm25 = pkl.load(r)["bm25"]

emb = SentenceTransformer(EMB_NAME)

# ---- LLM
tok = AutoTokenizer.from_pretrained(
    BASE,                                
    trust_remote_code=True
    #local_files_only=True
    )
model = AutoModelForCausalLM.from_pretrained(
    BASE, device_map="auto", 
    torch_dtype=DTYPE, 
    trust_remote_code=True,
    load_in_4bit=True
    #local_files_only=True
)

def hybrid_search(query, topk_dense=40, topk_bm25=30, final_k=6):
    qv = emb.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    D, I = index.search(qv.astype("float32"), topk_dense)
    dense_hits = [(int(i), float(D[0][j])) for j, i in enumerate(I[0])]

    bm25_scores = bm25.get_scores(query.split())
    bm25_hits = sorted([(i, float(bm25_scores[i])) for i in range(len(texts))], key=lambda x: x[1], reverse=True)[:topk_bm25]

    # 归一化并合并去重
    all_ids = {}
    for i, s in dense_hits:
        all_ids[i] = max(all_ids.get(i, 0.0), s)
    if bm25_hits:
        maxb = max(s for _, s in bm25_hits)
        for i, s in bm25_hits:
            all_ids[i] = max(all_ids.get(i, 0.0), s / (maxb + 1e-6))  # 粗糙归一化

    # 取 topN，简单 MMR 可后续加入
    sel = sorted(all_ids.items(), key=lambda x: x[1], reverse=True)[:max(final_k*2, 10)]
    sel_ids = [i for i,_ in sel][:final_k]
    blocks = []
    for rank, idx in enumerate(sel_ids, 1):
        m = metas[idx]
        blocks.append(f"[{rank}] ({m.get('doc')}) sec{m.get('section_id')}#{m.get('chunk_id')}\n{texts[idx]}")
    return "\n\n".join(blocks)

def build_prompt(q, ctx):
    system = "你是政府公文助手。仅依据下列资料作答，若依据不足请直说无法确认。回答用正式书面语，先给结论，再列要点，并在文末给出引用编号。"
    messages = [
        {"role":"system","content":system},
        {"role":"user","content":f"问题：{q}\n\n资料：\n{ctx}\n\n请按要求作答。"}
    ]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def ask(query):
    ctx = hybrid_search(query)
    prompt = build_prompt(query, ctx)
    ids = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**ids, max_new_tokens=600, temperature=0.3, top_p=0.9)
    text = tok.decode(out[0], skip_special_tokens=True)
    return text

    
if __name__ == "__main__":
    while True:
        q = input("\nQ> ")
        print(ask(q))
        if torch.cuda.is_available():
            print(f"Current device: {torch.cuda.get_device_name()}")
            print(f"Memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")