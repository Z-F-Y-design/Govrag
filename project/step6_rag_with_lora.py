import os, pickle as pkl, numpy as np, faiss, torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

IDX_DIR = "project/index"
EMB_NAME = "BAAI/bge-m3"
BASE = "Qwen/Qwen1.5-1.8B-Chat"
LORA_DIR = "project/models/qwen1.5-gov-lora"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# Load index & bm25
index = faiss.read_index(os.path.join(IDX_DIR, "faiss.index"))
with open(os.path.join(IDX_DIR, "meta.pkl"), "rb") as r: store = pkl.load(r)
texts, metas = store["texts"], store["metas"]
with open(os.path.join(IDX_DIR, "bm25.pkl"), "rb") as r: bm25 = pkl.load(r)["bm25"]
emb = SentenceTransformer(EMB_NAME)

# Load LoRA on base
tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(BASE, device_map="auto", torch_dtype=DTYPE, trust_remote_code=True)
model = PeftModel.from_pretrained(base_model, LORA_DIR)

# ---- 复用第3步的 hybrid_search / build_prompt（略），此处直接粘贴即可 ----
# ...（把 step3_rag_qa.py 里 hybrid_search/build_prompt/ask 的实现拷过来）

# 简短演示
if __name__ == "__main__":
    from step3_rag_qa import hybrid_search, build_prompt  # 若你把函数写到同文件，就不需要这行
    def ask(q):
        ctx = hybrid_search(q)  # 注意：若导入函数，需要确保共享同一个 index/emb
        prompt = build_prompt(q, ctx)
        ids = tok(prompt, return_tensors="pt").to(model.device)
        out = model.generate(**ids, max_new_tokens=600, temperature=0.25, top_p=0.9)
        return tok.decode(out[0], skip_special_tokens=True)
    while True:
        q = input("\nQ> ")
        print(ask(q))