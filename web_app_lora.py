import os, pickle, numpy as np, faiss, torch
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from peft import PeftModel 
import asyncio
import json

# --- 离线运行环境变量  ---
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# 初始化FastAPI应用
app = FastAPI(title="政府文档智能问答系统", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 请求/响应模型
class QueryRequest(BaseModel):
    question: str
    top_k: int = 6

class QueryResponse(BaseModel):
    answer: str
    sources: list
    processing_time: float

# 全局变量
rag_system = None

class RAGSystem:
    def __init__(self):
        """初始化RAG系统"""
        # 1. 索引文件所在的目录
        self.index_dir = "project/index"
        
        # 2. 嵌入模型 (Embedding Model) 的本地文件夹路径
        self.emb_model_path = "project/models/bge-m3"
        
        # 3. 基础大语言模型 (Base LLM) 的本地文件夹路径
        self.base_model_path = "project/models/Qwen1.5-1.8B-Chat"
        
        # 4. 自己微调的 LoRA 模型所在的本地文件夹路径
        self.lora_dir = "project/models/qwen1.5-gov-lora"
        
        # ==================================================================
        
        # 加载索引和模型
        print("正在加载检索索引...")
        self.load_indices()
        
        print(f"正在从本地加载嵌入模型: {self.emb_model_path}")
        self.emb = SentenceTransformer(self.emb_model_path, trust_remote_code=True, local_files_only=True)
        
        print("正在加载语言模型...")
        self.load_llm()
        
        print("系统初始化完成！")
    
    def load_indices(self):
        """加载所有索引文件"""
        try:
            self.index = faiss.read_index(os.path.join(self.index_dir, "faiss.index"))
    
            # self.vecs = np.load(os.path.join(self.index_dir, "dense.npy"))
            
            with open(os.path.join(self.index_dir, "meta.pkl"), "rb") as r:
                store = pickle.load(r)
            self.texts, self.metas = store["texts"], store["metas"]
            
            with open(os.path.join(self.index_dir, "bm25.pkl"), "rb") as r:
                self.bm25 = pickle.load(r)["bm25"]
        except Exception as e:
            raise RuntimeError(f"加载索引失败: {e}")
    
    def load_llm(self):
        """
        如果配置了 LoRA 路径，则加载基础模型并应用 LoRA 适配器。
        否则，直接加载基础模型。
        """
        try:
            dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
            
            print(f"从本地路径加载分词器和基础模型: {self.base_model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_path, trust_remote_code=True, local_files_only=True
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                device_map="auto",
                torch_dtype=dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                # local_files_only=True
            )
            
            # 检查是否需要加载 LoRA 模型
            if self.lora_dir and os.path.exists(self.lora_dir):
                print(f"检测到LoRA路径，正在加载并合并LoRA模型: {self.lora_dir}")
                self.model = PeftModel.from_pretrained(base_model, self.lora_dir, local_files_only=True)
                print("LoRA模型加载完成。")
            else:
                print("未配置或未找到LoRA路径，将直接使用基础模型。")
                self.model = base_model

        except Exception as e:
            raise RuntimeError(f"加载语言模型失败: {e}")
    
    def hybrid_search(self, query, topk_dense=40, topk_bm25=30, final_k=6):
        """混合检索"""
        # 稠密检索
        qv = self.emb.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        D, I = self.index.search(qv.astype("float32"), topk_dense)
        dense_hits = [(int(i), float(D[0][j])) for j, i in enumerate(I[0])]

        # 稀疏检索
        # 使用 Qwen 的分词器进行分词，以获得更好的 BM25 效果
        tokenized_query = self.tokenizer.tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_hits = sorted([(i, float(bm25_scores[i])) for i in range(len(self.texts))], 
                          key=lambda x: x[1], reverse=True)[:topk_bm25]

        # 结果融合 (RRF - Reciprocal Rank Fusion)
        rrf_scores = {}
        for rank, (doc_id, _) in enumerate(dense_hits):
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0
            rrf_scores[doc_id] += 1 / (rank + 60)
        
        for rank, (doc_id, _) in enumerate(bm25_hits):
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0
            rrf_scores[doc_id] += 1 / (rank + 60)
            
        # 选择最终结果
        sorted_docs = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
        sel_ids = [doc_id for doc_id, score in sorted_docs[:final_k]]
        
        # 构建返回结果
        sources = []
        for idx in sel_ids:
            meta = self.metas[idx]
            sources.append({
                "rank": len(sources) + 1,
                "document": meta.get("doc", "未知文档"),
                "section": meta.get("section_id", 0),
                "chunk": meta.get("chunk_id", 0),
                "content": self.texts[idx][:250] + "..." if len(self.texts[idx]) > 250 else self.texts[idx],
                "score": rrf_scores[idx]
            })
        
        return sel_ids, sources
    
    def build_prompt(self, question, context_texts):
        """构建提示词"""
        system = "你是政府公文助手。请仅依据下列资料回答问题，\n如果资料无法得出结论，请直接说“根据现有资料，无法回答该问题”。回答应采用严谨、正式的书面语，首先给出核心结论，然后分点进行详细阐述。在回答的末尾，必须以'来源：[编号]'的格式清晰地列出所有引用的资料编号。如果在问题中出现明确的时间或地区要求，请特别注意资料来源中的年份和地区信息匹配，确保回答符合这些要求！！！"

        context = "\n\n".join([f"[{i+1}] {text}" for i, text in enumerate(context_texts)])
        
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"问题：{question}\n\n参考资料：\n{context}"}
        ]
        
        # 使用 apply_chat_template 来生成完整的、符合模型预训练格式的提示
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    
    async def ask(self, question, top_k=6):
        """处理用户提问"""
        start_time = asyncio.get_event_loop().time()
        
        selected_ids, sources = self.hybrid_search(question, final_k=top_k)
        context_texts = [self.texts[i] for i in selected_ids]
        
        prompt = self.build_prompt(question, context_texts)
        input_ids = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **input_ids,
                max_new_tokens=800,
                temperature=0.2, 
                top_p=0.8,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response_ids = outputs[0][len(input_ids['input_ids'][0]):]
        answer = self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()
        
        processing_time = asyncio.get_event_loop().time() - start_time
        
        return QueryResponse(
            answer=answer,
            sources=sources,
            processing_time=round(processing_time, 2)
        )

# 启动时初始化系统
@app.on_event("startup")
async def startup_event():
    global rag_system
    rag_system = RAGSystem()

# API路由
@app.get("/")
async def read_index():
    return FileResponse("static/index.html")

@app.post("/api/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    if rag_system is None:
        raise HTTPException(status_code=503, detail="系统正在初始化，请稍后重试")
    
    try:
        response = await rag_system.ask(request.question, request.top_k)
        return response
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"处理问题时出错: {str(e)}")

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "model_loaded": rag_system is not None}

async def favicon():
    return FileResponse("static/favicon.ico")

# 挂载静态文件
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn  
    uvicorn.run("web_app_lora:app", host="127.0.0.1", port=8000, reload=True)