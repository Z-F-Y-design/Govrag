# web_app.py
import os, pickle, numpy as np, faiss, torch
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import asyncio
import json

# 初始化FastAPI应用
app = FastAPI(title="政府文档智能问答系统", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # 允许所有网址访问（星号代表"所有人"）
    allow_credentials=True,   # 允许携带身份信息
    allow_methods=["*"],      # 允许所有类型的请求（GET、POST等）
    allow_headers=["*"],      # 允许所有类型的文件头
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
        self.index_dir = "project/index"
        self.emb_name = "BAAI/bge-m3"
        self.model_name = "Qwen/Qwen1.5-1.8B-Chat"
        
        # 加载索引和模型
        print("正在加载检索索引...")
        self.load_indices()
        print("正在加载嵌入模型...")
        self.emb = SentenceTransformer(self.emb_name)
        print("正在加载语言模型...")
        self.load_llm()
        print("系统初始化完成！")
    
    def load_indices(self):
        """加载所有索引文件"""
        try:
            self.index = faiss.read_index(os.path.join(self.index_dir, "faiss.index"))
            self.vecs = np.load(os.path.join(self.index_dir, "dense.npy"))
            
            with open(os.path.join(self.index_dir, "meta.pkl"), "rb") as r:
                store = pickle.load(r)
            self.texts, self.metas = store["texts"], store["metas"]
            
            with open(os.path.join(self.index_dir, "bm25.pkl"), "rb") as r:
                self.bm25 = pickle.load(r)["bm25"]
        except Exception as e:
            raise RuntimeError(f"加载索引失败: {e}")
    
    def load_llm(self):
        """加载语言模型"""
        try:
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                torch_dtype=dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
        except Exception as e:
            raise RuntimeError(f"加载语言模型失败: {e}")
    
    def hybrid_search(self, query, topk_dense=40, topk_bm25=30, final_k=6):
        """混合检索"""
        # 稠密检索
        qv = self.emb.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        D, I = self.index.search(qv.astype("float32"), topk_dense)
        dense_hits = [(int(i), float(D[0][j])) for j, i in enumerate(I[0])]

        # 稀疏检索
        bm25_scores = self.bm25.get_scores(query.split())
        bm25_hits = sorted([(i, float(bm25_scores[i])) for i in range(len(self.texts))], 
                          key=lambda x: x[1], reverse=True)[:topk_bm25]

        # 结果融合
        all_ids = {}
        for i, s in dense_hits:
            all_ids[i] = max(all_ids.get(i, 0.0), s)
        
        if bm25_hits:
            maxb = max(s for _, s in bm25_hits)
            for i, s in bm25_hits:
                all_ids[i] = max(all_ids.get(i, 0.0), s / (maxb + 1e-6))

        # 选择最终结果
        sel = sorted(all_ids.items(), key=lambda x: x[1], reverse=True)[:final_k]
        sel_ids = [i for i, _ in sel]
        
        # 构建返回结果
        sources = []
        for idx in sel_ids:
            meta = self.metas[idx]
            sources.append({
                "rank": len(sources) + 1,
                "document": meta.get("doc", "未知文档"),
                "section": meta.get("section_id", 0),
                "chunk": meta.get("chunk_id", 0),
                "content": self.texts[idx][:200] + "..." if len(self.texts[idx]) > 200 else self.texts[idx],
                "score": all_ids[idx]
            })
        
        return sel_ids, sources
    
    def build_prompt(self, question, context_texts):
        """构建提示词"""
        system = "你是政府公文助手。仅依据下列资料作答，若依据不足请直说无法确认。回答用正式书面语，先给结论，再列要点，并在文末给出引用编号。"
        
        context = "\n\n".join([f"[{i+1}] {text}" for i, text in enumerate(context_texts)])
        
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"问题：{question}\n\n资料：\n{context}\n\n请按要求作答。"}
        ]
        
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    
    async def ask(self, question, top_k=6):
        """处理用户提问"""
        start_time = asyncio.get_event_loop().time()
        
        # 检索相关文档
        selected_ids, sources = self.hybrid_search(question, final_k=top_k)
        context_texts = [self.texts[i] for i in selected_ids]
        
        # 构建提示词并生成回答
        prompt = self.build_prompt(question, context_texts)
        input_ids = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **input_ids,
                max_new_tokens=600,
                temperature=0.3,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取生成的回答（去掉提示词部分）
        answer = response.split("请按要求作答。")[-1].strip()
        
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
        raise HTTPException(status_code=500, detail=f"处理问题时出错: {str(e)}")

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "model_loaded": rag_system is not None}

# 挂载静态文件
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)