# 政府文档智能问答系统

基于RAG（Retrieval-Augmented Generation）技术的智能文档检索与问答平台，专门针对政府公文和政策法规进行优化。

## 项目简介

本项目旨在构建一个智能问答系统，能够基于政府文档和政策法规提供准确、权威的问答服务。系统采用RAG架构，结合了信息检索和大语言模型生成的优势，确保回答的准确性和相关性。

## 技术架构

### 核心组件

1. **文档处理模块** (`step1_prepare_chunks.py`)
   - 文档解析与分块处理
   - 文本预处理和标准化

2. **索引构建模块** (`step2_build_index.py`)
   - 混合索引系统（稠密向量 + 稀疏BM25）
   - FAISS向量索引
   - 元数据存储

3. **基础问答模块** (`step3_rag_qa.py`)
   - 混合检索策略
   - 基础模型问答

4. **模型微调模块** (`step5_SFT_lora.py`)
   - 使用LoRA技术对大语言模型进行微调
   - 针对政府文档问答场景优化

5. **LoRA增强问答模块** (`step6_rag_with_lora.py`)
   - 结合微调后的LoRA模型进行问答
   - 更准确的政府文档理解能力

6. **Web应用模块** (`web_app.py` / `web_app_lora.py`)
   - FastAPI后端服务
   - 前端交互界面

### 技术栈

- **语言模型**: Qwen/Qwen1.5-1.8B-Chat
- **嵌入模型**: BAAI/bge-m3
- **检索系统**: FAISS + BM25
- **微调技术**: LoRA (Low-Rank Adaptation)
- **Web框架**: FastAPI + HTML/CSS/JavaScript
- **依赖管理**: requirements.txt

## 安装与部署

### 环境要求

- Python 3.10+
- CUDA 11.8+ (可选，用于GPU加速)
- 至少16GB RAM
- 至少10GB磁盘空间

### 安装步骤

1. **克隆项目仓库**：
```bash
git clone <repository-url>
cd govrag
```

2. **创建虚拟环境**：
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

3. **安装依赖**：
```bash
pip install -r requirements.txt
```

4. **准备模型文件**：
   - 下载基础模型到 `project/models/qwen1.5-1.8b-chat/`
   - 下载嵌入模型到 `project/models/bge-m3/`
   - (可选) 微调LoRA模型到 `project/models/qwen1.5-gov-lora/`

### 数据准备

1. 将政府文档放置在指定目录
2. 运行文档处理脚本：
```bash
python project/step1_prepare_chunks.py
```

3. **构建索引**：
```bash
python project/step2_build_index.py
```

### 模型微调（可选）

如果需要针对特定领域进行微调：
```bash
python project/step5_SFT_lora.py
```

### 启动服务

**启动基础版本**：
```bash
python web_app.py
```

**启动LoRA增强版本**：
```bash
python web_app_lora.py
```

**访问Web界面**：打开浏览器访问 `http://localhost:8000`

## 使用说明

### Web界面操作

1. 在输入框中输入关于政府文档或政策法规的问题
2. 调整返回结果数量（默认为6个）
3. 点击"提问"按钮获取答案
4. 系统将显示答案和相关文档来源

### API接口

- `POST /api/ask` - 提问接口
- `GET /api/health` - 健康检查接口

## 项目特点

### 技术优势

- **混合检索**: 结合稠密向量检索和BM25稀疏检索，提高检索准确性
- **LoRA微调**: 使用参数高效微调技术，在保持模型通用性的同时提升特定领域表现
- **离线部署**: 支持完全离线运行，保护数据隐私
- **模块化设计**: 各功能模块独立，便于维护和扩展

### 业务优势

- **准确性**: 基于真实政府文档回答，确保信息准确性
- **权威性**: 严格依据官方文档，避免虚假信息
- **可追溯性**: 提供答案来源，便于验证和深入查阅
- **易用性**: 简洁直观的Web界面，降低使用门槛

## 目录结构

```
govrag/
├── project/                 # 核心代码目录
│   ├── models/             # 模型文件目录
│   │   ├── bge-m3/         # 嵌入模型
│   │   ├── qwen1.5-1.8b-chat/  # 基础语言模型
│   │   └── qwen1.5-gov-lora/   # 微调后的LoRA模型
│   ├── index/              # 索引文件目录
│   ├── step1_prepare_chunks.py  # 文档处理
│   ├── step2_build_index.py     # 索引构建
│   ├── step3_rag_qa.py          # 基础问答
│   ├── step5_SFT_lora.py        # 模型微调
│   └── step6_rag_with_lora.py   # LoRA增强问答
├── static/                  # 静态文件目录
│   └── index.html           # 前端界面
├── web_app.py               # 基础版本后端
├── web_app_lora.py          # LoRA版本后端
├── requirements.txt         # 项目依赖
└── README.md               # 项目说明文档
```


## 致谢

- 感谢阿里云千问团队提供的Qwen模型
- 感谢Hugging Face提供的Transformers库
- 感谢Facebook AI Research提供的Faiss库
