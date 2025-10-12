from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
import torch
from transformers.utils.quantization_config import BitsAndBytesConfig

# 重写 to_dict 方法避免索引错误
def patched_to_dict(self):
    output = self.__dict__.copy()
    # 修复 bnb_4bit_compute_dtype 处理
    if "bnb_4bit_compute_dtype" in output and output["bnb_4bit_compute_dtype"] is not None:
        dtype_str = str(output["bnb_4bit_compute_dtype"])
        if "." in dtype_str:
            output["bnb_4bit_compute_dtype"] = dtype_str.split(".")[1]
        else:
            # 直接使用字符串值（Windows 兼容）
            output["bnb_4bit_compute_dtype"] = dtype_str
    return output

# 应用补丁
BitsAndBytesConfig.to_dict = patched_to_dict
BASE = "Qwen/Qwen1.5-1.8B-Chat" 
DATA = "project/data/gov_chat_train.jsonl"

tok = AutoTokenizer.from_pretrained(
    BASE, 
    use_fast=True, 
    trust_remote_code=True
    )
model = AutoModelForCausalLM.from_pretrained(
    BASE, 
    load_in_4bit=True, 
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    device_map="auto", 
    trust_remote_code=True
)

lora = LoraConfig(
    task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
)
model = get_peft_model(model, lora)

ds = load_dataset("json", data_files=DATA, split="train")

def format_row(ex):
    # 生成格式化的文本
    text = tok.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=False)
    return {"text": text}

ds = ds.map(format_row)

# 添加tokenization步骤 - 关键修复
def tokenize_function(example):
    # Tokenize文本
    tokenized = tok(
        example["text"],
        truncation=True,
        padding=False,
        max_length=512,  # 根据需要调整最大长度
        return_tensors=None  # 让Trainer处理张量转换
    )
    
    # 添加labels用于计算损失
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

# 应用tokenization并移除原始列
ds = ds.map(tokenize_function, batched=True, remove_columns=ds.column_names)
ds = ds.train_test_split(test_size=0.05, seed=42)

args = TrainingArguments(
    output_dir="models/qwen1.5-gov-lora",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=2,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=20,
    save_strategy="epoch",
    bf16=True,
    remove_unused_columns=False,
    save_strategy="epoch",
    save_total_limit=1,  # 只保留最新的检查点
    load_best_model_at_end=False  # 如果不需要加载最佳模型
)

trainer = Trainer(model=model, args=args, train_dataset=ds["train"], eval_dataset=ds["test"])
trainer.train()
model.save_pretrained("project/models/qwen1.5-gov-lora")
tok.save_pretrained("project/models/qwen1.5-gov-lora")
print("Saved LoRA to project/models/qwen1.5-gov-lora")