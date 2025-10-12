import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# --- 1. 配置模型和数据集 ---

# 基础模型标识符 (我们选用一个表现优秀且对硬件友好的中文模型)
# Qwen1.5-1.8B-Chat 是一个不错的选择
model_name = "Qwen/Qwen1.5-1.8B-Chat"

# 你的 JSONL 数据集文件路径
dataset_path = "project/data/gov_chat_train.jsonl"

# 微调后模型的保存路径
new_model_path = "project/models/qwen1.5-gov-lora"

# --- 2. 加载数据集并进行格式化 ---

# 加载 JSONL 数据集
dataset = load_dataset("json", data_files=dataset_path, split="train")


# 数据集格式化函数：将 instruction, input, output 拼接成一个对话格式
def format_dataset(example):
    # 对于 input 为空的情况
    if example.get("input"):
        prompt = f"Instruction: {example['instruction']}\nInput: {example['input']}\nOutput: {example['output']}"
    else:
        prompt = f"Instruction: {example['instruction']}\nOutput: {example['output']}"
    return {"text": prompt}
dataset = dataset.map(format_dataset)

# 应用格式化 (虽然SFTTrainer可以自动处理，但手动格式化可以更清晰地控制prompt模板)
# SFTTrainer可以直接处理'instruction', 'output'格式，这里为了清晰展示prompt结构
# 在 SFTTrainer 中，我们可以直接指定 `formatting_func` 来达到同样效果
# 为简化代码，我们让 SFTTrainer 自动处理列，这里仅作展示

# --- 3. 配置模型量化 (BitsAndBytes) ---

# 4-bit 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False,
)

# --- 4. 加载基础模型和分词器 ---

# 加载模型，应用量化配置
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto" # 自动将模型分布到可用的硬件上（如GPU）
)
# 如果模型出现 'None' is not a valid padding token 错误，取消下面这行注释
# model.config.use_cache = False 

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# 设置 padding token。对于很多模型，eos_token (end-of-sequence) 就是一个很好的选择
tokenizer.pad_token = tokenizer.eos_token

# --- 5. 配置 LoRA (PEFT) ---

# LoRA 配置
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    # 对于Qwen1.5模型，通常目标模块是 'q_proj', 'k_proj', 'v_proj', 'o_proj'
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"] 
)

# --- 6. 配置训练参数 ---

from trl import SFTConfig

training_arguments = SFTConfig(
    output_dir="./results",              # 训练过程中的检查点保存目录
    num_train_epochs=3,                  # 训练轮次
    per_device_train_batch_size=2,       # 每个设备的批处理大小
    gradient_accumulation_steps=1,       # 梯度累积步数
    optim="paged_adamw_32bit",           # 使用分页优化器节省显存
    save_steps=50,                       # 每50步保存一次检查点
    logging_steps=10,                    # 每10步记录一次日志
    learning_rate=2e-4,                  # 学习率
    weight_decay=0.001,                  # 权重衰减
    fp16=False,                          # 根据你的硬件设置，如果支持bf16，设为False
    bf16=True,                           # 如果你的GPU支持bf16（如Ampere架构），设为True以加速
    max_grad_norm=0.3,                   # 最大梯度范数
    max_steps=-1,                        # 如果设置为正数，则覆盖 num_train_epochs
    warmup_ratio=0.03,                   # 预热比例
    group_by_length=True,                # 按长度分组样本，提高效率
    lr_scheduler_type="constant",        # 学习率调度器类型
    packing=False,                       # 是否将多个短样本打包成一个长样本
    dataset_text_field="text"
)


# --- 7. 初始化 SFTTrainer ---

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    args=training_arguments,
)

# --- 8. 开始训练 ---
print("开始微调...")
trainer.train()
print("微调完成！")

# --- 9. 保存微调后的模型 ---
# 这里只保存了LoRA适配器部分，非常小
trainer.save_model(new_model_path)
print(f"模型已保存至 {new_model_path}")

'''# --- (可选) 10. 测试微调后的模型 ---
from transformers import pipeline

print("\n开始测试微调后的模型...")
prompt = "2023年的国内生产总值（GDP）增长率是多少？"

# 使用 pipeline 进行推理
pipe = pipeline(
    task="text-generation", 
    model=model, # Trainer内部的模型已经应用了PEFT
    tokenizer=tokenizer, 
    max_length=200
)

# 构建符合模型训练格式的 prompt
formatted_prompt = f"Instruction: {prompt}\nOutput:"
result = pipe(formatted_prompt)
print(result[0]['generated_text'])'''