import os
from sentence_transformers import SentenceTransformer
import requests
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download
import sys

def download_with_progress(url, filename):
    """带进度条的文件下载函数"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024
        
        with open(filename, 'wb') as file, tqdm(
            desc=filename,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for data in response.iter_content(block_size):
                size = file.write(data)
                progress_bar.update(size)
                
        return True
    except Exception as e:
        print(f"下载 {filename} 失败: {e}")
        return False

def download_sentence_transformer_model(model_name, save_path):
    """下载SentenceTransformer模型"""
    print(f"开始下载 {model_name}...")
    os.makedirs(save_path, exist_ok=True)
    
    # 使用SentenceTransformer自带的下载机制
    model = SentenceTransformer(model_name)
    model.save(save_path)
    print(f"模型已保存到 {save_path}")

def download_transformers_model(model_name, save_path):
    """下载Transformers模型"""
    print(f"开始下载 {model_name}...")
    os.makedirs(save_path, exist_ok=True)
    
    # 使用huggingface_hub的snapshot_download下载整个模型
    try:
        snapshot_download(
            repo_id=model_name,
            local_dir=save_path,
            local_dir_use_symlinks=False,
            tqdm_class=tqdm
        )
        print(f"模型已保存到 {save_path}")
    except Exception as e:
        print(f"下载 {model_name} 失败: {e}")
        print("尝试使用AutoTokenizer和AutoModelForCausalLM...")
        
        # 备用方法
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        tokenizer.save_pretrained(save_path)
        model.save_pretrained(save_path)
        print(f"模型已保存到 {save_path}")

if __name__ == "__main__":
    print("开始下载所需模型...")
    
    # 下载BGE-M3嵌入模型
    print("\n=== 下载BGE-M3嵌入模型 ===")
    download_sentence_transformer_model("BAAI/bge-m3", "project/models/bge-m3")
    
    # 下载Qwen1.5-1.8B-Chat语言模型
    print("\n=== 下载Qwen1.5-1.8B-Chat语言模型 ===")
    download_transformers_model("Qwen/Qwen1.5-1.8B-Chat", "project/models/qwen1.5-1.8b-chat")
    
    print("\n所有模型下载完成！")