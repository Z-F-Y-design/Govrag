# step1_prepare_chunks.py
import os, re, json, jieba
from pypdf import PdfReader

RAW_DIR = "project/data/raw"
OUT_PATH = "project/data/clean/chunks.jsonl"

header_pat = re.compile(r"^([一二三四五六七八九十]+、|（[一二三四五六七八九十]）|\(\d+\)|第[一二三四五六七八九十]+部分)", re.M)

def read_any(path):
    if path.lower().endswith(".pdf"):
        r = PdfReader(path); text = "\n".join(p.extract_text() or "" for p in r.pages)
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as f: text = f.read()
    return text

def extract_title(filename, text):
    """
    从文件名或文本中提取标题信息
    """
    # 首先尝试从文件名获取标题（去掉扩展名）
    title = os.path.splitext(filename)[0]
    
    # 如果文本不为空，尝试从文本开头获取标题
    if text.strip():
        # 尝试获取第一行作为标题
        first_line = text.strip().split('\n')[0].strip()
        # 如果第一行不是页码或目录等无关信息，可以作为标题
        if first_line and not re.match(r'^第\s*\d+\s*页|目录|……', first_line):
            # 可以选择使用第一行作为标题，这里我们保留文件名作为标题
            # 但你可以根据需要取消下面一行的注释来使用第一行作为标题
            # title = first_line
            pass
            
    return title

def clean_text(t):
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"第\s*\d+\s*页|目录|……", " ", t)
    return t.strip()

def split_sections(t):
    # 先按"一级标题"粗切，再细分
    parts = header_pat.split(t)
    if len(parts) <= 1:  # 没识别到标题，退化为整体
        return [t]
    merged = []
    cur = ""
    for p in parts:
        if header_pat.match(p):  # 是标题标记
            if cur: merged.append(cur)
            cur = p
        else:
            cur += p
    if cur: merged.append(cur)
    return merged

def chunk_by_tokens(text, size=512, overlap=96):
    # 简单按句切 + 滑窗拼块
    sents = re.split(r"[。！？；]\s*", text)
    blocks, buf = [], []
    tok_count = 0
    for s in sents:
        if not s: continue
        piece = " ".join(jieba.cut(s))
        n = len(piece.split())
        if tok_count + n > size and buf:
            blocks.append("".join(buf))
            # overlap：把末尾约 overlap 的词留下来
            keep = " ".join(" ".join(buf).split()[-overlap:])
            buf = [keep, piece]
            tok_count = len(keep.split()) + n
        else:
            buf.append(s + "。")
            tok_count += n
    if buf: blocks.append("".join(buf))
    return blocks

def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as w:
        for fn in os.listdir(RAW_DIR):
            path = os.path.join(RAW_DIR, fn)
            if not os.path.isfile(path): continue
            text = clean_text(read_any(path))
            
            # 提取文档标题
            title = extract_title(fn, text)
            
            secs = split_sections(text)
            for si, sec in enumerate(secs):
                chunks = chunk_by_tokens(sec, size=512, overlap=96)
                for ci, ch in enumerate(chunks):
                    # 在chunk内容前添加标题信息
                    chunk_with_title = f"文档标题：{title}\n\n{ch}"
                    
                    meta = {"doc": fn, "section_id": si, "chunk_id": ci}
                    w.write(json.dumps({"text": chunk_with_title, "meta": meta}, ensure_ascii=False) + "\n")
    print("Done ->", OUT_PATH)

if __name__ == "__main__":
    main()