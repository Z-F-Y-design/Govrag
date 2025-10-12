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

def clean_text(t):
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"第\s*\d+\s*页|目录|……", " ", t)
    return t.strip()

def split_sections(t):
    # 先按“一级标题”粗切，再细分
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
            secs = split_sections(text)
            for si, sec in enumerate(secs):
                chunks = chunk_by_tokens(sec, size=512, overlap=96)
                for ci, ch in enumerate(chunks):
                    meta = {"doc": fn, "section_id": si, "chunk_id": ci}
                    w.write(json.dumps({"text": ch, "meta": meta}, ensure_ascii=False) + "\n")
    print("Done ->", OUT_PATH)

if __name__ == "__main__":
    main()