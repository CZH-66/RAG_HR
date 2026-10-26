import os

# 必须在导入 llama_index / huggingface 之前设置；模型已缓存，直接离线模式跳过联网检查
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import json
import sys
import numpy as np
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike
import chromadb

from config import (
    BASE_DIR, CHROMA_DIR, COLLECTION_NAME, EMBED_MODEL,
    LLM_MODEL, LLM_BASE_URL, API_KEY, SIMILARITY_TOP_K,
)

Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
Settings.llm = OpenAILike(
    model=LLM_MODEL, api_key=API_KEY, api_base=LLM_BASE_URL,
    temperature=0, is_chat_model=True,
    is_function_calling_model=False, context_window=64000,
)

chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = chroma_client.get_collection(COLLECTION_NAME)
vector_store = ChromaVectorStore(chroma_collection=collection)
index = VectorStoreIndex.from_vector_store(vector_store, embed_model=Settings.embed_model)
query_engine = index.as_query_engine(similarity_top_k=SIMILARITY_TOP_K)

# 语义相似度：用同一个 Embedding 模型算余弦相似度
def cosine_sim(text_a: str, text_b: str) -> float:
    embs = Settings.embed_model.get_text_embedding_batch([text_a, text_b])
    va, vb = np.array(embs[0]), np.array(embs[1])
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)

# 选择评估集：默认基准集；`python evaluate.py real` 跑口语化真实问法集
SET_NAME = sys.argv[1] if len(sys.argv) > 1 else "baseline"
if SET_NAME == "real":
    EVAL_FILE = BASE_DIR / "data" / "eval" / "eval_real.jsonl"
    QUESTION_FIELD = "question_real"
    RESULT_FILE = BASE_DIR / "data" / "eval" / "results_real.json"
else:
    EVAL_FILE = BASE_DIR / "data" / "eval" / "eval_set.jsonl"
    QUESTION_FIELD = "question"
    RESULT_FILE = BASE_DIR / "data" / "eval" / "results_baseline.json"

with open(EVAL_FILE, encoding="utf-8") as f:
    eval_set = [json.loads(line) for line in f]

SIM_THRESHOLD = 0.8  # 相似度 ≥ 0.8 视为"答案正确"（可按观测分布调整）

hits = 0
passed = 0
sims = []
results = []
for i, item in enumerate(eval_set, 1):
    q = item[QUESTION_FIELD]
    gt = item["ground_truth"]
    response = query_engine.query(q)
    answer = str(response)

    # 检索命中判断：ground_truth 的原文是否出现在某个检索片段中
    retrieved_texts = [n.text for n in response.source_nodes]
    hit = any(gt[:80] in t for t in retrieved_texts)  # 用前 80 字符做近似匹配
    hits += int(hit)

    # 答案语义相似度：answer vs ground_truth
    sim = cosine_sim(answer, gt)
    sims.append(sim)
    ok = sim >= SIM_THRESHOLD
    passed += int(ok)

    results.append({
        "question": q,
        "question_original": item.get("question"),
        "answer": answer,
        "ground_truth": gt,
        "hit": hit,
        "answer_sim": round(sim, 4),
        "answer_passed": ok,
        "top_scores": [round(n.score, 4) if n.score else None for n in response.source_nodes],
    })
    print(f"[{i}/{len(eval_set)}] hit={hit}  sim={sim:.3f}  {q[:50]}...")

mean_sim = sum(sims) / len(sims)
print(f"\n评估集: {SET_NAME} ({EVAL_FILE.name}, {len(eval_set)} 条)")
print(f"检索命中率: {hits}/{len(eval_set)} = {hits/len(eval_set):.1%}")
print(f"答案语义相似度: 平均 {mean_sim:.3f} | 最小 {min(sims):.3f} | 最大 {max(sims):.3f}")
print(f"答案通过率(相似度>={SIM_THRESHOLD}): {passed}/{len(eval_set)} = {passed/len(eval_set):.1%}")

with open(RESULT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"结果已写入 {RESULT_FILE}")
