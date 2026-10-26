import json, random
import os
from pathlib import Path

random.seed(42)  # 固定种子，保证每次抽的是同一批

# 锚定到项目根目录，与运行时工作目录无关
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "hr_dataset.json"
EVAL_DIR = BASE_DIR / "data" / "eval"

with open(DATA_FILE, encoding="utf-8") as f:
    data = json.load(f)

sample = random.sample(data, 40)

os.makedirs(EVAL_DIR, exist_ok=True)

with open(EVAL_DIR / "eval_set.jsonl", "w", encoding="utf-8") as f:
    for row in sample:
        f.write(json.dumps({
            "question": row["instruction"],
            "ground_truth": row["response"]
        }, ensure_ascii=False) + "\n")

print(f"已生成 {len(sample)} 条评估样本 -> {EVAL_DIR / 'eval_set.jsonl'}")
