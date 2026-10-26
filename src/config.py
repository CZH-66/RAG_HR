import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# 数据与存储
DATA_FILE = BASE_DIR / "data" / "hr_dataset.json"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "hr_policies"

# 模型：英文数据 -> 英文 Embedding 模型
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "deepseek-chat"                      # 按你实际使用的模型改
LLM_BASE_URL = "https://api.deepseek.com/v1"     # 按你实际使用的服务改
API_KEY = os.getenv("API_KEY")

# 检索
SIMILARITY_TOP_K = 4