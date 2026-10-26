import json
from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

from config import DATA_FILE, CHROMA_DIR, COLLECTION_NAME, EMBED_MODEL

# 1. 配置 Embedding（首次运行会下载约 130MB 模型）
Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)

# 2. 读取本地 JSON（已确认是 list）
with open(DATA_FILE, "r", encoding="utf-8") as f:
    raw = json.load(f)

# 3. 字段映射：instruction / response
documents = []
for i, row in enumerate(raw):
    instruction = row.get("instruction", "").strip()
    response = row.get("response", "").strip()
    if not instruction or not response:
        continue

    text = f"Question: {instruction}\nAnswer: {response}"
    metadata = {"id": i, "source": "hr_dataset.json"}
    documents.append(Document(text=text, metadata=metadata))

print(f"共构造 {len(documents)} 条 Document")
print(f"示例：{documents[0].text[:120]}...")

# 4. 写入 Chroma（每次重建，避免重复运行导致数据翻倍）
chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
if COLLECTION_NAME in [c.name for c in chroma_client.list_collections()]:
    chroma_client.delete_collection(COLLECTION_NAME)
chroma_collection = chroma_client.create_collection(COLLECTION_NAME)

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

VectorStoreIndex.from_documents(documents, storage_context=storage_context)
print(f"✅ 索引构建完成，已写入 {CHROMA_DIR}")