import os

# 必须在导入 llama_index / huggingface 之前设置
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from llama_index.core import VectorStoreIndex, Settings
from llama_index.core import PromptTemplate
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike
import chromadb

from config import (
    CHROMA_DIR, COLLECTION_NAME, EMBED_MODEL,
    LLM_MODEL, LLM_BASE_URL, API_KEY, SIMILARITY_TOP_K,
)

# 1. Embedding 必须与 ingest 时一致
Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)

# 2. LLM
Settings.llm = OpenAILike(
    model=LLM_MODEL,          # "deepseek-chat"
    api_key=API_KEY,
    api_base=LLM_BASE_URL,    # "https://api.deepseek.com/v1"
    temperature=0,
    is_chat_model=True,       # 关键：声明这是聊天模型
    is_function_calling_model=False,  # DeepSeek 目前不支持函数调用
    context_window=64000,     # 手动指定上下文窗口，避免 LlamaIndex 再次推断失败
)

# 3. 连接已有集合，加载索引（不重新算 Embedding）
chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
chroma_collection = chroma_client.get_collection(COLLECTION_NAME)
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

index = VectorStoreIndex.from_vector_store(
    vector_store, embed_model=Settings.embed_model
)

# 4. 查询引擎（带约束的 QA 提示词）
qa_prompt = PromptTemplate(
    """You are an HR policy assistant. Answer the question using ONLY the provided context.
Follow these rules strictly:
1. If the context does not explicitly state the answer, respond ONLY with: "I cannot answer this based on the available HR policies." Do not speculate, infer, or combine partial information.
2. Do NOT add information that is not in the context.
3. Preserve all specific numbers, deadlines, portal names, and action steps mentioned in the context.
4. Be concise but complete — do not omit procedural instructions.

Context:
{context_str}

Question: {query_str}
Answer:"""
)

query_engine = index.as_query_engine(
    similarity_top_k=SIMILARITY_TOP_K,
    text_qa_template=qa_prompt,
)

if __name__ == "__main__":
    question = "What is the CEO's annual salary?"
    response = query_engine.query(question)

    print("=" * 60)
    print(f"Q: {question}")
    print("-" * 60)
    print(f"A: {response}")
    print("-" * 60)
    print("检索来源：")
    for i, node in enumerate(response.source_nodes, 1):
        score = node.score if node.score is not None else float("nan")
        print(f"  [{i}] score={score:.4f}")
        print(f"      {node.text[:100]}...")