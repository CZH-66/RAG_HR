import os

# 必须在导入 llama_index / huggingface 之前设置
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from llama_index.core import VectorStoreIndex, Settings
from llama_index.core import PromptTemplate
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import CondenseQuestionChatEngine
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
    model=LLM_MODEL, api_key=API_KEY, api_base=LLM_BASE_URL,
    temperature=0, is_chat_model=True,
    is_function_calling_model=False, context_window=64000,
)

# 3. 连接已有集合，加载索引
chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
chroma_collection = chroma_client.get_collection(COLLECTION_NAME)
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
index = VectorStoreIndex.from_vector_store(vector_store, embed_model=Settings.embed_model)

# 4. 基础查询引擎（单轮，与 query.py 保持一致）
qa_prompt = PromptTemplate(
    """You are an HR policy assistant. Answer the question using ONLY the provided context.
Follow these rules strictly:
1. If the context does not explicitly state the answer, respond ONLY with: "I cannot answer this based on the available HR policies." Do not speculate, infer, or combine partial information.
2. Do NOT add information that the is not in the context.
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

# 5. 追问改写提示词：把依赖上下文的追问改写成独立问题
condense_prompt = PromptTemplate(
    """Given a conversation between a user and an HR assistant, and a follow-up message from the user,
rewrite the follow-up message as a standalone question that captures all necessary context from the conversation.

Rules:
1. The rewritten question must be fully understandable without reading the conversation history.
2. If the follow-up is already a standalone question, return it unchanged.
3. Do NOT answer the question, only rewrite it.
4. Keep the same language as the follow-up message.

Conversation:
{chat_history}

Follow-up: {question}
Standalone question:"""
)

# 6. 多轮会话引擎：维护历史 + 检索前先改写
memory = ChatMemoryBuffer.from_defaults(token_limit=4000)

chat_engine = CondenseQuestionChatEngine.from_defaults(
    query_engine=query_engine,
    memory=memory,
    condense_prompt=condense_prompt,
    verbose=True,  # 打印改写后的独立问题，便于调试
)


def reset_session():
    """清空会话历史（API 层新会话时调用）"""
    chat_engine.reset()


if __name__ == "__main__":
    print("HR 多轮问答（输入 exit 退出 / reset 重置会话）")
    print("=" * 60)
    while True:
        try:
            q = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q.lower() == "exit":
            break
        if q.lower() == "reset":
            reset_session()
            print("[会话已重置]")
            continue

        response = chat_engine.chat(q)
        print(f"HR 助手: {response}")
        print("-" * 60)
        print("检索来源：")
        for i, node in enumerate(response.source_nodes, 1):
            score = node.score if node.score is not None else float("nan")
            print(f"  [{i}] score={score:.4f}")
            print(f"      {node.text[:100]}...")