import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from llama_index.core import VectorStoreIndex, Settings, PromptTemplate
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

# ---------- 全局初始化（进程内只加载一次） ----------
Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
Settings.llm = OpenAILike(
    model=LLM_MODEL, api_key=API_KEY, api_base=LLM_BASE_URL,
    temperature=0, is_chat_model=True,
    is_function_calling_model=False, context_window=64000,
)

chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
chroma_collection = chroma_client.get_collection(COLLECTION_NAME)
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
index = VectorStoreIndex.from_vector_store(vector_store, embed_model=Settings.embed_model)

# ---------- 回答语言（zh=中文 / en=英文） ----------
LANGUAGE_NAMES = {
    "zh": "Simplified Chinese (简体中文)",
    "en": "English",
}
REFUSAL_TEXT = {
    "zh": "根据现有 HR 政策，无法回答该问题。",
    "en": "I cannot answer this based on the available HR policies.",
}
DEFAULT_LANG = "zh"

# 提示词版本号：改提示词规则时递增，便于日志里对比效果差异
PROMPT_VERSION = "qa_v2_multilang"


def normalize_lang(lang: str | None) -> str:
    return lang if lang in LANGUAGE_NAMES else DEFAULT_LANG


def build_qa_prompt(lang: str = DEFAULT_LANG) -> PromptTemplate:
    """按回答语言生成 QA 提示词：检索上下文仍为英文原文。"""
    lang = normalize_lang(lang)
    return PromptTemplate(
        f"""You are an HR policy assistant. Answer the question using ONLY the provided context.
Follow these rules strictly:
1. If the context does not explicitly state the answer, respond with exactly this sentence and nothing else: "{REFUSAL_TEXT[lang]}"
2. Do NOT add information that is not in the context.
3. Preserve all specific numbers, deadlines, portal names, and action steps exactly as written; do not translate proper nouns.
4. Be concise but complete — do not omit procedural instructions.
5. Write the entire answer in {LANGUAGE_NAMES[lang]}.

Context:
{{context_str}}

Question: {{query_str}}
Answer:"""
    )


_QUERY_ENGINES: dict[str, object] = {}


def get_query_engine(lang: str = DEFAULT_LANG):
    """按语言缓存查询引擎，避免每次请求重建。"""
    lang = normalize_lang(lang)
    if lang not in _QUERY_ENGINES:
        _QUERY_ENGINES[lang] = index.as_query_engine(
            similarity_top_k=SIMILARITY_TOP_K,
            text_qa_template=build_qa_prompt(lang),
        )
    return _QUERY_ENGINES[lang]


# 兼容旧引用：默认中文回答
query_engine = get_query_engine(DEFAULT_LANG)

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


def create_chat_engine(lang: str = DEFAULT_LANG) -> CondenseQuestionChatEngine:
    """创建一个带独立会话历史的多轮引擎（每个用户会话、每种语言各一个）。"""
    return RecordingCondenseQuestionChatEngine.from_defaults(
        query_engine=get_query_engine(lang),
        memory=ChatMemoryBuffer.from_defaults(token_limit=4000),
        condense_prompt=condense_prompt,
    )


class RecordingCondenseQuestionChatEngine(CondenseQuestionChatEngine):
    """在改写步骤完成后记录 standalone question，供 API 写日志复盘。

    追问改写是"检索前"的独立 LLM 调用，改写结果直接决定召回质量，
    但 LlamaIndex 默认不保存它，只能靠 verbose 打印。
    """

    last_condensed_question: str | None = None

    def _condense_question(self, *args, **kwargs):
        question = super()._condense_question(*args, **kwargs)
        self.last_condensed_question = question
        return question
