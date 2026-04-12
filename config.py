# ===================== RAG 全局配置 =====================
# 导入文本类型配置
from text_type_config import get_config

# 已标注的JSONL文件路径（动态获取，根据文本类型自动切换）
def get_annotated_jsonl():
    """获取当前配置的数据文件路径"""
    return get_config().get_data_file()

ANNOTATED_JSONL = get_annotated_jsonl()

# 向量库持久化存储文件夹（动态获取，根据文本类型自动切换）
def get_chroma_persist_dir():
    """获取当前配置的向量库路径"""
    return get_config().get_vector_db_path()

CHROMA_PERSIST_DIR = get_chroma_persist_dir()

# 中文轻量级Embedding模型（与LangChain嵌入函数完美适配）
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"

# 检索核心参数
TOP_K = 10  # 最终返回的高价值片段数量（减少到5，因为有rerank）
FILTER_IMPORTANCE = ["high"]  # 仅保留高重要度片段，提升回答精度

# 向量库集合名称（唯一标识，不可重复）
COLLECTION_NAME = "novel_rag"

# ===================== 优化参数 =====================
# 相似度阈值（距离分数，越小越相似，建议0.5-0.7）
DISTANCE_THRESHOLD = 0.6

# 是否启用Reranker（强烈推荐）
ENABLE_RERANKER = True

# 是否启用BM25混合检索（推荐）
ENABLE_BM25 = True

# 混合检索权重
VECTOR_WEIGHT = 0.7  # 向量检索权重
BM25_WEIGHT = 0.3    # BM25检索权重

# 是否启用Query扩展（可选，会增加响应时间）
ENABLE_QUERY_EXPANSION = False

# 证据最大token数（避免超长context）
MAX_EVIDENCE_TOKENS = 2000

# ===================== 意图感知优化 =====================
# 是否启用意图感知检索（强烈推荐）
ENABLE_INTENT_AWARE = True

# 意图感知配置
INTENT_TYPE_FILTER = True  # 是否根据意图过滤chunk类型
INTENT_QUERY_ENHANCEMENT = False  # 是否根据意图增强查询

# ===================== 语义关系感知优化 =====================
# 是否启用语义关系提取（深度理解query）
ENABLE_SEMANTIC_RELATION = True

# 是否启用语义关系验证（用LLM验证chunk是否包含目标关系）
ENABLE_RELATION_VERIFICATION = True

# 关系验证配置
RELATION_VERIFICATION_THRESHOLD = 0.6  # 验证置信度阈值
MAX_RELATION_VERIFY = 15  # 最多验证数量（避免调用太多次LLM）

# ===================== 文本类型配置 =====================
# 文本类型：支持长文本（小说）、短文本（聊天记录）和工业说明书
# 可选值: "novel" (长文本/小说) / "chat" (短文本/聊天记录) / "manual" (工业说明书)
TEXT_TYPE = "chat"  # 默认为聊天记录模式

# ===================== 短文本处理配置 =====================
# 话题分段配置
TOPIC_SEGMENT_METHOD = "hybrid"  # "vector" / "llm" / "hybrid"
VECTOR_SIMILARITY_THRESHOLD = 0.7  # 向量相似度阈值（低于此值认为话题转换）
TIME_GAP_THRESHOLD = 300  # 时间间隔阈值（秒），超过此值强制切分
MIN_SEGMENT_LENGTH = 15  # 最小段落长度（字符数）
MAX_SEGMENT_LENGTH = 200  # 最大段落长度（字符数）

# 滑动窗口配置
CONTEXT_WINDOW_SIZE = 2  # 前后各保留N个话题段落
MAX_CONTEXT_CHARS = 150  # 上下文最大字符数
INCLUDE_TOPIC_LABEL = True  # 是否包含话题标签

# 向量化策略
VECTORIZE_STRATEGY = "full"  # "core" / "full" / "dual"

# ===================== TXT文件上传配置 =====================
# 短对话TXT文件处理配置
TXT_LINE_SEPARATOR = "\n"  # TXT文件行分隔符
TXT_MIN_LINE_LENGTH = 2  # 最小行长度（字符数），低于此值的行会被过滤
TXT_ENCODING_PRIORITY = ["utf-8", "gbk", "gb2312"]  # 编码尝试顺序

# 说话人格式支持
ENABLE_SPEAKER_DETECTION = True  # 是否自动检测说话人格式（格式：说话人: 文本）
CLEAN_EMOJIS = True  # 是否清理表情符号（如 [旺柴]、[微笑]）
SPEAKER_SEPARATOR = [":", "："]  # 说话人分隔符（英文冒号、中文冒号）

# 自动标注配置（用于TXT文件上传后的自动标注）
AUTO_ANNOTATE_BATCH_SIZE = 50  # 自动标注批量大小
AUTO_ANNOTATE_ENABLED = True  # 是否启用自动标注