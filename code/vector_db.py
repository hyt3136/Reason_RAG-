# 仅删除无用的try-except兼容代码，其余完全和你写的一样
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from config import (
    EMBEDDING_MODEL, CHROMA_PERSIST_DIR,
    COLLECTION_NAME, FILTER_IMPORTANCE, TOP_K
)

# ===================== 核心优化：支持GPU编码 + 自动释放显存 =====================
def get_embedding_function(model_name: str = None, device: str = "cuda"):
    """
    获取embedding函数
    
    Args:
        model_name: 模型名称（可选，默认使用config中的值）
        device: 设备选择 "cuda" 或 "cpu"（默认cuda，如果不可用自动降级到cpu）
    """
    _model_name = model_name or EMBEDDING_MODEL
    
    # 自动检测GPU可用性
    import torch
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠️  GPU不可用，降级到CPU")
        device = "cpu"
    
    return HuggingFaceEmbeddings(
        model_name=_model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )

def build_embedding_text(chunk: dict) -> str:
    char_str = " ".join(chunk["characters"])
    kw_str = " ".join(chunk["keywords"])
    summary = chunk["summary"]
    content = chunk["content"][:300]
    return f"{char_str} {kw_str} {summary} {content}"

def init_vector_db(chunks: list, persist_dir: str = None, collection_name: str = None, embedding_model: str = None, use_gpu: bool = True):
    """
    初始化向量数据库（支持持久化，避免重复编码）
    
    Args:
        chunks: chunk列表
        persist_dir: 持久化目录（可选，默认使用config中的值）
        collection_name: 集合名称（可选，默认使用config中的值）
        embedding_model: embedding模型（可选，默认使用config中的值）
        use_gpu: 是否使用GPU编码（默认True，编码完成后自动释放显存）
    """
    # 使用传入的参数或默认值
    _persist_dir = persist_dir or CHROMA_PERSIST_DIR
    _collection_name = collection_name or COLLECTION_NAME
    _embedding_model = embedding_model or EMBEDDING_MODEL
    
    device = "cuda" if use_gpu else "cpu"
    
    # 先用CPU模式快速检查向量库是否存在（避免加载GPU模型）
    try:
        import os
        db_path = os.path.join(_persist_dir, "chroma.sqlite3")
        if os.path.exists(db_path):
            # 数据库文件存在，用CPU模式快速加载检查
            temp_embed_func = get_embedding_function(_embedding_model, device="cpu")
            temp_store = Chroma(
                collection_name=_collection_name,
                embedding_function=temp_embed_func,
                persist_directory=_persist_dir,
            )
            existing_count = temp_store._collection.count()
            
            if existing_count > 0:
                print(f"✅ 向量库已存在 | 已有 {existing_count} 条数据，直接加载（无需编码）")
                return temp_store
    except Exception as e:
        print(f"   ℹ️  向量库检查失败，将重新构建: {e}")
    
    # 向量库不存在或为空，需要编码 - 此时才加载GPU模型
    print(f"🔄 向量库为空，开始编码 {len(chunks)} 个chunks...")
    if use_gpu:
        print(f"🚀 使用GPU加速编码...")
    else:
        print(f"⚠️  使用CPU编码，可能需要几分钟...")
    
    embed_func = get_embedding_function(_embedding_model, device=device)
    vector_store = Chroma(
        collection_name=_collection_name,
        embedding_function=embed_func,
        persist_directory=_persist_dir,
    )
    
    documents = []
    ids = []
    metadatas = []
    for chunk in chunks:
        documents.append(build_embedding_text(chunk))
        ids.append(str(chunk["chunk_id"]))
        metadatas.append({
            "chunk_id": chunk["chunk_id"],
            "characters": ",".join(chunk["characters"]),
            "types": ",".join(chunk["types"]),
            "importance": chunk["importance"],
            "summary": chunk["summary"],
            "content": chunk["content"]
        })

    vector_store.add_texts(texts=documents, ids=ids, metadatas=metadatas)
    print(f"✅ 向量库构建完成 | 入库 {len(chunks)} 条数据")
    
    # 编码完成后释放GPU显存
    if use_gpu:
        _release_gpu_memory()
        print(f"✅ GPU显存已释放，可用于后续LLM推理")
    
    return vector_store


def _release_gpu_memory():
    """释放GPU显存"""
    try:
        import torch
        import gc
        
        # 清理Python垃圾回收
        gc.collect()
        
        # 清空CUDA缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print(f"   💾 GPU显存已清理")
    except Exception as e:
        print(f"   ⚠️  显存清理失败: {e}")

def search_vector_db(vector_store: Chroma, rewrite_query: str, distance_threshold: float = 0.6):
    """
    向量检索（带相似度阈值过滤）
    
    Args:
        vector_store: 向量数据库
        rewrite_query: 改写后的查询
        distance_threshold: 距离阈值，超过此值的结果会被过滤
    
    Returns:
        过滤后的chunk列表
    """
    # 召回更多候选，为后续rerank做准备
    results = vector_store.similarity_search_with_score(query=rewrite_query, k=TOP_K * 3)
    valid_chunks = []
    
    for doc, score in results:
        meta = doc.metadata
        # 过滤：重要度 + 相似度阈值
        if meta["importance"] in FILTER_IMPORTANCE and score <= distance_threshold:
            meta["distance"] = score
            valid_chunks.append(meta)
    
    # 按距离排序
    return sorted(valid_chunks, key=lambda x: x["distance"])[:TOP_K * 2]  # 返回2倍，供rerank使用


def hybrid_search(vector_store: Chroma, bm25_retriever, query: str, 
                  vector_weight: float = 0.7, bm25_weight: float = 0.3,
                  distance_threshold: float = 0.6):
    """
    混合检索：向量检索 + BM25关键词检索
    使用RRF (Reciprocal Rank Fusion) 融合结果
    
    Args:
        vector_store: 向量数据库
        bm25_retriever: BM25检索器
        query: 查询文本
        vector_weight: 向量检索权重
        bm25_weight: BM25检索权重
        distance_threshold: 向量检索的距离阈值
    
    Returns:
        融合后的chunk列表
    """
    # 1. 向量检索
    vector_results = search_vector_db(vector_store, query, distance_threshold)
    
    # 2. BM25检索
    bm25_results = []
    if bm25_retriever:
        bm25_results = bm25_retriever.search(query, top_k=TOP_K * 2)
    
    # 3. RRF融合
    chunk_scores = {}
    k = 60  # RRF参数
    
    # 向量检索结果
    for rank, chunk in enumerate(vector_results, 1):
        chunk_id = chunk['chunk_id']
        rrf_score = vector_weight / (k + rank)
        if chunk_id not in chunk_scores:
            chunk_scores[chunk_id] = {'chunk': chunk, 'score': 0}
        chunk_scores[chunk_id]['score'] += rrf_score
    
    # BM25检索结果
    for rank, chunk in enumerate(bm25_results, 1):
        chunk_id = chunk['chunk_id']
        rrf_score = bm25_weight / (k + rank)
        if chunk_id not in chunk_scores:
            chunk_scores[chunk_id] = {'chunk': chunk, 'score': 0}
        chunk_scores[chunk_id]['score'] += rrf_score
        # 保留BM25分数
        if 'bm25_score' in chunk:
            chunk_scores[chunk_id]['chunk']['bm25_score'] = chunk['bm25_score']
    
    # 4. 排序并返回
    sorted_results = sorted(chunk_scores.values(), key=lambda x: x['score'], reverse=True)
    return [item['chunk'] for item in sorted_results[:TOP_K * 2]]
