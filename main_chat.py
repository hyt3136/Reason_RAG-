"""
短对话RAG系统 - 专门用于处理聊天记录和对话内容
"""
# 原生导入
from llm import ask_llm
from data_loader import load_annotated_chunks
from vector_db import init_vector_db, search_vector_db, hybrid_search
from utils import query_rewrite, build_rag_prompt
from evaluation import print_retrieved_chunks
from prompts import struct_evidence_by_type_chat

# 短对话专用配置
ANNOTATED_JSONL = r"D:\rag\venv\对话记录_标注版.jsonl"
CHROMA_PERSIST_DIR = "./chroma_chat_db"
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
COLLECTION_NAME = "chat_rag"

# 检索参数（短对话优化）
TOP_K = 15  # 短对话需要更多上下文
DISTANCE_THRESHOLD = 0.5  # 更宽松的阈值
ENABLE_RERANKER = True
ENABLE_BM25 = True
VECTOR_WEIGHT = 0.6  # 短对话更依赖关键词
BM25_WEIGHT = 0.4
MAX_EVIDENCE_TOKENS = 2500  # 短对话需要更多上下文

# 短对话特定优化
ENABLE_SPEAKER_FILTER = True  # 说话人过滤
ENABLE_EMOTION_ANALYSIS = True  # 情感分析
ENABLE_TOPIC_GROUPING = True  # 话题分组

# 懒加载模块
_reranker = None
_bm25_retriever = None

def get_reranker():
    """懒加载Reranker"""
    global _reranker
    if _reranker is None and ENABLE_RERANKER:
        from reranker import get_reranker as load_reranker
        _reranker = load_reranker()
    return _reranker

def get_bm25(chunks):
    """懒加载BM25"""
    global _bm25_retriever
    if _bm25_retriever is None and ENABLE_BM25:
        from bm25_retriever import get_bm25_retriever
        _bm25_retriever = get_bm25_retriever(chunks)
    return _bm25_retriever

def filter_by_speaker(chunks, query):
    """根据查询中的说话人过滤chunk"""
    # 简单实现：如果query中提到某个说话人，优先返回包含该说话人的chunk
    # 可以根据需要扩展
    return chunks

def group_by_topic(chunks):
    """按话题分组chunk"""
    # 简单实现：按topic_id分组
    topic_groups = {}
    for chunk in chunks:
        topic_id = chunk.get('topic_id', 0)
        if topic_id not in topic_groups:
            topic_groups[topic_id] = []
        topic_groups[topic_id].append(chunk)
    return topic_groups

def get_rag_answer_chat(question: str, vector_store, all_chunks, enable_eval: bool = True):
    """
    短对话RAG问答流程
    
    优化点：
    1. Query改写（针对对话场景）
    2. 混合检索（向量+BM25，更重视关键词）
    3. Reranker重排序
    4. 说话人过滤
    5. 话题分组
    6. 情感分析
    """
    try:
        # 1. Query处理（短对话专用）
        rewritten_question = query_rewrite(question)
        print(f"📝 原始问题：{question}")
        print(f"✨ 改写后：{rewritten_question}")
        
        # 2. 检索
        if ENABLE_BM25:
            bm25 = get_bm25(all_chunks)
            chunks = hybrid_search(
                vector_store, bm25, rewritten_question,
                vector_weight=VECTOR_WEIGHT,
                bm25_weight=BM25_WEIGHT,
                distance_threshold=DISTANCE_THRESHOLD
            )
            print(f"🔍 混合检索召回: {len(chunks)} 个候选")
        else:
            chunks = search_vector_db(vector_store, rewritten_question, DISTANCE_THRESHOLD)
            print(f"🔍 向量检索召回: {len(chunks)} 个候选")
        
        # 3. Reranker重排序
        if ENABLE_RERANKER and chunks:
            print(f"🔄 Reranker重排序中...")
            reranker = get_reranker()
            if reranker:
                chunks = reranker.rerank(question, chunks, top_n=TOP_K * 2)
                print(f"✅ 重排序完成，保留Top-{len(chunks)}")
        else:
            chunks = chunks[:TOP_K * 2]
        
        # 4. 说话人过滤（短对话专用）
        if ENABLE_SPEAKER_FILTER:
            chunks = filter_by_speaker(chunks, question)
        
        # 5. 话题分组（短对话专用）
        if ENABLE_TOPIC_GROUPING:
            print(f"\n🎯 话题分组中...")
            topic_groups = group_by_topic(chunks)
            print(f"   发现 {len(topic_groups)} 个话题")
            
            # 从每个话题中选择最相关的chunk
            selected_chunks = []
            for topic_id, topic_chunks in sorted(topic_groups.items()):
                # 每个话题最多取3个chunk
                selected_chunks.extend(topic_chunks[:3])
            
            chunks = selected_chunks[:TOP_K]
        else:
            chunks = chunks[:TOP_K]
        
        # 6. 打印召回结果
        if enable_eval:
            print_retrieved_chunks(chunks, question)
        
        # 7. 构建短对话专用证据
        evidence = struct_evidence_by_type_chat(chunks, max_tokens=MAX_EVIDENCE_TOKENS)
        
        # 8. 生成答案（短对话专用prompt）
        prompt = f"""你是一个专业的对话分析助手。请根据以下聊天记录片段回答问题。

聊天记录：
{evidence}

问题：{question}

请注意：
1. 分析对话中的情感态度和语气
2. 注意说话人之间的互动关系
3. 理解对话的上下文和隐含意图
4. 如果信息不足，说明需要更多上下文

回答："""
        
        answer = ask_llm(prompt)
        
        return {
            "answer": answer,
            "chunks": chunks,
            "evidence": evidence,
            "query": rewritten_question
        }
    except Exception as e:
        import traceback
        print(f"⚠️ RAG执行出错: {e}")
        print(traceback.format_exc())
        return {
            "answer": f"⚠️ 执行出错: {e}",
            "chunks": [],
            "evidence": "",
            "query": question
        }

if __name__ == "__main__":
    print("=" * 80)
    print("💬 短对话RAG系统")
    print("=" * 80)
    print("✨ 优化特性:")
    print(f"  - Reranker重排序: {'✅' if ENABLE_RERANKER else '❌'}")
    print(f"  - BM25混合检索: {'✅' if ENABLE_BM25 else '❌'}")
    print(f"  - 说话人过滤: {'✅' if ENABLE_SPEAKER_FILTER else '❌'}")
    print(f"  - 话题分组: {'✅' if ENABLE_TOPIC_GROUPING else '❌'}")
    print(f"  - 情感分析: {'✅' if ENABLE_EMOTION_ANALYSIS else '❌'}")
    print(f"  - 相似度阈值: {DISTANCE_THRESHOLD}")
    print(f"  - 最终返回数: TOP-{TOP_K}")
    print("=" * 80)

    # 加载数据
    print("\n🔄 加载对话标注数据...")
    all_chunks = load_annotated_chunks(ANNOTATED_JSONL)
    print(f"✅ 加载完成，共 {len(all_chunks)} 个chunk")

    # 初始化向量库
    print("\n🔄 初始化向量库...")
    vector_store = init_vector_db(all_chunks, CHROMA_PERSIST_DIR, COLLECTION_NAME, EMBEDDING_MODEL)
    
    # 初始化BM25
    if ENABLE_BM25:
        print("\n🔄 初始化BM25检索器...")
        get_bm25(all_chunks)
    
    # 预加载Reranker
    if ENABLE_RERANKER:
        print("\n🔄 预加载Reranker模型...")
        get_reranker()
    
    print("\n" + "=" * 80)
    print("✅ 短对话RAG系统就绪！")
    print("=" * 80)
    print("\n💡 使用提示：")
    print("  - 直接输入问题进行查询")
    print("  - 输入 'eval on' 开启评估模式（默认开启）")
    print("  - 输入 'eval off' 关闭评估模式")
    print("  - 输入 'quit' 退出\n")
    print("💡 示例问题：")
    print("  - 她喜欢我吗")
    print("  - 他们在讨论什么")
    print("  - 谁的情绪比较激动\n")

    enable_evaluation = True

    while True:
        user_question = input("❓ 请输入你的问题：").strip()
        
        if user_question.lower() == "quit":
            print("\n👋 已退出")
            break
        
        if user_question.lower() == "eval on":
            enable_evaluation = True
            print("✅ 评估模式已开启")
            continue
        
        if user_question.lower() == "eval off":
            enable_evaluation = False
            print("✅ 评估模式已关闭")
            continue
        
        if not user_question:
            print("⚠️ 请输入有效问题")
            continue

        print("\n" + "="*80)
        print("⏳ 处理中...")
        print("="*80)
        
        result = get_rag_answer_chat(
            user_question, 
            vector_store,
            all_chunks,
            enable_eval=enable_evaluation
        )
        
        print("\n" + "="*80)
        print("💡 最终回答:")
        print("="*80)
        print(f"{result['answer']}\n")
        print("="*80)
