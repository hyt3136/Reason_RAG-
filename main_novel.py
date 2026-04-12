"""
小说RAG系统 - 专门用于处理长文本小说内容
"""
# 原生导入
from llm import ask_llm
from data_loader import load_annotated_chunks
from vector_db import init_vector_db, search_vector_db, hybrid_search
from utils import query_rewrite, query_expansion, build_rag_prompt
from evaluation import print_retrieved_chunks, print_evaluation_report, load_ground_truth
from prompts import struct_evidence_by_type_novel

# 小说专用配置
ANNOTATED_JSONL = r"D:\rag\venv\逆天邪神_自动标注版_最终.jsonl"
CHROMA_PERSIST_DIR = "./chroma_novel_db"
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
COLLECTION_NAME = "novel_rag"

# 检索参数
TOP_K = 10
DISTANCE_THRESHOLD = 0.6
ENABLE_RERANKER = True
ENABLE_BM25 = True
VECTOR_WEIGHT = 0.7
BM25_WEIGHT = 0.3
ENABLE_QUERY_EXPANSION = False
MAX_EVIDENCE_TOKENS = 2000

# 小说特定优化
ENABLE_INTENT_AWARE = True
ENABLE_SEMANTIC_RELATION = True
ENABLE_RELATION_VERIFICATION = True  # 使用LLM验证语义关系（不是向量模型）
RELATION_VERIFICATION_THRESHOLD = 0.6
MAX_RELATION_VERIFY = 15  # LLM验证次数

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

def get_rag_answer_novel(question: str, vector_store, all_chunks, enable_eval: bool = True):
    """
    小说RAG问答流程
    
    优化点：
    1. Query改写
    2. 混合检索（向量+BM25）
    3. Reranker重排序
    4. 意图感知优化（人物/剧情/动机/关系查询）
    5. 语义关系验证
    """
    try:
        # 1. Query处理
        if ENABLE_QUERY_EXPANSION:
            print("\n🔄 Query扩展中...")
            queries = query_expansion(question, num_variants=2)
            print(f"✨ 生成查询变体: {queries}")
        else:
            rewritten_question = query_rewrite(question)
            queries = [rewritten_question]
        
        # 2. 检索
        all_retrieved_chunks = []
        seen_ids = set()
        
        for query in queries:
            if ENABLE_BM25:
                bm25 = get_bm25(all_chunks)
                chunks = hybrid_search(
                    vector_store, bm25, query,
                    vector_weight=VECTOR_WEIGHT,
                    bm25_weight=BM25_WEIGHT,
                    distance_threshold=DISTANCE_THRESHOLD
                )
                print(f"🔍 混合检索召回: {len(chunks)} 个候选")
            else:
                chunks = search_vector_db(vector_store, query, DISTANCE_THRESHOLD)
                print(f"🔍 向量检索召回: {len(chunks)} 个候选")
            
            for chunk in chunks:
                if chunk['chunk_id'] not in seen_ids:
                    all_retrieved_chunks.append(chunk)
                    seen_ids.add(chunk['chunk_id'])
        
        # 3. Reranker重排序
        if ENABLE_RERANKER and all_retrieved_chunks:
            print(f"🔄 Reranker重排序中...")
            reranker = get_reranker()
            if reranker:
                all_retrieved_chunks = reranker.rerank(question, all_retrieved_chunks, top_n=TOP_K * 2)
                print(f"✅ 重排序完成，保留Top-{len(all_retrieved_chunks)}")
        else:
            all_retrieved_chunks = all_retrieved_chunks[:TOP_K * 2]
        
        # 4. 意图感知优化（小说专用）
        if ENABLE_INTENT_AWARE:
            print(f"\n🎯 小说意图感知优化中...")
            from intent_aware_search import intent_aware_search_pipeline
            intent_result = intent_aware_search_pipeline(
                chunks=all_retrieved_chunks,
                query=question,
                top_k=TOP_K * 2,
                enable_type_filter=True,
                enable_query_enhancement=False
            )
            all_retrieved_chunks = intent_result['chunks']
        
        # 5. 语义关系验证（小说专用）
        if ENABLE_SEMANTIC_RELATION:
            from semantic_aware_search import semantic_relation_search_pipeline
            semantic_result = semantic_relation_search_pipeline(
                chunks=all_retrieved_chunks,
                query=question,
                enable_relation_extraction=True,
                enable_relation_verification=ENABLE_RELATION_VERIFICATION,
                verification_threshold=RELATION_VERIFICATION_THRESHOLD,
                max_verify=MAX_RELATION_VERIFY,
                top_k=TOP_K
            )
            all_retrieved_chunks = semantic_result['chunks']
            if semantic_result['verified_count'] > 0:
                print(f"   ✅ {semantic_result['verified_count']} 个chunk通过语义关系验证")
        else:
            all_retrieved_chunks = all_retrieved_chunks[:TOP_K]
        
        # 6. 打印召回结果
        if enable_eval:
            print_retrieved_chunks(all_retrieved_chunks, question)
        
        # 7. 构建小说专用证据
        evidence = struct_evidence_by_type_novel(all_retrieved_chunks, max_tokens=MAX_EVIDENCE_TOKENS)
        
        # 8. 生成答案
        prompt = build_rag_prompt(evidence, question)
        answer = ask_llm(prompt)
        
        # 9. 评估
        if enable_eval:
            print_evaluation_report(
                query=question,
                retrieved_chunks=all_retrieved_chunks,
                answer=answer,
                evidence=evidence,
                ground_truth_ids=None,
                enable_llm_eval=True
            )
        
        return {
            "answer": answer,
            "chunks": all_retrieved_chunks,
            "evidence": evidence,
            "queries": queries
        }
    except Exception as e:
        import traceback
        print(f"⚠️ RAG执行出错: {e}")
        print(traceback.format_exc())
        return {
            "answer": f"⚠️ 执行出错: {e}",
            "chunks": [],
            "evidence": "",
            "queries": [question]
        }

if __name__ == "__main__":
    print("=" * 80)
    print("📚 小说RAG系统")
    print("=" * 80)
    print("✨ 优化特性:")
    print(f"  - Reranker重排序: {'✅' if ENABLE_RERANKER else '❌'}")
    print(f"  - BM25混合检索: {'✅' if ENABLE_BM25 else '❌'}")
    print(f"  - Query扩展: {'✅' if ENABLE_QUERY_EXPANSION else '❌'}")
    print(f"  - 意图感知优化: {'✅' if ENABLE_INTENT_AWARE else '❌'}")
    print(f"  - 语义关系验证: {'✅' if ENABLE_SEMANTIC_RELATION else '❌'}")
    print(f"  - 相似度阈值: {DISTANCE_THRESHOLD}")
    print(f"  - 最终返回数: TOP-{TOP_K}")
    print("=" * 80)

    # 加载数据
    print("\n🔄 加载小说标注数据...")
    all_chunks = load_annotated_chunks(ANNOTATED_JSONL)
    print(f"✅ 加载完成，共 {len(all_chunks)} 个chunk")

    # 初始化向量库
    print("\n🔄 初始化向量库...")
    vector_store = init_vector_db(all_chunks, CHROMA_PERSIST_DIR, COLLECTION_NAME, EMBEDDING_MODEL, use_gpu=True)
    
    # 初始化BM25
    if ENABLE_BM25:
        print("\n🔄 初始化BM25检索器...")
        get_bm25(all_chunks)
    
    # 预加载Reranker
    if ENABLE_RERANKER:
        print("\n🔄 预加载Reranker模型...")
        get_reranker()
    
    print("\n" + "=" * 80)
    print("✅ 小说RAG系统就绪！")
    print("=" * 80)
    print("\n💡 使用提示：")
    print("  - 直接输入问题进行查询")
    print("  - 输入 'eval on' 开启评估模式（默认开启）")
    print("  - 输入 'eval off' 关闭评估模式")
    print("  - 输入 'quit' 退出\n")

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
        
        result = get_rag_answer_novel(
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
