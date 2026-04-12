# 原生导入，无任何报错！
from llm import ask_llm
from data_loader import load_annotated_chunks
from vector_db import init_vector_db, search_vector_db, hybrid_search
from utils import query_rewrite, query_expansion, struct_evidence_by_type, build_rag_prompt
from evaluation import (
    print_retrieved_chunks,
    print_evaluation_report,
    load_ground_truth
)
from config import (
    ENABLE_RERANKER, ENABLE_BM25, ENABLE_QUERY_EXPANSION,
    DISTANCE_THRESHOLD, VECTOR_WEIGHT, BM25_WEIGHT, MAX_EVIDENCE_TOKENS, TOP_K,
    ENABLE_SEMANTIC_RELATION, ENABLE_RELATION_VERIFICATION,
    RELATION_VERIFICATION_THRESHOLD, MAX_RELATION_VERIFY
)
from intent_aware_search import intent_aware_search_pipeline
from semantic_aware_search import semantic_relation_search_pipeline, enhance_answer_with_relations

# 可选模块（懒加载）
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

try:
    from langchain_core.runnables import RunnableLambda  # type: ignore[import]
except ImportError:
    try:
        # Fallback for older LangChain versions
        from langchain.schema.runnable import RunnableLambda  # type: ignore[import]
    except ImportError as e:
        raise ImportError(
            "无法导入 RunnableLambda，请确保已安装兼容版本的 langchain-core 或 langchain。"
        ) from e

# ==================== 优化的RAG流程 ====================
def get_rag_answer_with_eval(question: str, vector_store, all_chunks, 
                              enable_eval: bool = True, ground_truth_ids: list = None):
    """
    执行优化后的RAG问答流程
    
    优化点：
    1. Query改写/扩展
    2. 混合检索（向量+BM25）
    3. Reranker重排序
    4. 意图感知优化（类型过滤、意图识别）
    5. 语义关系验证（深度理解query语义）
    6. 改进的证据组织
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
        
        # 2. 检索（支持多query合并）
        all_retrieved_chunks = []
        seen_ids = set()
        
        for query in queries:
            if ENABLE_BM25:
                # 混合检索
                bm25 = get_bm25(all_chunks)
                chunks = hybrid_search(
                    vector_store, bm25, query,
                    vector_weight=VECTOR_WEIGHT,
                    bm25_weight=BM25_WEIGHT,
                    distance_threshold=DISTANCE_THRESHOLD
                )
                print(f"🔍 混合检索召回: {len(chunks)} 个候选")
            else:
                # 纯向量检索
                chunks = search_vector_db(vector_store, query, DISTANCE_THRESHOLD)
                print(f"🔍 向量检索召回: {len(chunks)} 个候选")
            
            # 合并去重
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
            # 不使用reranker时，直接取TOP_K
            all_retrieved_chunks = all_retrieved_chunks[:TOP_K * 2]
        
        # 4. 意图感知的二次排序和过滤
        print(f"\n🎯 意图感知优化中...")
        intent_result = intent_aware_search_pipeline(
            chunks=all_retrieved_chunks,
            query=question,
            top_k=TOP_K * 2,  # 先取2倍，供语义关系验证
            enable_type_filter=True,
            enable_query_enhancement=False
        )
        
        all_retrieved_chunks = intent_result['chunks']
        intent_info = intent_result['intent_info']
        
        # 5. 语义关系感知优化（可选）
        relation_info = {}
        if ENABLE_SEMANTIC_RELATION:
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
            relation_info = semantic_result['relation_info']
            
            if semantic_result['verified_count'] > 0:
                print(f"   ✅ {semantic_result['verified_count']} 个chunk通过语义关系验证")
        else:
            # 不使用语义关系验证时，直接取TOP_K
            all_retrieved_chunks = all_retrieved_chunks[:TOP_K]
        
        # 6. 打印召回结果
        print_retrieved_chunks(all_retrieved_chunks, question)
        
        # 7. 构建证据
        evidence = struct_evidence_by_type(all_retrieved_chunks, max_tokens=MAX_EVIDENCE_TOKENS)
        
        # 8. 如果有语义关系信息，增强证据
        if relation_info:
            relation_enhancement = enhance_answer_with_relations(all_retrieved_chunks, relation_info)
            if relation_enhancement:
                evidence += relation_enhancement
        
        # 9. 生成答案
        prompt = build_rag_prompt(evidence, question)
        answer = ask_llm(prompt)
        
        # 10. 评估（可选）
        if enable_eval:
            print_evaluation_report(
                query=question,
                retrieved_chunks=all_retrieved_chunks,
                answer=answer,
                evidence=evidence,
                ground_truth_ids=ground_truth_ids,
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

# ==================== 主程序 ====================
if __name__ == "__main__":
    print("=" * 80)
    print("🚀 【优化版】LangChain RAG系统")
    print("=" * 80)
    print("✨ 优化特性:")
    print(f"  - Reranker重排序: {'✅ 已启用' if ENABLE_RERANKER else '❌ 未启用'}")
    print(f"  - BM25混合检索: {'✅ 已启用' if ENABLE_BM25 else '❌ 未启用'}")
    print(f"  - Query扩展: {'✅ 已启用' if ENABLE_QUERY_EXPANSION else '❌ 未启用'}")
    print(f"  - 意图感知优化: ✅ 已启用")
    print(f"  - 语义关系验证: {'✅ 已启用' if ENABLE_SEMANTIC_RELATION else '❌ 未启用'}")
    print(f"  - 相似度阈值: {DISTANCE_THRESHOLD}")
    print(f"  - 最终返回数: TOP-{TOP_K}")
    print("=" * 80)

    # 加载数据
    print("\n🔄 加载标注数据...")
    all_chunks = load_annotated_chunks()
    print(f"✅ 加载完成，共 {len(all_chunks)} 个chunk")

    # 初始化向量库
    print("\n🔄 初始化向量库...")
    vector_store = init_vector_db(all_chunks, use_gpu=True)
    
    # 初始化BM25（如果启用）
    if ENABLE_BM25:
        print("\n🔄 初始化BM25检索器...")
        get_bm25(all_chunks)
    
    # 预加载Reranker（如果启用）
    if ENABLE_RERANKER:
        print("\n🔄 预加载Reranker模型...")
        get_reranker()
    
    # 加载ground truth（可选）
    ground_truth_data = load_ground_truth()
    
    print("\n" + "=" * 80)
    print("✅ RAG 系统就绪！")
    print("=" * 80)
    print("\n💡 使用提示：")
    print("  - 直接输入问题进行查询")
    print("  - 输入 'eval on' 开启评估模式（默认开启）")
    print("  - 输入 'eval off' 关闭评估模式")
    print("  - 输入 'config' 查看当前配置")
    print("  - 输入 'quit' 退出\n")

    # 评估模式开关
    enable_evaluation = True

    # 交互循环
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
        
        if user_question.lower() == "config":
            print("\n📋 当前配置:")
            print(f"  TOP_K: {TOP_K}")
            print(f"  DISTANCE_THRESHOLD: {DISTANCE_THRESHOLD}")
            print(f"  ENABLE_RERANKER: {ENABLE_RERANKER}")
            print(f"  ENABLE_BM25: {ENABLE_BM25}")
            print(f"  ENABLE_QUERY_EXPANSION: {ENABLE_QUERY_EXPANSION}")
            print(f"  VECTOR_WEIGHT: {VECTOR_WEIGHT}")
            print(f"  BM25_WEIGHT: {BM25_WEIGHT}")
            print(f"  MAX_EVIDENCE_TOKENS: {MAX_EVIDENCE_TOKENS}\n")
            continue
        
        if not user_question:
            print("⚠️ 请输入有效问题")
            continue

        print("\n" + "="*80)
        print("⏳ 处理中...")
        print("="*80)
        
        # 获取ground truth（如果有）
        gt_ids = ground_truth_data.get(user_question, None)
        
        # 执行RAG并评估
        result = get_rag_answer_with_eval(
            user_question, 
            vector_store,
            all_chunks,
            enable_eval=enable_evaluation,
            ground_truth_ids=gt_ids
        )
        
        print("\n" + "="*80)
        print("💡 最终回答:")
        print("="*80)
        print(f"{result['answer']}\n")
        print("="*80)