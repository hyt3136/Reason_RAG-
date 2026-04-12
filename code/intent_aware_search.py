"""
意图感知的检索模块
根据问题意图优化检索策略
"""
from typing import List, Dict
from intent_classifier import get_intent_classifier


def score_chunk_by_intent(chunk: Dict, intent_info: Dict) -> float:
    """
    根据意图给chunk打分
    
    分数越低越好（因为distance越小越相似）
    
    Args:
        chunk: chunk字典
        intent_info: 意图信息
    
    Returns:
        综合分数
    """
    # 基础分数（distance或默认值）
    base_score = chunk.get('distance', 0.5)
    if base_score == 'N/A':
        base_score = 0.5
    
    # Rerank分数（如果有）
    rerank_score = chunk.get('rerank_score')
    if rerank_score is not None:
        # Rerank分数越高越好，转换为越小越好
        base_score = 1.0 - (rerank_score / 10.0)  # 假设rerank分数0-10
    
    # 类型匹配加分
    preferred_types = intent_info.get('preferred_types', [])
    chunk_types = chunk.get('types', '')
    if isinstance(chunk_types, list):
        chunk_types = ','.join(chunk_types)
    
    type_match_count = 0
    for ptype in preferred_types:
        if ptype in chunk_types:
            type_match_count += 1
    
    # 每匹配一个类型，分数降低0.1（更优先）
    type_boost = type_match_count * 0.1
    
    # 摘要质量加分
    summary_boost = 0
    if intent_info.get('boost_summary', False):
        summary = chunk.get('summary', '')
        if len(summary) > 15:  # 摘要足够详细
            summary_boost = 0.05
        if len(summary) > 25:  # 非常详细
            summary_boost = 0.08
    
    # 重要度加分
    importance = chunk.get('importance', 'middle')
    importance_boost = {
        'high': 0.1,
        'middle': 0.05,
        'low': 0.0
    }.get(importance, 0.0)
    
    # 综合分数
    final_score = base_score - type_boost - summary_boost - importance_boost
    
    return max(final_score, 0.0)  # 确保非负


def intent_aware_rerank(chunks: List[Dict], query: str, intent_info: Dict) -> List[Dict]:
    """
    基于意图重新排序chunks
    
    Args:
        chunks: chunk列表
        query: 查询文本
        intent_info: 意图信息
    
    Returns:
        重排序后的chunks
    """
    if not chunks:
        return []
    
    # 给每个chunk打分
    for chunk in chunks:
        chunk['intent_score'] = score_chunk_by_intent(chunk, intent_info)
    
    # 按意图分数排序（分数越小越好）
    sorted_chunks = sorted(chunks, key=lambda x: x['intent_score'])
    
    return sorted_chunks


def filter_by_type(chunks: List[Dict], preferred_types: List[str], 
                   min_match: int = 1) -> List[Dict]:
    """
    按类型过滤chunks
    
    Args:
        chunks: chunk列表
        preferred_types: 优先类型列表
        min_match: 最少匹配数量
    
    Returns:
        过滤后的chunks
    """
    filtered = []
    
    for chunk in chunks:
        chunk_types = chunk.get('types', '')
        if isinstance(chunk_types, list):
            chunk_types = ','.join(chunk_types)
        
        match_count = sum(1 for ptype in preferred_types if ptype in chunk_types)
        
        if match_count >= min_match:
            filtered.append(chunk)
    
    return filtered


def enhance_query_by_intent(query: str, intent_info: Dict) -> str:
    """
    根据意图增强查询
    
    Args:
        query: 原始查询
        intent_info: 意图信息
    
    Returns:
        增强后的查询
    """
    intent = intent_info.get('intent', 'general')
    
    # 根据不同意图添加关键词
    enhancements = {
        'character_analysis': ' 性格 心理 动机 行为',
        'plot_query': ' 事件 经过 发生',
        'causality': ' 原因 导致 因为',
        'relationship': ' 关系 感情 态度',
        'motivation': ' 动机 目的 想法'
    }
    
    enhancement = enhancements.get(intent, '')
    
    # 避免重复添加
    for word in enhancement.split():
        if word not in query:
            query += f" {word}"
    
    return query.strip()


def intent_aware_search_pipeline(chunks: List[Dict], query: str, 
                                  top_k: int = 5, 
                                  enable_type_filter: bool = True,
                                  enable_query_enhancement: bool = False) -> Dict:
    """
    完整的意图感知检索流程
    
    Args:
        chunks: 召回的chunks
        query: 查询文本
        top_k: 返回数量
        enable_type_filter: 是否启用类型过滤
        enable_query_enhancement: 是否启用查询增强
    
    Returns:
        {
            'chunks': 处理后的chunks,
            'intent_info': 意图信息,
            'enhanced_query': 增强后的查询（如果启用）
        }
    """
    # 1. 意图分类
    classifier = get_intent_classifier()
    intent_info = classifier.classify(query)
    
    print(f"\n🎯 问题意图: {intent_info['description']} (置信度: {intent_info['confidence']:.2f})")
    print(f"   优先类型: {', '.join(intent_info['preferred_types'])}")
    
    # 2. 查询增强（可选）
    enhanced_query = query
    if enable_query_enhancement:
        enhanced_query = enhance_query_by_intent(query, intent_info)
        if enhanced_query != query:
            print(f"   增强查询: {enhanced_query}")
    
    # 3. 类型过滤（可选）
    if enable_type_filter and intent_info['confidence'] > 0.6:
        original_count = len(chunks)
        chunks = filter_by_type(chunks, intent_info['preferred_types'], min_match=1)
        if len(chunks) < top_k and original_count > 0:
            # 如果过滤后太少，放宽条件
            print(f"   ⚠️ 类型过滤后只剩{len(chunks)}个，使用原始结果")
            chunks = chunks  # 保持过滤后的结果，但会在后续补充
        else:
            print(f"   类型过滤: {original_count} → {len(chunks)} 个chunk")
    
    # 4. 意图感知重排序
    chunks = intent_aware_rerank(chunks, query, intent_info)
    
    # 5. 取TOP-K
    final_chunks = chunks[:top_k]
    
    return {
        'chunks': final_chunks,
        'intent_info': intent_info,
        'enhanced_query': enhanced_query
    }
