"""
语义关系感知的检索流程
核心思想：理解query的语义关系，而非文字匹配
"""
from typing import List, Dict
from semantic_relation_extractor import (
    get_relation_extractor,
    get_relation_verifier
)


def semantic_relation_search_pipeline(
    chunks: List[Dict],
    query: str,
    enable_relation_extraction: bool = True,
    enable_relation_verification: bool = True,
    verification_threshold: float = 0.6,
    max_verify: int = 15,
    top_k: int = 5
) -> Dict:
    """
    语义关系感知的检索流程
    
    流程：
    1. 提取query中的语义关系
    2. 用抽象查询重新检索（可选）
    3. 用LLM验证chunk是否包含目标关系
    4. 返回验证通过的chunks
    
    Args:
        chunks: 初步召回的chunks
        query: 用户问题
        enable_relation_extraction: 是否启用关系提取
        enable_relation_verification: 是否启用关系验证
        verification_threshold: 验证置信度阈值
        max_verify: 最多验证数量
        top_k: 最终返回数量
    
    Returns:
        {
            'chunks': 处理后的chunks,
            'relation_info': 关系信息,
            'verified_count': 验证通过数量
        }
    """
    if not chunks:
        return {
            'chunks': [],
            'relation_info': {},
            'verified_count': 0
        }
    
    # 1. 提取语义关系
    relation_info = {}
    if enable_relation_extraction:
        print("\n🧠 提取语义关系...")
        extractor = get_relation_extractor()
        relation_info = extractor.extract_relation(query)
        
        print(f"   关系类型: {relation_info.get('relation_type', 'unknown')}")
        print(f"   主体: {relation_info.get('subject', 'unknown')}")
        print(f"   客体: {relation_info.get('object', 'unknown')}")
        print(f"   抽象查询: {relation_info.get('abstract_query', '')[:50]}...")
    
    # 2. 关系验证
    verified_chunks = chunks
    verified_count = 0
    
    if enable_relation_verification and relation_info.get('relation_type') != 'general':
        verifier = get_relation_verifier()
        verified_chunks = verifier.batch_verify(
            chunks=chunks,
            relation_info=relation_info,
            threshold=verification_threshold,
            max_verify=max_verify
        )
        verified_count = len(verified_chunks)
        
        # 如果验证通过的太少，补充一些原始chunks
        if len(verified_chunks) < top_k and len(chunks) > len(verified_chunks):
            print(f"   ⚠️ 验证通过的chunk较少，补充原始结果")
            # 添加未验证的chunks
            verified_ids = {c['chunk_id'] for c in verified_chunks}
            for chunk in chunks:
                if chunk['chunk_id'] not in verified_ids:
                    chunk['relation_verified'] = False
                    verified_chunks.append(chunk)
                    if len(verified_chunks) >= top_k * 2:
                        break
    
    # 3. 排序：验证通过的优先
    def sort_key(chunk):
        """排序键"""
        # 验证通过的优先
        if chunk.get('relation_verified', False):
            confidence = chunk.get('relation_confidence', 0.5)
            return (1, confidence, -chunk.get('intent_score', 0.5))
        else:
            return (0, 0, -chunk.get('intent_score', 0.5))
    
    verified_chunks.sort(key=sort_key, reverse=True)
    
    # 4. 取TOP-K
    final_chunks = verified_chunks[:top_k]
    
    return {
        'chunks': final_chunks,
        'relation_info': relation_info,
        'verified_count': verified_count
    }


def enhance_answer_with_relations(chunks: List[Dict], relation_info: Dict) -> str:
    """
    基于语义关系增强答案生成
    
    Args:
        chunks: 验证通过的chunks
        relation_info: 关系信息
    
    Returns:
        增强的prompt提示
    """
    relation_type = relation_info.get('relation_type', '')
    obj = relation_info.get('object', '')
    
    # 提取验证通过的关系信息
    relations = []
    for chunk in chunks:
        if chunk.get('relation_verified', False):
            extracted = chunk.get('extracted_relation', '')
            if extracted:
                relations.append(extracted)
    
    if not relations:
        return ""
    
    # 构建增强提示
    enhancement = f"\n\n【语义关系分析】\n"
    enhancement += f"问题关注的关系类型：{relation_type}\n"
    enhancement += f"目标对象：{obj}\n"
    enhancement += f"识别到的关系：\n"
    for i, rel in enumerate(relations, 1):
        enhancement += f"{i}. {rel}\n"
    
    return enhancement
