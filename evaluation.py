"""
RAG召回评估模块
提供召回率、精准率、相关性评分等量化指标
"""
import json
from typing import List, Dict, Any
from llm import ask_llm


def print_retrieved_chunks(chunks: List[Dict], query: str):
    """
    打印召回的chunk详细信息
    """
    print("\n" + "="*80)
    print(f"📊 召回结果分析 | 查询: {query}")
    print("="*80)
    
    if not chunks:
        print("⚠️ 未召回任何chunk")
        return
    
    print(f"\n✅ 共召回 {len(chunks)} 个chunk:\n")
    
    for idx, chunk in enumerate(chunks, 1):
        print(f"【Chunk {idx}】")
        print(f"  ID: {chunk.get('chunk_id', 'N/A')}")
        print(f"  重要度: {chunk.get('importance', 'N/A')}")
        
        # 安全处理types（可能是字符串或列表）
        types = chunk.get('types', 'N/A')
        if isinstance(types, list):
            types = ', '.join(types)
        print(f"  类型: {types}")
        
        # 安全处理characters（可能是字符串或列表）
        characters = chunk.get('characters', 'N/A')
        if isinstance(characters, list):
            characters = ', '.join(characters)
        print(f"  角色: {characters}")
        
        # 安全处理距离分数
        distance = chunk.get('distance', 'N/A')
        if isinstance(distance, (int, float)):
            print(f"  距离分数: {distance:.4f}")
        else:
            print(f"  距离分数: {distance}")
        
        # 安全处理rerank分数
        rerank_score = chunk.get('rerank_score')
        if rerank_score is not None and isinstance(rerank_score, (int, float)):
            print(f"  Rerank分数: {rerank_score:.4f}")
        
        print(f"  摘要: {chunk.get('summary', 'N/A')}")
        print(f"  内容预览: {chunk.get('content', '')[:100]}...")
        print("-" * 80)


def calculate_recall_precision(
    retrieved_chunks: List[Dict],
    ground_truth_ids: List[int] = None,
    relevance_threshold: float = 0.7
) -> Dict[str, float]:
    """
    计算召回率和精准率
    
    Args:
        retrieved_chunks: 召回的chunk列表
        ground_truth_ids: 真实相关的chunk_id列表（如果有标注数据）
        relevance_threshold: 相关性阈值
    
    Returns:
        包含召回率、精准率、F1分数的字典
    """
    if not retrieved_chunks:
        return {
            "recall": 0.0,
            "precision": 0.0,
            "f1_score": 0.0,
            "retrieved_count": 0,
            "relevant_count": 0
        }
    
    retrieved_ids = [chunk.get('chunk_id') for chunk in retrieved_chunks]
    
    # 如果有ground truth标注
    if ground_truth_ids:
        relevant_retrieved = set(retrieved_ids) & set(ground_truth_ids)
        
        recall = len(relevant_retrieved) / len(ground_truth_ids) if ground_truth_ids else 0
        precision = len(relevant_retrieved) / len(retrieved_ids) if retrieved_ids else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "recall": recall,
            "precision": precision,
            "f1_score": f1_score,
            "retrieved_count": len(retrieved_ids),
            "relevant_count": len(relevant_retrieved),
            "ground_truth_count": len(ground_truth_ids)
        }
    
    # 如果没有ground truth，基于距离分数估算
    else:
        # 假设距离分数越小越相关（cosine distance）
        avg_distance = sum(chunk.get('distance', 1.0) for chunk in retrieved_chunks) / len(retrieved_chunks)
        estimated_precision = max(0, 1 - avg_distance)  # 简单估算
        
        return {
            "estimated_precision": estimated_precision,
            "avg_distance": avg_distance,
            "retrieved_count": len(retrieved_ids),
            "note": "无ground truth，基于距离分数估算"
        }


def evaluate_answer_relevance(question: str, answer: str, evidence: str) -> Dict[str, Any]:
    """
    使用LLM评估答案的相关性和质量
    
    Args:
        question: 用户问题
        answer: 模型回答
        evidence: 使用的证据
    
    Returns:
        评估结果字典
    """
    eval_prompt = f"""请评估以下RAG系统的回答质量，给出1-5分评分和理由。

评分标准：
5分：完全回答问题，逻辑清晰，证据充分
4分：基本回答问题，有少量不足
3分：部分回答问题，存在明显不足
2分：勉强相关，回答质量差
1分：完全不相关或错误

问题：{question}

证据：
{evidence[:500]}...

回答：
{answer}

请按以下JSON格式输出（只输出JSON，不要其他内容）：
{{
    "score": 评分(1-5),
    "relevance": "相关性评价",
    "completeness": "完整性评价",
    "reasoning": "评分理由"
}}
"""
    
    try:
        eval_result = ask_llm(eval_prompt)
        # 尝试解析JSON
        eval_result = eval_result.strip()
        if eval_result.startswith("```json"):
            eval_result = eval_result.split("```json")[1].split("```")[0].strip()
        elif eval_result.startswith("```"):
            eval_result = eval_result.split("```")[1].split("```")[0].strip()
        
        result = json.loads(eval_result)
        return result
    except Exception as e:
        return {
            "score": 0,
            "relevance": "评估失败",
            "completeness": "评估失败",
            "reasoning": f"评估过程出错: {str(e)}",
            "error": str(e)
        }


def print_evaluation_report(
    query: str,
    retrieved_chunks: List[Dict],
    answer: str,
    evidence: str,
    ground_truth_ids: List[int] = None,
    enable_llm_eval: bool = True
):
    """
    打印完整的评估报告
    """
    print("\n" + "="*80)
    print("📈 RAG系统评估报告")
    print("="*80)
    
    # 1. 召回指标
    print("\n【召回指标】")
    metrics = calculate_recall_precision(retrieved_chunks, ground_truth_ids)
    
    if ground_truth_ids:
        print(f"  召回率 (Recall): {metrics['recall']:.2%}")
        print(f"  精准率 (Precision): {metrics['precision']:.2%}")
        print(f"  F1分数: {metrics['f1_score']:.4f}")
        print(f"  召回数量: {metrics['relevant_count']}/{metrics['ground_truth_count']}")
    else:
        print(f"  估算精准率: {metrics.get('estimated_precision', 0):.2%}")
        print(f"  平均距离分数: {metrics.get('avg_distance', 0):.4f}")
        print(f"  召回数量: {metrics['retrieved_count']}")
        print(f"  ⚠️ {metrics.get('note', '')}")
    
    # 2. LLM评估答案质量
    if enable_llm_eval:
        print("\n【答案质量评估】")
        print("⏳ 正在使用LLM评估答案质量...")
        eval_result = evaluate_answer_relevance(query, answer, evidence)
        
        print(f"  综合评分: {eval_result.get('score', 0)}/5")
        print(f"  相关性: {eval_result.get('relevance', 'N/A')}")
        print(f"  完整性: {eval_result.get('completeness', 'N/A')}")
        print(f"  评分理由: {eval_result.get('reasoning', 'N/A')}")
    
    print("\n" + "="*80)


def load_ground_truth(ground_truth_file: str = None) -> Dict[str, List[int]]:
    """
    加载ground truth标注数据（如果有的话）
    
    格式示例：
    {
        "云澈为什么要去天玄大陆": [1, 5, 12, 23],
        "夏倾月的身份是什么": [45, 67, 89]
    }
    """
    if not ground_truth_file:
        return {}
    
    try:
        with open(ground_truth_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ 加载ground truth失败: {e}")
        return {}
