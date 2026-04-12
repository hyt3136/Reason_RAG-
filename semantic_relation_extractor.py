"""
语义关系提取器
从query中提取抽象的语义关系，而非具体词汇
"""
from llm import ask_llm
from prompts import (
    build_semantic_relation_extraction_prompt,
    build_semantic_relation_matching_prompt
)
import json
import re


class SemanticRelationExtractor:
    """语义关系提取器"""
    
    def extract_relation(self, query: str) -> dict:
        """
        从query中提取语义关系
        
        Args:
            query: 用户问题
        
        Returns:
            {
                'relation_type': str,  # 关系类型
                'entities': list,  # 相关实体
                'description': str  # 关系描述
            }
        """
        try:
            prompt = build_semantic_relation_extraction_prompt(query)
            response = ask_llm(prompt)
            # 提取JSON
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return result
        except Exception as e:
            print(f"  ⚠️ 语义关系提取失败: {e}")
        
        # 默认返回
        return {
            'relation_type': 'general',
            'subject': 'unknown',
            'object': 'unknown',
            'abstract_query': query,
            'search_keywords': []
        }
    
    def build_abstract_query(self, relation_info: dict) -> str:
        """
        构建抽象查询
        
        Args:
            relation_info: 关系信息
        
        Returns:
            抽象查询字符串
        """
        relation_type = relation_info.get('relation_type', '')
        obj = relation_info.get('object', '')
        keywords = relation_info.get('search_keywords', [])
        
        # 构建查询
        query_parts = [obj]
        
        # 添加关系相关的关键词
        if keywords:
            query_parts.extend(keywords[:5])  # 最多5个关键词
        
        return ' '.join(query_parts)


class SemanticRelationVerifier:
    """语义关系验证器"""
    
    def verify_chunk_relation(self, chunk: dict, relation_info: dict) -> dict:
        """
        验证chunk是否包含目标语义关系
        
        Args:
            chunk: chunk字典
            relation_info: 关系信息
        
        Returns:
            {
                'has_relation': bool,
                'confidence': float,
                'extracted_info': str,
                'reason': str
            }
        """
        relation_type = relation_info.get('relation_type', '')
        obj = relation_info.get('object', '')
        content = chunk.get('content', '')
        
        prompt = f"""判断以下文本片段是否包含目标语义关系。

目标关系类型：{relation_type}
目标客体：{obj}
抽象查询：{relation_info.get('abstract_query', '')}

文本片段：
{content[:300]}

判断标准：
- 如果是"爱慕关系"，需要包含：某角色对{obj}产生爱慕、好感、喜欢、爱、倾慕等正向情感
- 不要被"喜欢"这个词限制，要理解语义：如"心动"、"在意"、"舍不得"、"想念"等都是爱慕的表现
        """
        relation_type = relation_info.get('relation_type', '')
        entities = relation_info.get('entities', [])
        description = relation_info.get('description', '')
        content = chunk.get('content', '')
        
        try:
            prompt = build_semantic_relation_matching_prompt(
                relation_type, entities, description, content
            )
            response = ask_llm(prompt)
            # 提取JSON
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return result
        except Exception as e:
            print(f"  ⚠️ 关系验证失败: {e}")
        
        return {
            'has_relation': False,
            'confidence': 0.0,
            'extracted_info': '',
            'reason': '验证失败'
        }
    
    def batch_verify(self, chunks: list, relation_info: dict, 
                     threshold: float = 0.6, max_verify: int = 20) -> list:
        """
        批量验证chunks
        
        Args:
            chunks: chunk列表
            relation_info: 关系信息
            threshold: 置信度阈值
            max_verify: 最多验证数量（避免调用太多次LLM）
        
        Returns:
            验证通过的chunks
        """
        verified_chunks = []
        
        print(f"\n🔍 语义关系验证中...")
        print(f"   目标关系: {relation_info.get('relation_type', 'unknown')}")
        print(f"   验证数量: 最多{min(len(chunks), max_verify)}个chunk")
        
        for i, chunk in enumerate(chunks[:max_verify]):
            if i > 0 and i % 5 == 0:
                print(f"   已验证 {i}/{min(len(chunks), max_verify)} 个...")
            
            verification = self.verify_chunk_relation(chunk, relation_info)
            
            if verification['has_relation'] and verification['confidence'] >= threshold:
                chunk['relation_verified'] = True
                chunk['relation_confidence'] = verification['confidence']
                chunk['extracted_relation'] = verification['extracted_info']
                verified_chunks.append(chunk)
        
        print(f"   ✅ 验证完成，{len(verified_chunks)}/{min(len(chunks), max_verify)} 个chunk包含目标关系")
        
        return verified_chunks


# 全局单例
_extractor_instance = None
_verifier_instance = None

def get_relation_extractor():
    """获取全局关系提取器"""
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = SemanticRelationExtractor()
    return _extractor_instance

def get_relation_verifier():
    """获取全局关系验证器"""
    global _verifier_instance
    if _verifier_instance is None:
        _verifier_instance = SemanticRelationVerifier()
    return _verifier_instance
