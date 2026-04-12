"""
问题意图分类器
识别用户问题的意图，用于优化检索策略
支持长文本（小说）和短文本（聊天记录）两种模式
"""
import re
import json
from typing import Dict, List
from llm import ask_llm
from prompts import build_intent_classification_prompt
from text_type_config import get_config


class IntentClassifier:
    """问题意图分类器"""
    
    def __init__(self):
        # 从配置获取意图模式
        self.update_intent_patterns()
    
    def update_intent_patterns(self):
        """更新意图模式（根据当前文本类型）"""
        config = get_config()
        self.intent_patterns = config.get_intent_patterns()
    
    def classify(self, query: str) -> Dict:
        """
        分类问题意图
        
        Args:
            query: 用户问题
        
        Returns:
            {
                'intent': str,  # 意图类型
                'preferred_types': List[str],  # 优先的chunk类型
                'boost_summary': bool,  # 是否优先摘要丰富的chunk
                'confidence': float,  # 置信度
                'description': str  # 意图描述
            }
        """
        query_lower = query.lower()
        
        # 匹配意图
        matched_intents = []
        
        for intent, config in self.intent_patterns.items():
            match_count = 0
            
            for keyword in config['keywords']:
                # 支持正则匹配
                if '*' in keyword or '.' in keyword:
                    if re.search(keyword, query):
                        match_count += 2  # 正则匹配权重更高
                else:
                    if keyword in query:
                        match_count += 1
            
            if match_count > 0:
                matched_intents.append({
                    'intent': intent,
                    'score': match_count,
                    'config': config
                })
        
        # 选择得分最高的意图
        if matched_intents:
            matched_intents.sort(key=lambda x: x['score'], reverse=True)
            best_match = matched_intents[0]
            
            return {
                'intent': best_match['intent'],
                'preferred_types': best_match['config']['preferred_types'],
                'boost_summary': best_match['config']['boost_summary'],
                'confidence': min(best_match['score'] / 3.0, 1.0),  # 归一化到0-1
                'description': best_match['config']['description']
            }
        
        # 默认意图
        config = get_config()
        return {
            'intent': 'general',
            'preferred_types': config.get_types(),
            'boost_summary': False,
            'confidence': 0.5,
            'description': '通用问题'
        }
    
    def extract_character_names(self, query: str, all_chunks: List[Dict] = None) -> List[str]:
        """
        从问题中提取人物名
        
        Args:
            query: 用户问题
            all_chunks: 所有chunks（用于提取常见人名）
        
        Returns:
            人物名列表
        """
        # 简单实现：提取2-4个连续中文字符
        # 更好的方式是维护一个人名词典
        potential_names = re.findall(r'[\u4e00-\u9fa5]{2,4}', query)
        
        # 过滤常见词
        common_words = ['为什么', '怎么样', '什么样', '怎么会', '如何', '详细', '剖析', '分析']
        names = [n for n in potential_names if n not in common_words]
        
        return names


# 全局单例
_classifier_instance = None

def get_intent_classifier():
    """获取全局意图分类器实例"""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = IntentClassifier()
    return _classifier_instance
