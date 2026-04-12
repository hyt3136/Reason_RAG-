"""
Reranker模块 - 对召回结果进行重排序
"""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class Reranker:
    def __init__(self, model_name="BAAI/bge-reranker-base"):
        """
        初始化Reranker模型
        使用CPU运行，避免显存占用
        """
        print(f"🔄 加载Reranker模型: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            trust_remote_code=True
        )
        self.model.eval()
        print("✅ Reranker模型加载完成")
    
    def rerank(self, query: str, chunks: list, top_n: int = None):
        """
        对chunks进行重排序
        
        Args:
            query: 查询文本
            chunks: chunk字典列表，需包含'content'字段
            top_n: 返回前N个，None则返回全部
        
        Returns:
            重排序后的chunks列表
        """
        if not chunks:
            return []
        
        # 构建query-document对
        pairs = [[query, chunk.get('content', '')] for chunk in chunks]
        
        # 计算相关性分数
        with torch.no_grad():
            inputs = self.tokenizer(
                pairs, 
                padding=True, 
                truncation=True,
                max_length=512, 
                return_tensors="pt"
            )
            scores = self.model(**inputs).logits.squeeze(-1).float().tolist()
        
        # 将分数添加到chunk中
        for chunk, score in zip(chunks, scores):
            chunk['rerank_score'] = score
        
        # 按分数排序
        sorted_chunks = sorted(chunks, key=lambda x: x['rerank_score'], reverse=True)
        
        if top_n:
            return sorted_chunks[:top_n]
        return sorted_chunks


# 全局单例
_reranker_instance = None

def get_reranker():
    """获取全局Reranker实例（懒加载）"""
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = Reranker()
    return _reranker_instance
