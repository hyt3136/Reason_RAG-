"""
BM25关键词检索模块 - 补充向量检索的不足
"""
from typing import List, Dict
import jieba
import math
from collections import Counter


class BM25Retriever:
    def __init__(self, chunks: List[Dict], k1=1.5, b=0.75):
        """
        初始化BM25检索器
        
        Args:
            chunks: chunk列表
            k1: BM25参数，控制词频饱和度
            b: BM25参数，控制文档长度归一化
        """
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        
        # 分词并构建倒排索引
        self.tokenized_docs = []
        self.doc_lengths = []
        self.avgdl = 0
        self.doc_freqs = Counter()
        self.idf = {}
        
        self._build_index()
    
    def _tokenize(self, text: str) -> List[str]:
        """中文分词"""
        return list(jieba.cut(text))
    
    def _build_index(self):
        """构建BM25索引"""
        print("🔄 构建BM25索引...")
        
        # 分词
        for chunk in self.chunks:
            tokens = self._tokenize(chunk.get('content', ''))
            self.tokenized_docs.append(tokens)
            self.doc_lengths.append(len(tokens))
        
        # 计算平均文档长度
        self.avgdl = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0
        
        # 计算文档频率
        for tokens in self.tokenized_docs:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.doc_freqs[token] += 1
        
        # 计算IDF
        num_docs = len(self.chunks)
        for token, freq in self.doc_freqs.items():
            self.idf[token] = math.log((num_docs - freq + 0.5) / (freq + 0.5) + 1)
        
        print(f"✅ BM25索引构建完成，共 {num_docs} 个文档")
    
    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        BM25检索
        
        Args:
            query: 查询文本
            top_k: 返回前K个结果
        
        Returns:
            检索结果列表
        """
        query_tokens = self._tokenize(query)
        scores = []
        
        for idx, (doc_tokens, doc_len) in enumerate(zip(self.tokenized_docs, self.doc_lengths)):
            score = 0
            doc_token_counts = Counter(doc_tokens)
            
            for token in query_tokens:
                if token not in self.idf:
                    continue
                
                # BM25公式
                tf = doc_token_counts.get(token, 0)
                idf = self.idf[token]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                score += idf * numerator / denominator
            
            scores.append((idx, score))
        
        # 排序并返回top_k
        scores.sort(key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in scores[:top_k]:
            chunk = self.chunks[idx].copy()
            chunk['bm25_score'] = score
            
            # 确保格式一致：将列表转换为字符串（与向量库格式一致）
            if isinstance(chunk.get('types'), list):
                chunk['types'] = ','.join(chunk['types'])
            if isinstance(chunk.get('characters'), list):
                chunk['characters'] = ','.join(chunk['characters'])
            if isinstance(chunk.get('keywords'), list):
                chunk['keywords'] = ','.join(chunk['keywords'])
            
            results.append(chunk)
        
        return results


# 全局单例
_bm25_instance = None

def get_bm25_retriever(chunks: List[Dict] = None):
    """获取全局BM25检索器实例"""
    global _bm25_instance
    if _bm25_instance is None and chunks:
        _bm25_instance = BM25Retriever(chunks)
    return _bm25_instance
