# PromptTemplate 在不同 LangChain 版本/包布局下可能位于不同模块
try:
    from langchain_core.prompts import PromptTemplate  # type: ignore[import]
except ImportError:
    try:
        from langchain.prompts import PromptTemplate  # type: ignore[import]
    except ImportError as e:
        raise ImportError(
            "无法导入 PromptTemplate，请确认已安装兼容版本的 `langchain-core` 或 `langchain`。"
        ) from e
# 直接用你原生的llm，无额外文件！
from llm import ask_llm
from prompts import (
    build_query_rewrite_prompt,
    build_rag_prompt,
    build_query_expansion_prompt,
    struct_evidence_by_type
)


# ===================== 轻量智能查询改写（带打印 + 无报错） =====================
def query_rewrite(original_query: str) -> str:
    """
    智能改写：共用全局LLM，打印改写日志
    """
    try:
        # 使用prompts.py中的prompt
        prompt = build_query_rewrite_prompt(original_query)
        rewritten_query = ask_llm(prompt)
        
        # 打印改写日志
        print(f"\n📝 原始问题：{original_query}")
        print(f"✨ 改写后：{rewritten_query}")
        return rewritten_query
    except Exception as e:
        print(f"\n📝 原始问题：{original_query}")
        print(f"✨ 改写后：{original_query}（改写失败，使用原问题）")
        return original_query


# ===================== 以下代码使用prompts.py中的函数 =====================

def query_expansion(original_query: str, num_variants: int = 2) -> list:
    """
    Query多角度扩展
    生成不同角度的查询变体，提高召回率
    
    Args:
        original_query: 原始问题
        num_variants: 生成变体数量
    
    Returns:
        查询列表（包含原始query和变体）
    """
    try:
        prompt = build_query_expansion_prompt(original_query, num_variants)
        response = ask_llm(prompt)
        variants = [line.strip() for line in response.split('\n') if line.strip()]
        # 去除可能的编号
        variants = [v.lstrip('0123456789.、-） ') for v in variants]
        # 限制数量
        variants = variants[:num_variants]
        # 添加原始query
        return [original_query] + variants
    except Exception as e:
        print(f"⚠️ Query扩展失败: {e}")
        return [original_query]
