"""
本地LLM调用模块
"""
import requests

def ask_llm(prompt):
    """调用本地Ollama的Qwen3.5-9B生成答案"""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "qwen3.5:9b",
                "prompt": prompt,
                "stream": False,
                "think": False, 
                "options": {
                    "temperature": 0.3,
                    "num_ctx": 4096
                }
            },
            timeout=60
        )
        response.raise_for_status()
        return response.json()["response"]
    except Exception as e:
        return f"大模型调用失败：{str(e)}"
