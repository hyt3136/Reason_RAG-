# 纯原生读取，无任何第三方依赖，零报错！
import json
from config import ANNOTATED_JSONL

def load_annotated_chunks(jsonl_path: str = ANNOTATED_JSONL) -> list:
    """
    加载已标注完成的JSONL格式chunk数据
    【原生版】无依赖、不报错、Windows/Linux全兼容
    """
    chunks = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks