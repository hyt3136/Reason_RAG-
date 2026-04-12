import json
import re
from typing import List
from llm import ask_llm
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
# ==================== LangChain 0.3+ 新版核心导入（修复所有报错） ====================
from langchain_core.output_parsers import BaseOutputParser  # 新版路径
from langchain_core.prompts import PromptTemplate            # 新版路径
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 新版独立包
from pydantic import BaseModel, Field

# ===================== 配置项（完全保留） =====================
INPUT_TXT = r"C:\Users\h'y't\Downloads\逆天邪神.txt"
OUTPUT_JSONL = r"D:\rag\venv\逆天邪神_自动标注版.jsonl"
CHUNK_SIZE = 260
CHUNK_OVERLAP = 60
AD_WORDS = ["笔趣阁", "789", "最快更新", "免费连载中", "www.", ".com"]

# ==================================================

# ==================== 1. LangChain 结构化输出模型（强制JSON格式） ====================
class NovelAnnotation(BaseModel):
    """情感文本标注数据模型"""
    characters: List[str] = Field(description="最多3个核心对话参与者", default_factory=lambda: ["未知"])
    types: List[str] = Field(description="从[情感表达,问题讨论,日常闲聊,冲突争执]选1-2个", default_factory=lambda: ["日常闲聊"])
    keywords: List[str] = Field(description="2-5个核心关键词", default_factory=lambda: ["对话片段"])
    summary: str = Field(description="20字以内对话总结", default="对话描述")
    importance: str = Field(description="high/middle/low", default="middle")

    # # 自定义解析器（适配你的 ask_llm 函数）
    # class AnnotationParser(BaseOutputParser[NovelAnnotation]):
    #     def parse(self, text: str) -> NovelAnnotation:
    #         try:
    #             # 自动提取JSON，容错拉满
    #             json_start = text.find('{')
    #             json_end = text.rfind('}') + 1
    #             data = json.loads(text[json_start:json_end])
    #             return NovelAnnotation(**data)
    #         except:
    #             return NovelAnnotation()  # 兜底返回，永不报错
parser = PydanticOutputParser(pydantic_object=NovelAnnotation)
# ==================== 2. LangChain Prompt 模板（替代手动拼接） ====================
prompt = PromptTemplate(
    template="""你是专业情感文本标注助手，只输出JSON，无任何解释。
【任务】提取对话价值信息，分析情感和关系动态。
【规则】
1. characters：最多3个核心对话参与者
2. types：只能从 ["情感表达","问题讨论","日常闲聊","冲突争执"] 选1-2个
3. keywords：2-5个核心关键词
4. summary：20字以内
5. importance：必须为 high/middle/low

对话片段：{chunk}
严格按照以下JSON格式输出（不换行）：
{format_instructions}
""",
    input_variables=["chunk"],
    partial_variables={
        "format_instructions": AnnotationParser().get_format_instructions()
    }
)

# ==================== 3. 文本读取清洗（保留你的健壮逻辑） ====================
def load_clean_text(file_path):
    text = ""
    for enc in ["utf-8", "gbk", "ansi"]:
        try:
            with open(file_path, "r", encoding=enc) as f:
                text = f.read()
            break
        except:
            continue
    if not text:
        raise Exception("文件读取失败")

    # 清洗规则完全保留
    text = re.sub(r'第.*?章|www\..*?\.com|[\x00-\x1F]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    for ad in AD_WORDS:
        text = text.replace(ad, "")
    return text.strip()

# ==================== 4. LangChain 文本切分（替代你的50行手动切分代码） ====================
def split_to_chunks(text: str) -> list:
    """
    LangChain 递归字符切分：
    自动按 。！？；换行 分割，完美替代手动句子切分+滑动窗口
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["。", "！", "？", "；", "……", "\n", " "],  # 按语义分割
        length_function=len
    )
    chunks = splitter.split_text(text)

    # 保留你的过滤规则（长度80-350）
    final_chunks = []
    for c in chunks:
        c_clean = re.sub(r'[“”\"\' ]+', '', c).strip()
        if 80 < len(c_clean) < 350:
            final_chunks.append(c_clean)
    return final_chunks

# ==================== 5. LLM 标注链（LangChain 简化调用） ====================
def annotate_chunk(chunk: str) -> NovelAnnotation:
    """一键调用，直接返回结构化对象，无JSON解析风险"""
    formatted_prompt = prompt.format(chunk=chunk)
    llm_response = ask_llm(formatted_prompt)
    return AnnotationParser().parse(llm_response)

# ==================== 6. 断点续标 + 保存（完全保留你的核心逻辑） ====================
def get_finished_count(output_path):
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            ids = [json.loads(line)["chunk_id"] for line in f if line.strip()]
        return max(ids) if ids else 0
    except:
        return 0

def save_single_item(item, output_path):
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

# ==================== 主函数（逻辑不变，代码更简洁） ====================
if __name__ == "__main__":
    print("=" * 60)
    print("🔥 LangChain简化版 · 小说自动标注脚本")
    print("=" * 60)

    # 1. 读取+清洗
    clean_text = load_clean_text(INPUT_TXT)
    # 2. LangChain切分
    chunks = split_to_chunks(clean_text)
    total = len(chunks)
    print(f"✅ 切分完成：总 {total} 块")

    # 3. 断点续跑
    finished = get_finished_count(OUTPUT_JSONL)
    print(f"✅ 已完成：{finished} 块，从 {finished + 1} 块开始")

    # 4. 批量标注
    for i in range(finished, total):
        chunk_id = i + 1
        content = chunks[i]
        print(f"📌 正在标注：{chunk_id}/{total}")

        # 直接获取结构化对象
        label = annotate_chunk(content)

        # 组装数据（直接调用模型属性，永不报错）
        item = {
            "chunk_id": chunk_id,
            "characters": label.characters,
            "types": label.types,
            "keywords": label.keywords,
            "summary": label.summary,
            "importance": label.importance,
            "content": content
        }
        save_single_item(item, OUTPUT_JSONL)

    print("🎉 全部完成！")