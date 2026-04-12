"""
统一文本标注脚本
支持长文本（小说）和短文本（聊天记录）两种模式
"""
import json
import re
from typing import List
from llm import ask_llm
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from text_type_config import (
    get_config, set_text_type, prompt_user_for_text_type,
    TEXT_TYPE_NOVEL, TEXT_TYPE_CHAT
)

# ===================== 配置项 =====================
INPUT_TXT = r"C:\Users\h'y't\Downloads\逆天邪神.txt"
OUTPUT_JSONL = r"D:\rag\venv\标注结果.jsonl"

# 切分参数
CHUNK_SIZE = 280
CHUNK_OVERLAP = 70
MIN_CHUNK_LEN = 120
MAX_CHUNK_LEN = 320

AD_WORDS = ["笔趣阁", "789", "最快更新", "免费连载中", "www.", ".com"]

# ==================== 数据模型 ====================
class TextAnnotation(BaseModel):
    """文本标注数据模型（动态适配）"""
    characters: List[str] = Field(description="最多3个核心角色/参与者", default_factory=lambda: ["未知"])
    types: List[str] = Field(description="内容类型", default_factory=list)
    keywords: List[str] = Field(description="2-5个核心关键词", default_factory=lambda: ["内容片段"])
    summary: str = Field(description="20字以内总结", default="内容描述")
    importance: str = Field(description="high/middle/low", default="middle")

class AnnotationParser:
    """自定义解析器"""
    def __init__(self):
        self.pydantic_parser = PydanticOutputParser(pydantic_object=TextAnnotation)
    
    def get_format_instructions(self):
        return self.pydantic_parser.get_format_instructions()
    
    def parse(self, text: str) -> TextAnnotation:
        try:
            json_start = text.find('{')
            json_end = text.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = text[json_start:json_end]
                data = json.loads(json_str)
                return TextAnnotation(**data)
        except Exception as e:
            print(f"  ⚠️ 解析失败: {e}，使用默认值")
        
        # 返回默认值（根据当前文本类型）
        config = get_config()
        return TextAnnotation(
            characters=["未知"],
            types=[config.get_default_type()],
            keywords=["内容片段"],
            summary="内容描述",
            importance="middle"
        )

# ==================== 动态Prompt生成 ====================
def build_annotation_prompt() -> PromptTemplate:
    """根据文本类型构建标注prompt"""
    config = get_config()
    parser = AnnotationParser()
    
    types_str = '","'.join(config.get_types())
    type_descriptions = config.get_type_descriptions()
    type_desc_text = "\n   - ".join([f"{k}：{v}" for k, v in type_descriptions.items()])
    
    importance_criteria = config.get_importance_criteria()
    importance_text = "\n   - ".join([f"{k}：{v}" for k, v in importance_criteria.items()])
    
    template = f"""你是{config.get_assistant_role()}，只输出JSON，无任何解释。

【任务】{config.get_task_description()}

【规则】
1. characters：{config.get_character_description()}（不包括"未知"、"众人"等泛指）
2. types：只能从 ["{types_str}"] 选1-2个
   - {type_desc_text}
3. keywords：{config.get_keywords_description()}
4. summary：{config.get_summary_description()}
5. importance：严格按以下标准判断
   - {importance_text}

⚠️ 重要：不要把所有内容都标为high，要严格区分重要度！

{config.get_content_name()}：
{{chunk}}

严格按照以下JSON格式输出（不换行）：
{{format_instructions}}
"""
    
    return PromptTemplate(
        template=template,
        input_variables=["chunk"],
        partial_variables={
            "format_instructions": parser.get_format_instructions()
        }
    )

# ==================== 文本处理 ====================
def load_clean_text(file_path):
    """读取并清洗文本"""
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
    
    # 清洗
    text = re.sub(r'第.*?章|www\..*?\.com|[\x00-\x1F]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    for ad in AD_WORDS:
        text = text.replace(ad, "")
    
    return text.strip()

def split_by_sentence(text: str) -> List[str]:
    """按句子切分文本"""
    sentence_endings = ['。', '！', '？', '；', '……']
    sentences = []
    current = ""
    
    for char in text:
        current += char
        if char in sentence_endings:
            if current.strip():
                sentences.append(current.strip())
            current = ""
    
    if current.strip():
        sentences.append(current.strip())
    
    return sentences

def split_to_chunks(text: str) -> list:
    """优化的文本切分（带overlap）"""
    sentences = split_by_sentence(text)
    
    chunks = []
    current_chunk = ""
    sentence_buffer = []
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) > CHUNK_SIZE:
            if len(current_chunk) >= MIN_CHUNK_LEN:
                chunks.append(current_chunk)
            
            # 计算overlap
            overlap_text = ""
            overlap_sentences = []
            for s in reversed(sentence_buffer):
                if len(overlap_text) + len(s) <= CHUNK_OVERLAP:
                    overlap_text = s + overlap_text
                    overlap_sentences.insert(0, s)
                else:
                    break
            
            current_chunk = overlap_text + sentence
            sentence_buffer = overlap_sentences + [sentence]
        else:
            current_chunk += sentence
            sentence_buffer.append(sentence)
    
    if len(current_chunk) >= MIN_CHUNK_LEN:
        chunks.append(current_chunk)
    
    # 过滤长度
    final_chunks = []
    for chunk in chunks:
        c_clean = re.sub(r'[""\"\' ]+', '', chunk).strip()
        if MIN_CHUNK_LEN <= len(c_clean) <= MAX_CHUNK_LEN:
            final_chunks.append(c_clean)
    
    return final_chunks

def annotate_chunk(chunk: str, prompt: PromptTemplate, parser: AnnotationParser, retry_count: int = 2) -> TextAnnotation:
    """标注chunk，带重试机制"""
    formatted_prompt = prompt.format(chunk=chunk)
    
    for attempt in range(retry_count):
        try:
            llm_response = ask_llm(formatted_prompt)
            result = parser.parse(llm_response)
            
            if result.characters == ["未知"] and attempt < retry_count - 1:
                print(f"    ⚠️ 角色为默认值，重试...")
                continue
            
            return result
        except Exception as e:
            if attempt < retry_count - 1:
                print(f"    ⚠️ 标注失败，重试... ({e})")
                continue
            else:
                print(f"    ❌ 标注失败，使用默认值")
                return parser.parse("")
    
    return parser.parse("")

# ==================== 断点续标 ====================
def get_finished_count(output_path):
    """获取已完成的数量"""
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            ids = [json.loads(line)["chunk_id"] for line in f if line.strip()]
        return max(ids) if ids else 0
    except:
        return 0

def save_single_item(item, output_path):
    """保存单条数据"""
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

# ==================== 统计功能 ====================
class Statistics:
    """统计信息"""
    def __init__(self):
        self.importance_count = {"high": 0, "middle": 0, "low": 0}
        self.total = 0
    
    def add(self, importance: str):
        self.total += 1
        self.importance_count[importance] = self.importance_count.get(importance, 0) + 1
    
    def print_stats(self):
        if self.total == 0:
            return
        print("\n" + "=" * 60)
        print("📊 当前统计:")
        print(f"  总数: {self.total}")
        for imp in ["high", "middle", "low"]:
            count = self.importance_count.get(imp, 0)
            ratio = count / self.total * 100
            print(f"  {imp}: {count} ({ratio:.1f}%)")
        print("=" * 60)

# ==================== 主函数 ====================
if __name__ == "__main__":
    print("=" * 80)
    print("🚀 统一文本自动标注脚本")
    print("=" * 80)
    
    # 1. 选择文本类型
    text_type = prompt_user_for_text_type()
    set_text_type(text_type)
    
    config = get_config()
    print(f"\n✅ 使用类型: {', '.join(config.get_types())}")
    
    # 2. 构建prompt和parser
    print("\n🔄 初始化标注工具...")
    prompt = build_annotation_prompt()
    parser = AnnotationParser()
    
    # 3. 读取+清洗
    print("\n🔄 读取文本...")
    clean_text = load_clean_text(INPUT_TXT)
    print(f"✅ 文本长度: {len(clean_text)} 字符")
    
    # 4. 切分
    print("\n🔄 切分文本...")
    chunks = split_to_chunks(clean_text)
    total = len(chunks)
    print(f"✅ 切分完成: {total} 个chunk")
    print(f"   长度范围: {MIN_CHUNK_LEN}-{MAX_CHUNK_LEN} 字符")
    
    # 5. 断点续跑
    finished = get_finished_count(OUTPUT_JSONL)
    print(f"\n✅ 已完成: {finished} 个")
    print(f"🔄 从第 {finished + 1} 个开始标注\n")
    
    # 6. 批量标注
    stats = Statistics()
    
    for i in range(finished, total):
        chunk_id = i + 1
        content = chunks[i]
        
        print(f"📌 [{chunk_id}/{total}] 标注中... (长度: {len(content)})")
        
        # 标注
        label = annotate_chunk(content, prompt, parser)
        
        # 统计
        stats.add(label.importance)
        
        # 保存
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
        
        # 每10个打印一次统计
        if chunk_id % 10 == 0:
            stats.print_stats()
    
    # 最终统计
    print("\n" + "=" * 80)
    print("🎉 标注完成！")
    stats.print_stats()
    print("\n💡 建议:")
    high_ratio = stats.importance_count["high"] / stats.total if stats.total > 0 else 0
    if high_ratio > 0.6:
        print("  ⚠️ high占比过高，建议检查prompt或手动调整部分数据")
    elif high_ratio < 0.3:
        print("  ⚠️ high占比过低，可能遗漏重要内容")
    else:
        print("  ✅ importance分布合理")
    print("=" * 80)
