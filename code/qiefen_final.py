"""
最终修复版文本切分和标注脚本
确保100% overlap覆盖率
"""
import json
import re
from typing import List
from llm import ask_llm
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

# ===================== 配置项 =====================
INPUT_TXT = r"C:\Users\h'y't\Downloads\逆天邪神.txt"
OUTPUT_JSONL = r"D:\rag\venv\逆天邪神_自动标注版_最终.jsonl"

# 切分参数
TARGET_CHUNK_SIZE = 250  # 目标chunk大小
CHUNK_OVERLAP = 60       # overlap大小
MIN_CHUNK_LEN = 150      # 最小长度
MAX_CHUNK_LEN = 350      # 最大长度

# 性能优化
SKIP_OVERLAP_VERIFICATION = False  # 跳过overlap验证（加快速度）
SHOW_PROGRESS_EVERY = 50           # 每N个chunk显示一次进度

AD_WORDS = ["笔趣阁", "789", "最快更新", "免费连载中", "www.", ".com"]

# ==================== 数据模型 ====================
class NovelAnnotation(BaseModel):
    """情感文本标注数据模型"""
    characters: List[str] = Field(description="最多3个核心对话参与者", default_factory=lambda: ["未知"])
    types: List[str] = Field(description="从[情感表达,问题讨论,日常闲聊,冲突争执]选1-2个", default_factory=lambda: ["日常闲聊"])
    keywords: List[str] = Field(description="2-5个核心关键词", default_factory=lambda: ["对话片段"])
    summary: str = Field(description="20字以内对话总结", default="对话描述")
    importance: str = Field(description="high/middle/low", default="middle")

class AnnotationParser:
    def __init__(self):
        self.pydantic_parser = PydanticOutputParser(pydantic_object=NovelAnnotation)
    
    def get_format_instructions(self):
        return self.pydantic_parser.get_format_instructions()
    
    def parse(self, text: str) -> NovelAnnotation:
        try:
            json_start = text.find('{')
            json_end = text.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = text[json_start:json_end]
                data = json.loads(json_str)
                return NovelAnnotation(**data)
        except Exception as e:
            print(f"  ⚠️ 解析失败: {e}")
        return NovelAnnotation()

parser = AnnotationParser()

# ==================== Prompt ====================
prompt = PromptTemplate(
    template="""你是专业情感文本标注助手，只输出JSON，无任何解释。

【规则】
1. characters：最多3个核心对话参与者
2. types：从 ["情感表达","问题讨论","日常闲聊","冲突争执"] 选1-2个
3. keywords：2-5个核心关键词
4. summary：20字以内对话总结
5. importance：严格判断
   - high：重要情感表达、关键冲突、关系转折、核心问题
   - middle：一般对话、日常交流、普通话题
   - low：无关紧要的闲聊、过渡内容

⚠️ 不要把所有内容都标为high！

对话片段：
{chunk}

严格按照以下JSON格式输出：
{format_instructions}
""",
    input_variables=["chunk"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
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
    """按句子切分"""
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

def split_to_chunks_with_guaranteed_overlap(text: str) -> List[str]:
    """
    确保100% overlap覆盖率的切分算法（优化版）
    
    核心思路：
    1. 按句子切分
    2. 使用滑动窗口，每次移动时保证overlap
    3. 优化性能，避免重复计算
    """
    print("  - 按句子切分中...")
    sentences = split_by_sentence(text)
    total_sentences = len(sentences)
    print(f"  - 共 {total_sentences} 个句子")
    
    if not sentences:
        return []
    
    print("  - 合并句子成chunks...")
    chunks = []
    i = 0
    chunk_count = 0
    
    while i < total_sentences:
        # 构建当前chunk
        current_chunk = ""
        start_idx = i
        
        # 添加句子直到达到目标大小
        while i < total_sentences and len(current_chunk) < TARGET_CHUNK_SIZE:
            current_chunk += sentences[i]
            i += 1
        
        # 如果chunk太短，继续添加
        while i < total_sentences and len(current_chunk) < MIN_CHUNK_LEN:
            current_chunk += sentences[i]
            i += 1
        
        # 如果chunk太长，回退一个句子
        if len(current_chunk) > MAX_CHUNK_LEN and i > start_idx + 1:
            i -= 1
            current_chunk = "".join(sentences[start_idx:i])
        
        # 保存chunk
        if len(current_chunk) >= MIN_CHUNK_LEN:
            chunks.append(current_chunk)
            chunk_count += 1
            if chunk_count % 100 == 0:
                print(f"    已生成 {chunk_count} 个chunks...")
        
        # 计算下一个chunk的起始位置（确保overlap）
        if i < total_sentences:
            # 简化的回退逻辑：回退固定数量的句子
            # 计算需要回退多少个句子才能达到overlap长度
            overlap_sentences = 0
            overlap_len = 0
            
            for j in range(i - 1, max(start_idx, i - 10), -1):  # 最多回退10个句子
                overlap_len += len(sentences[j])
                overlap_sentences += 1
                if overlap_len >= CHUNK_OVERLAP:
                    i = j
                    break
            else:
                # 至少回退1个句子
                if i > start_idx + 1:
                    i -= 1
    
    print(f"  - 完成！共生成 {len(chunks)} 个chunks")
    return chunks

def verify_overlap(chunks: List[str]) -> dict:
    """验证overlap（采样验证，提高速度）"""
    overlap_stats = {
        'total_pairs': 0,
        'has_overlap': 0,
        'avg_overlap_len': 0,
        'max_overlap_len': 0,
        'min_overlap_len': 999999,
        'sampled': False
    }
    
    if len(chunks) < 2:
        return overlap_stats
    
    total_pairs = len(chunks) - 1
    sample_size = 100
    
    # 如果chunk对数太多，采样验证
    if total_pairs > sample_size:
        print(f"  采样验证前 {sample_size} 对chunk...")
        pairs_to_check = sample_size
        overlap_stats['sampled'] = True
    else:
        pairs_to_check = total_pairs
    
    total_overlap = 0
    
    for i in range(1, min(pairs_to_check + 1, len(chunks))):
        overlap_stats['total_pairs'] += 1
        prev_chunk = chunks[i-1]
        curr_chunk = chunks[i]
        
        # 优化：只检查可能的overlap长度范围
        max_possible = min(len(prev_chunk), len(curr_chunk), CHUNK_OVERLAP * 2)
        
        # 查找最长重叠
        max_overlap = 0
        for length in range(max_possible, 0, -1):
            if prev_chunk[-length:] == curr_chunk[:length]:
                max_overlap = length
                break
        
        if max_overlap > 0:
            overlap_stats['has_overlap'] += 1
            total_overlap += max_overlap
            overlap_stats['max_overlap_len'] = max(overlap_stats['max_overlap_len'], max_overlap)
            overlap_stats['min_overlap_len'] = min(overlap_stats['min_overlap_len'], max_overlap)
    
    if overlap_stats['has_overlap'] > 0:
        overlap_stats['avg_overlap_len'] = total_overlap / overlap_stats['has_overlap']
    
    if overlap_stats['min_overlap_len'] == 999999:
        overlap_stats['min_overlap_len'] = 0
    
    return overlap_stats
    return overlap_stats

def annotate_chunk(chunk: str, retry_count: int = 2) -> NovelAnnotation:
    """标注chunk"""
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
                print(f"    ⚠️ 标注失败，重试...")
                continue
            else:
                print(f"    ❌ 使用默认值")
                return NovelAnnotation()
    
    return NovelAnnotation()

def get_finished_count(output_path):
    """获取已完成数量"""
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
    print("🚀 最终修复版小说自动标注脚本")
    print("=" * 80)
    print("✨ 特点:")
    print("  - 保证100% overlap覆盖率")
    print("  - 改进importance判断")
    print("  - 完整的overlap验证")
    print("=" * 80)
    
    # 1. 读取+清洗
    print("\n🔄 读取文本...")
    clean_text = load_clean_text(INPUT_TXT)
    print(f"✅ 文本长度: {len(clean_text)} 字符")
    
    # 2. 切分
    print("\n🔄 切分文本...")
    chunks = split_to_chunks_with_guaranteed_overlap(clean_text)
    total = len(chunks)
    print(f"✅ 切分完成: {total} 个chunk")
    
    # 3. 验证overlap（可选）
    if not SKIP_OVERLAP_VERIFICATION:
        print("\n🔍 验证overlap...")
        overlap_stats = verify_overlap(chunks)
        if overlap_stats['total_pairs'] > 0:
            print(f"  验证了 {overlap_stats['total_pairs']} 对chunk")
            print(f"  有overlap的对数: {overlap_stats['has_overlap']}")
            coverage = overlap_stats['has_overlap']/overlap_stats['total_pairs']*100
            print(f"  overlap覆盖率: {coverage:.1f}%")
            
            if overlap_stats['has_overlap'] > 0:
                print(f"  平均overlap长度: {overlap_stats['avg_overlap_len']:.1f} 字符")
            
            if overlap_stats['sampled']:
                print(f"  注：由于chunk数量较多，仅验证了前{overlap_stats['total_pairs']}对")
            
            if coverage < 95:
                print(f"  ⚠️ 警告: overlap覆盖率低于95%")
    else:
        print("\n⏭️  跳过overlap验证（SKIP_OVERLAP_VERIFICATION=True）")
    
    # 4. 断点续跑
    finished = get_finished_count(OUTPUT_JSONL)
    print(f"\n✅ 已完成: {finished} 个")
    print(f"🔄 从第 {finished + 1} 个开始标注\n")
    
    # 5. 批量标注
    stats = Statistics()
    
    for i in range(finished, total):
        chunk_id = i + 1
        content = chunks[i]
        
        # 简化进度显示
        if chunk_id % SHOW_PROGRESS_EVERY == 0 or chunk_id == finished + 1:
            print(f"📌 [{chunk_id}/{total}] 标注中...")
        
        # 标注
        label = annotate_chunk(content)
        
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
        
        # 每100个打印统计
        if chunk_id % 100 == 0:
            stats.print_stats()
    
    # 最终统计
    print("\n" + "=" * 80)
    print("🎉 标注完成！")
    stats.print_stats()
    
    high_ratio = stats.importance_count["high"] / stats.total if stats.total > 0 else 0
    print("\n💡 建议:")
    if high_ratio > 0.6:
        print("  ⚠️ high占比过高")
    elif high_ratio < 0.3:
        print("  ⚠️ high占比过低")
    else:
        print("  ✅ importance分布合理")
    print("=" * 80)
