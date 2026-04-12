"""
短对话标注脚本 - 完整版
结合向量模型 + 合并策略 + 完整标注
"""
import json
import re
from typing import List, Dict
from llm import ask_llm
from config import EMBEDDING_MODEL
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

# ===================== 配置项 =====================
# 合并参数
MIN_DIALOG_LENGTH = 100
MAX_DIALOG_LENGTH = 300
MIN_TURNS = 3
MAX_TURNS = 10

# 向量模型配置
USE_VECTOR_MODEL = True  # 是否使用向量模型进行语义切分
VECTOR_SIMILARITY_THRESHOLD = 0.6  # 向量相似度阈值（低于此值认为话题转换）

# ==================== 数据模型（短文本专用）====================
class ChatAnnotation(BaseModel):
    """短对话标注数据模型"""
    characters: List[str] = Field(description="最多3个核心对话参与者", default_factory=lambda: ["未知"])
    types: List[str] = Field(description="从[情感表达,问题讨论,日常闲聊,冲突争执]选1-2个", default_factory=lambda: ["日常闲聊"])
    keywords: List[str] = Field(description="2-5个核心关键词", default_factory=lambda: ["对话片段"])
    summary: str = Field(description="20字以内对话总结", default="对话描述")
    importance: str = Field(description="high/middle/low", default="middle")

class AnnotationParser:
    def __init__(self):
        self.pydantic_parser = PydanticOutputParser(pydantic_object=ChatAnnotation)
    
    def get_format_instructions(self):
        return self.pydantic_parser.get_format_instructions()
    
    def parse(self, text: str) -> ChatAnnotation:
        try:
            json_start = text.find('{')
            json_end = text.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = text[json_start:json_end]
                data = json.loads(json_str)
                return ChatAnnotation(**data)
        except Exception as e:
            print(f"  ⚠️ 解析失败: {e}")
        return ChatAnnotation()

parser = AnnotationParser()

# ==================== Prompt（短文本专用）====================
prompt = PromptTemplate(
    template="""你是专业对话标注助手，只输出JSON，无任何解释。

【规则】
1. characters：最多3个核心对话参与者（使用对话中出现的真实姓名或昵称，不要用"我"、"你"等泛指）
2. types：从 ["情感表达","问题讨论","日常闲聊","冲突争执"] 选1-2个
   - 情感表达：表达情绪、感受、态度、关心
   - 问题讨论：讨论问题、寻求建议、分析情况
   - 日常闲聊：日常交流、寒暄、闲谈
   - 冲突争执：争论、冲突、不满、抱怨
3. keywords：2-5个核心关键词（人名、情感词、关键话题）
4. summary：20字以内对话总结
5. importance：严格判断
   - high：重要情感表达、关键冲突、关系转折、核心问题
   - middle：一般对话、日常交流、普通话题
   - low：无关紧要的闲聊、过渡内容

⚠️ 重要提示：
- characters 必须使用对话中的真实姓名（如"元"、"菻荺"），不要用"我"、"你"等代词
- 如果对话格式是"姓名: 内容"，请提取姓名作为 characters

⚠️ 不要把所有内容都标为high！

对话片段：
{chunk}

严格按照以下JSON格式输出：
{format_instructions}
""",
    input_variables=["chunk"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# ==================== 向量模型加载 ====================
def load_embedding_model():
    """加载向量模型（复用short_text_processor.py的逻辑）"""
    if not USE_VECTOR_MODEL:
        return None
    
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        print(f"🔄 加载向量模型: {EMBEDDING_MODEL}")
        model = SentenceTransformer(EMBEDDING_MODEL)
        print("✅ 向量模型加载成功")
        return model
    except Exception as e:
        print(f"❌ 向量模型加载失败: {e}")
        return None

def calculate_similarity(embedding1, embedding2) -> float:
    """计算余弦相似度"""
    import numpy as np
    return np.dot(embedding1, embedding2) / (
        np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
    )

# ==================== 短对话切分逻辑 ====================
def parse_line(line: str) -> tuple:
    """解析一行，提取说话人和内容"""
    # 清理换行符和特殊标记
    line = line.replace('\n', ' ').strip()
    
    # 清理表情符号标记
    line = re.sub(r'\[该消息类型暂不能展示\]', '', line)
    line = re.sub(r'\[.*?\]', '', line)  # 清理所有方括号表情
    line = line.strip()
    
    if not line:
        return None, None
    
    # 尝试用中文冒号分割
    if '：' in line:
        parts = line.split('：', 1)
        if len(parts) == 2:
            return parts[0].strip(), parts[1].strip()
    
    # 尝试用英文冒号分割
    if ':' in line:
        parts = line.split(':', 1)
        if len(parts) == 2:
            return parts[0].strip(), parts[1].strip()
    
    return None, line

def load_and_parse_txt(file_path: str) -> List[Dict]:
    """读取并解析TXT文件"""
    print(f"\n🔄 读取TXT文件: {file_path}")
    
    for encoding in ["utf-8", "gbk", "gb2312"]:
        try:
            with open(file_path, "r", encoding=encoding) as f:
                lines = f.readlines()
            print(f"✅ 使用编码: {encoding}")
            break
        except UnicodeDecodeError:
            continue
    else:
        raise Exception("无法识别文件编码")
    
    # 解析每一行
    turns = []
    for i, line in enumerate(lines, 1):
        speaker, text = parse_line(line)
        if text and len(text) >= 2:  # 至少2个字符
            turns.append({
                'turn_id': i,
                'speaker': speaker,
                'text': text
            })
    
    print(f"✅ 解析完成: {len(turns)}轮对话")
    return turns

def merge_turns_to_dialogs(turns: List[Dict]) -> List[Dict]:
    """将轮次合并为对话段（复用v2的修复逻辑）"""
    print(f"\n🔄 合并为长对话...")
    print(f"   目标长度: {MIN_DIALOG_LENGTH}-{MAX_DIALOG_LENGTH}字符")
    print(f"   目标轮数: {MIN_TURNS}-{MAX_TURNS}轮")
    
    dialogs = []
    dialog_id = 1
    i = 0
    
    while i < len(turns):
        if dialog_id % 50 == 0 or dialog_id == 1:
            print(f"   合并进度: {i}/{len(turns)}轮 -> {dialog_id}个对话段")
        
        current_turns = []
        current_chars = 0
        
        # 至少收集MIN_TURNS轮
        while i < len(turns) and len(current_turns) < MIN_TURNS:
            current_turns.append(turns[i])
            current_chars += len(turns[i]['text'])
            i += 1
        
        # 继续收集直到达到MAX_TURNS或MAX_DIALOG_LENGTH
        while i < len(turns) and len(current_turns) < MAX_TURNS and current_chars < MAX_DIALOG_LENGTH:
            current_turns.append(turns[i])
            current_chars += len(turns[i]['text'])
            i += 1
        
        # 如果字符数不足，继续收集
        while i < len(turns) and current_chars < MIN_DIALOG_LENGTH and len(current_turns) < MAX_TURNS:
            current_turns.append(turns[i])
            current_chars += len(turns[i]['text'])
            i += 1
        
        # 构建对话段
        if current_turns:
            speakers = list(set(t['speaker'] for t in current_turns if t['speaker']))
            
            # 构建文本：保留说话人信息，用空格分隔（不用换行符）
            text_parts = []
            for t in current_turns:
                if t['speaker']:
                    text_parts.append(f"{t['speaker']}: {t['text']}")
                else:
                    text_parts.append(t['text'])
            text = ' '.join(text_parts)  # 用空格连接，不用换行符
            
            dialogs.append({
                'dialog_id': dialog_id,
                'speakers': speakers,
                'text': text,
                'char_count': current_chars,
                'turn_count': len(current_turns),
                'turns': current_turns
            })
            dialog_id += 1
            
            # 不回退，直接前进（避免无限循环）
    
    print(f"✅ 合并完成: {len(dialogs)}个长对话段")
    print(f"   平均长度: {sum(d['char_count'] for d in dialogs) / len(dialogs):.1f}字符")
    print(f"   平均轮数: {sum(d['turn_count'] for d in dialogs) / len(dialogs):.1f}轮")
    
    return dialogs

def semantic_split_with_vector(dialogs: List[Dict], embedding_model) -> List[Dict]:
    """使用向量模型进行语义切分"""
    if not embedding_model:
        print("\n⏭️  跳过向量语义切分（模型未加载）")
        return dialogs
    
    print(f"\n🔄 使用向量模型进行语义切分...")
    print(f"   相似度阈值: {VECTOR_SIMILARITY_THRESHOLD}")
    
    refined_dialogs = []
    dialog_id = 1
    
    for i, dialog in enumerate(dialogs, 1):
        # 只对足够长的对话进行切分
        if dialog['turn_count'] < MIN_TURNS * 2:
            dialog['dialog_id'] = dialog_id
            refined_dialogs.append(dialog)
            dialog_id += 1
            continue
        
        # 对话过长，尝试切分
        if dialog['char_count'] > MAX_DIALOG_LENGTH:
            if i % 10 == 0 or i == 1:
                print(f"   处理进度: {i}/{len(dialogs)}")
            
            turns = dialog['turns']
            texts = [t['text'] for t in turns]
            
            # 编码所有轮次
            embeddings = embedding_model.encode(texts)
            
            # 查找话题转换点
            split_points = []
            for j in range(1, len(embeddings)):
                similarity = calculate_similarity(embeddings[j-1], embeddings[j])
                if similarity < VECTOR_SIMILARITY_THRESHOLD:
                    split_points.append(j)
            
            # 如果找到切分点，进行切分
            if split_points:
                print(f"     对话{dialog['dialog_id']}识别到{len(split_points)}个话题转换点")
                sub_dialogs = split_dialog_at_indices(turns, split_points)
                for sub_dialog in sub_dialogs:
                    sub_dialog['dialog_id'] = dialog_id
                    refined_dialogs.append(sub_dialog)
                    dialog_id += 1
            else:
                # 没有找到切分点，保留原对话
                dialog['dialog_id'] = dialog_id
                refined_dialogs.append(dialog)
                dialog_id += 1
        else:
            # 对话长度合适，保留
            dialog['dialog_id'] = dialog_id
            refined_dialogs.append(dialog)
            dialog_id += 1
    
    print(f"✅ 语义切分完成: {len(refined_dialogs)}个对话段")
    return refined_dialogs

def split_dialog_at_indices(turns: List[Dict], split_indices: List[int]) -> List[Dict]:
    """在指定索引处切分对话"""
    sub_dialogs = []
    start_idx = 0
    
    for split_idx in sorted(split_indices):
        if split_idx > start_idx:
            sub_turns = turns[start_idx:split_idx]
            if len(sub_turns) >= MIN_TURNS:  # 确保子对话足够长
                sub_dialog = build_dialog_from_turns(sub_turns)
                sub_dialogs.append(sub_dialog)
            start_idx = split_idx
    
    # 添加最后一段
    if start_idx < len(turns):
        sub_turns = turns[start_idx:]
        if len(sub_turns) >= MIN_TURNS:
            sub_dialog = build_dialog_from_turns(sub_turns)
            sub_dialogs.append(sub_dialog)
    
    return sub_dialogs

def build_dialog_from_turns(turns: List[Dict]) -> Dict:
    """从轮次列表构建对话"""
    speakers = list(set(t['speaker'] for t in turns if t['speaker']))
    text = '\n'.join([f"{t['speaker']}: {t['text']}" if t['speaker'] else t['text'] 
                     for t in turns])
    
    return {
        'dialog_id': 0,  # 稍后重新编号
        'speakers': speakers,
        'text': text,
        'char_count': sum(len(t['text']) for t in turns),
        'turn_count': len(turns),
        'turns': turns
    }

# ==================== 标注逻辑（复用qiefen_final.py）====================
def annotate_dialog(dialog: Dict, retry_count: int = 2) -> ChatAnnotation:
    """标注对话段"""
    formatted_prompt = prompt.format(chunk=dialog['text'])
    
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
                return ChatAnnotation()
    
    return ChatAnnotation()

# ==================== 统计类 ====================
class Statistics:
    """统计信息"""
    def __init__(self):
        self.importance_count = {"high": 0, "middle": 0, "low": 0}
        self.type_count = {}
        self.total = 0
    
    def add(self, importance: str, types: List[str]):
        self.total += 1
        self.importance_count[importance] = self.importance_count.get(importance, 0) + 1
        for t in types:
            self.type_count[t] = self.type_count.get(t, 0) + 1
    
    def print_stats(self):
        if self.total == 0:
            return
        print("\n" + "=" * 60)
        print("📊 当前统计:")
        print(f"  总数: {self.total}")
        
        print(f"\n  重要性分布:")
        for imp in ["high", "middle", "low"]:
            count = self.importance_count.get(imp, 0)
            ratio = count / self.total * 100
            print(f"    {imp}: {count} ({ratio:.1f}%)")
        
        print(f"\n  类型分布:")
        for t, count in sorted(self.type_count.items(), key=lambda x: x[1], reverse=True):
            ratio = count / self.total * 100
            print(f"    {t}: {count} ({ratio:.1f}%)")
        
        print("=" * 60)

# ==================== 主函数 ====================
def main():
    import sys
    
    print("=" * 80)
    print("🚀 短对话标注脚本 - 完整版")
    print("=" * 80)
    print("✨ 特点:")
    print("  - 加载向量模型进行语义切分")
    print("  - 复用v2的合并逻辑（无限循环已修复）")
    print("  - 复用qiefen_final的完整标注逻辑")
    print("  - 短文本专用字段定义")
    print("=" * 80)
    
    # 获取输入输出路径
    if len(sys.argv) >= 3:
        input_txt = sys.argv[1]
        output_jsonl = sys.argv[2]
    else:
        print("\n请输入文件路径:")
        input_txt = input("TXT文件路径: ").strip().strip('"')
        output_jsonl = input("输出JSONL路径: ").strip().strip('"')
    
    print(f"\n输入文件: {input_txt}")
    print(f"输出文件: {output_jsonl}")
    
    # 步骤1：加载向量模型
    embedding_model = load_embedding_model()
    
    # 步骤2：读取并解析
    turns = load_and_parse_txt(input_txt)
    
    # 步骤3：合并为对话段
    dialogs = merge_turns_to_dialogs(turns)
    
    # 步骤4：向量语义切分
    dialogs = semantic_split_with_vector(dialogs, embedding_model)
    
    # 步骤5：标注
    print("\n" + "=" * 80)
    print("🔄 开始标注")
    print("=" * 80)
    
    stats = Statistics()
    annotated_dialogs = []
    
    for i, dialog in enumerate(dialogs, 1):
        if i % 10 == 0 or i == 1:
            print(f"  标注进度: {i}/{len(dialogs)}")
        
        label = annotate_dialog(dialog)
        
        # 统计
        stats.add(label.importance, label.types)
        
        # 合并结果
        item = {
            "chunk_id": dialog['dialog_id'],
            "characters": label.characters,
            "types": label.types,
            "keywords": label.keywords,
            "summary": label.summary,
            "importance": label.importance,
            "content": dialog['text'],
            "speakers": dialog['speakers'],
            "turn_count": dialog['turn_count'],
            "char_count": dialog['char_count']
        }
        annotated_dialogs.append(item)
        
        # 每50个打印统计
        if i % 50 == 0:
            stats.print_stats()
    
    # 步骤6：保存
    print(f"\n🔄 保存结果到: {output_jsonl}")
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for item in annotated_dialogs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"✅ 保存完成: {len(annotated_dialogs)} 个对话段")
    
    # 步骤7：最终统计
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


if __name__ == "__main__":
    main()
