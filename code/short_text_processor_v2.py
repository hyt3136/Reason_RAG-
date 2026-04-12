"""
短文本处理器 V2
改进策略：先合并为长对话，然后语义切分
"""
import json
import re
from typing import List, Dict
from llm import ask_llm

# ===================== 配置参数 =====================
# 合并配置
MIN_DIALOG_LENGTH = 100  # 最小对话长度（字符）
MAX_DIALOG_LENGTH = 300  # 最大对话长度（字符）
MIN_TURNS = 3  # 最少对话轮数
MAX_TURNS = 10  # 最多对话轮数

# 滑动窗口配置
OVERLAP_TURNS = 2  # 重叠轮数（前后各保留N轮）

# 语义切分配置
ENABLE_SEMANTIC_SPLIT = True  # 是否启用语义切分（使用LLM识别话题转换点，质量更高）

# ===================== 步骤1：合并为长对话 =====================
def merge_to_long_dialogs(lines: List[str]) -> List[Dict]:
    """
    将短对话合并为长对话
    
    策略：
    1. 按轮次合并（3-10轮为一个对话段）
    2. 保证每个段落有足够的上下文
    3. 使用滑动窗口保证连续性
    
    Args:
        lines: 原始对话行列表
    
    Returns:
        长对话列表，每个包含：
        - dialog_id: 对话ID
        - turns: 对话轮次列表
        - speakers: 说话人列表
        - text: 完整文本
        - char_count: 字符数
    """
    print("\n🔄 步骤1：合并为长对话...")
    print(f"   目标长度: {MIN_DIALOG_LENGTH}-{MAX_DIALOG_LENGTH}字符")
    print(f"   目标轮数: {MIN_TURNS}-{MAX_TURNS}轮")
    
    # 解析每一行
    turns = []
    for i, line in enumerate(lines, 1):
        line = line.strip()
        if not line or len(line) < 2:
            continue
        
        speaker, text = parse_line(line)
        if text:
            turns.append({
                'turn_id': i,
                'speaker': speaker,
                'text': text
            })
    
    print(f"   解析完成: {len(turns)}轮对话")
    
    # 合并为长对话
    print(f"   开始合并...")
    dialogs = []
    dialog_id = 1
    i = 0
    
    while i < len(turns):
        # 显示进度（每50个对话段显示一次，减少输出）
        if dialog_id % 50 == 0 or dialog_id == 1:
            print(f"   合并进度: {i}/{len(turns)}轮 -> {dialog_id}个对话段")
        
        # 收集当前对话段的轮次
        current_turns = []
        current_chars = 0
        start_i = i  # 记录起始位置
        
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
            text = '\n'.join([f"{t['speaker']}: {t['text']}" if t['speaker'] else t['text'] 
                             for t in current_turns])
            
            dialog = {
                'dialog_id': dialog_id,
                'turns': current_turns,
                'speakers': speakers,
                'text': text,
                'char_count': current_chars,
                'turn_count': len(current_turns),
                'start_turn': current_turns[0]['turn_id'],
                'end_turn': current_turns[-1]['turn_id']
            }
            dialogs.append(dialog)
            dialog_id += 1
            
            # 简化滑动窗口逻辑：不回退，直接前进
            # 这样可以避免任何潜在的无限循环问题
            # 如果需要上下文，在后续的add_sliding_window_context中处理
            # i 已经在上面的循环中前进了，这里不需要额外操作
    
    print(f"✅ 合并完成: {len(dialogs)}个长对话段")
    print(f"   平均长度: {sum(d['char_count'] for d in dialogs) / len(dialogs):.1f}字符")
    print(f"   平均轮数: {sum(d['turn_count'] for d in dialogs) / len(dialogs):.1f}轮")
    
    return dialogs


# ===================== 步骤2：语义切分（可选） =====================
def semantic_split_if_needed(dialogs: List[Dict], enable_llm_split: bool = False) -> List[Dict]:
    """
    对对话段进行语义切分
    
    策略：
    1. 如果启用LLM切分：对所有对话段使用LLM识别话题转换点
    2. 如果禁用LLM切分：只对超过MAX_DIALOG_LENGTH的对话段强制切分
    
    Args:
        dialogs: 长对话列表
        enable_llm_split: 是否启用LLM语义切分
    
    Returns:
        切分后的对话列表
    """
    print("\n🔄 步骤2：语义切分...")
    
    if enable_llm_split:
        print(f"   使用LLM语义切分（高质量，较慢）")
        print(f"   将对 {len(dialogs)} 个对话段进行话题识别...")
    else:
        # 统计需要切分的对话
        need_split = [d for d in dialogs if d['char_count'] > MAX_DIALOG_LENGTH]
        if need_split:
            print(f"   发现{len(need_split)}个过长对话段（>{MAX_DIALOG_LENGTH}字符）")
            print(f"   使用强制切分（快速，质量一般）")
        else:
            print(f"   所有对话段长度合适，无需切分")
            return dialogs
    
    refined_dialogs = []
    dialog_id = 1
    
    for i, dialog in enumerate(dialogs, 1):
        # 显示进度
        if enable_llm_split and (i % 10 == 0 or i == 1):
            print(f"   处理进度: {i}/{len(dialogs)}")
        
        # 判断是否需要切分
        if enable_llm_split:
            # LLM模式：对所有对话段进行语义切分
            if dialog['turn_count'] >= MIN_TURNS * 2:  # 只有足够长的对话才值得切分
                print(f"   对话{dialog['dialog_id']}({dialog['char_count']}字符，{dialog['turn_count']}轮)，识别话题...")
                split_points = identify_topic_changes_llm(dialog)
                
                if split_points:
                    print(f"     识别到{len(split_points)}个话题转换点")
                    sub_dialogs = split_dialog_at_points(dialog, split_points)
                    for sub_dialog in sub_dialogs:
                        sub_dialog['dialog_id'] = dialog_id
                        refined_dialogs.append(sub_dialog)
                        dialog_id += 1
                else:
                    # LLM没有识别出切分点，保留原对话
                    print(f"     未识别到话题转换，保留原对话")
                    dialog['dialog_id'] = dialog_id
                    refined_dialogs.append(dialog)
                    dialog_id += 1
            else:
                # 对话太短，不切分
                dialog['dialog_id'] = dialog_id
                refined_dialogs.append(dialog)
                dialog_id += 1
        else:
            # 强制模式：只对过长对话段切分
            if dialog['char_count'] <= MAX_DIALOG_LENGTH:
                dialog['dialog_id'] = dialog_id
                refined_dialogs.append(dialog)
                dialog_id += 1
            else:
                # 对话过长，强制切分
                print(f"   对话{dialog['dialog_id']}过长({dialog['char_count']}字符)，强制切分...")
                sub_dialogs = force_split_by_turns(dialog, MAX_TURNS)
                for sub_dialog in sub_dialogs:
                    sub_dialog['dialog_id'] = dialog_id
                    refined_dialogs.append(sub_dialog)
                    dialog_id += 1
    
    print(f"✅ 语义切分完成: {len(refined_dialogs)}个对话段")
    return refined_dialogs


def identify_topic_changes_llm(dialog: Dict) -> List[int]:
    """使用LLM识别话题转换点"""
    prompt = f"""分析以下对话，识别话题转换点。

对话内容：
{dialog['text']}

请识别对话中的话题转换点（turn_id），输出JSON格式：
{{
    "split_points": [turn_id1, turn_id2, ...]
}}

只输出JSON："""
    
    try:
        response = ask_llm(prompt)
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            result = json.loads(response[json_start:json_end])
            return result.get('split_points', [])
    except:
        pass
    
    return []


def split_dialog_at_points(dialog: Dict, split_points: List[int]) -> List[Dict]:
    """在指定点切分对话"""
    sub_dialogs = []
    turns = dialog['turns']
    
    start_idx = 0
    for split_point in sorted(split_points):
        # 找到对应的turn索引
        split_idx = next((i for i, t in enumerate(turns) if t['turn_id'] >= split_point), len(turns))
        
        if split_idx > start_idx:
            sub_turns = turns[start_idx:split_idx]
            sub_dialog = build_dialog_from_turns(sub_turns)
            sub_dialogs.append(sub_dialog)
            start_idx = split_idx
    
    # 添加最后一段
    if start_idx < len(turns):
        sub_turns = turns[start_idx:]
        sub_dialog = build_dialog_from_turns(sub_turns)
        sub_dialogs.append(sub_dialog)
    
    return sub_dialogs


def force_split_by_turns(dialog: Dict, max_turns: int) -> List[Dict]:
    """强制按轮数切分"""
    sub_dialogs = []
    turns = dialog['turns']
    
    for i in range(0, len(turns), max_turns):
        sub_turns = turns[i:i+max_turns]
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
        'turns': turns,
        'speakers': speakers,
        'text': text,
        'char_count': sum(len(t['text']) for t in turns),
        'turn_count': len(turns),
        'start_turn': turns[0]['turn_id'],
        'end_turn': turns[-1]['turn_id']
    }


# ===================== 步骤3：添加上下文 =====================
def add_sliding_window_context(dialogs: List[Dict], window_size: int = 1) -> List[Dict]:
    """
    为每个对话段添加滑动窗口上下文
    
    Args:
        dialogs: 对话列表
        window_size: 窗口大小（前后各保留N个对话段）
    
    Returns:
        带上下文的对话列表
    """
    print(f"\n🔄 步骤3：添加滑动窗口上下文...")
    print(f"   窗口大小: 前后各{window_size}个对话段")
    
    enhanced_dialogs = []
    
    for i, dialog in enumerate(dialogs):
        # 获取前文
        prev_dialogs = dialogs[max(0, i-window_size):i]
        prev_context = '\n---\n'.join([d['text'] for d in prev_dialogs])
        
        # 获取后文
        next_dialogs = dialogs[i+1:min(len(dialogs), i+window_size+1)]
        next_context = '\n---\n'.join([d['text'] for d in next_dialogs])
        
        # 构建增强对话
        enhanced_dialog = dialog.copy()
        enhanced_dialog['prev_context'] = prev_context
        enhanced_dialog['next_context'] = next_context
        enhanced_dialog['has_prev'] = len(prev_dialogs) > 0
        enhanced_dialog['has_next'] = len(next_dialogs) > 0
        
        # 构建完整文本（用于向量化）
        full_text_parts = []
        if prev_context:
            full_text_parts.append(f"[前文]\n{prev_context}")
        full_text_parts.append(f"[核心对话]\n{dialog['text']}")
        if next_context:
            full_text_parts.append(f"[后文]\n{next_context}")
        
        enhanced_dialog['full_text'] = '\n\n'.join(full_text_parts)
        
        enhanced_dialogs.append(enhanced_dialog)
    
    print(f"✅ 上下文添加完成: {len(enhanced_dialogs)}个对话段")
    return enhanced_dialogs


# ===================== 工具函数 =====================
def parse_line(line: str) -> tuple:
    """
    解析一行文本，提取说话人和内容
    
    Returns:
        (speaker, text) 元组
    """
    # 尝试用英文冒号分割
    if ':' in line:
        parts = line.split(':', 1)
        if len(parts) == 2:
            speaker = parts[0].strip()
            text = parts[1].strip()
            text = clean_emojis(text)
            return speaker, text
    
    # 尝试用中文冒号分割
    if '：' in line:
        parts = line.split('：', 1)
        if len(parts) == 2:
            speaker = parts[0].strip()
            text = parts[1].strip()
            text = clean_emojis(text)
            return speaker, text
    
    # 没有说话人，返回整行
    return None, line


def clean_emojis(text: str) -> str:
    """清理表情符号 [表情名称]"""
    text = re.sub(r'\[.*?\]', '', text)
    return text.strip()


def load_txt_file(file_path: str) -> List[str]:
    """读取TXT文件"""
    print(f"\n🔄 读取TXT文件: {file_path}")
    
    # 尝试多种编码
    for encoding in ["utf-8", "gbk", "gb2312"]:
        try:
            with open(file_path, "r", encoding=encoding) as f:
                lines = f.readlines()
            print(f"✅ 使用编码: {encoding}")
            print(f"✅ 读取完成: {len(lines)}行")
            return [line.strip() for line in lines if line.strip()]
        except UnicodeDecodeError:
            continue
    
    raise Exception("无法识别文件编码")


def save_to_jsonl(dialogs: List[Dict], output_path: str):
    """保存为JSONL格式"""
    print(f"\n🔄 保存结果到: {output_path}")
    
    with open(output_path, "w", encoding="utf-8") as f:
        for dialog in dialogs:
            # 简化输出格式
            output_dict = {
                'chunk_id': dialog['dialog_id'],
                'content': dialog['text'],
                'speakers': dialog['speakers'],
                'turn_count': dialog['turn_count'],
                'char_count': dialog['char_count'],
                'prev_context': dialog.get('prev_context', ''),
                'next_context': dialog.get('next_context', ''),
                'full_text': dialog.get('full_text', dialog['text']),
                'start_turn': dialog['start_turn'],
                'end_turn': dialog['end_turn']
            }
            f.write(json.dumps(output_dict, ensure_ascii=False) + "\n")
    
    print(f"✅ 保存完成: {len(dialogs)}个对话段")


# ===================== 主处理流程 =====================
def process_short_text(input_txt: str, output_jsonl: str):
    """
    处理短文本对话
    
    流程：
    1. 读取TXT文件
    2. 合并为长对话（3-10轮）
    3. 语义切分（可选，处理过长对话）
    4. 添加滑动窗口上下文
    5. 保存为JSONL
    
    Args:
        input_txt: 输入TXT文件路径
        output_jsonl: 输出JSONL文件路径
    """
    print("=" * 80)
    print("🚀 短文本处理器 V2")
    print("=" * 80)
    print("策略: 先合并为长对话，然后语义切分")
    print("=" * 80)
    
    # 步骤1：读取文件
    lines = load_txt_file(input_txt)
    
    # 步骤2：合并为长对话
    dialogs = merge_to_long_dialogs(lines)
    
    # 步骤3：语义切分（处理过长对话）
    dialogs = semantic_split_if_needed(dialogs, enable_llm_split=ENABLE_SEMANTIC_SPLIT)
    
    # 步骤4：添加滑动窗口上下文
    dialogs = add_sliding_window_context(dialogs, window_size=1)
    
    # 步骤5：保存结果
    save_to_jsonl(dialogs, output_jsonl)
    
    # 打印统计
    print("\n" + "=" * 80)
    print("📊 处理统计")
    print("=" * 80)
    print(f"输入行数: {len(lines)}")
    print(f"输出对话段: {len(dialogs)}")
    print(f"平均长度: {sum(d['char_count'] for d in dialogs) / len(dialogs):.1f}字符")
    print(f"平均轮数: {sum(d['turn_count'] for d in dialogs) / len(dialogs):.1f}轮")
    print(f"最短: {min(d['char_count'] for d in dialogs)}字符")
    print(f"最长: {max(d['char_count'] for d in dialogs)}字符")
    print("=" * 80)
    
    return dialogs


# ===================== 使用示例 =====================
if __name__ == "__main__":
    # 示例
    input_txt = r"D:\rag\venv\短文本对话.txt"
    output_jsonl = r"D:\rag\venv\短文本_v2.jsonl"
    
    dialogs = process_short_text(input_txt, output_jsonl)
    
    # 查看前3个对话段
    print("\n" + "=" * 80)
    print("📋 示例对话段（前3个）")
    print("=" * 80)
    for dialog in dialogs[:3]:
        print(f"\n对话段 {dialog['dialog_id']}:")
        print(f"  轮数: {dialog['turn_count']}")
        print(f"  字符数: {dialog['char_count']}")
        print(f"  说话人: {', '.join(dialog['speakers'])}")
        print(f"  内容预览: {dialog['text'][:100]}...")
        if dialog.get('has_prev'):
            print(f"  前文: 有")
        if dialog.get('has_next'):
            print(f"  后文: 有")
