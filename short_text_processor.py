"""
短文本处理器
结合话题分段 + 滑动窗口增强策略
处理每条不到10个字的短文本，上下文关联极强的场景
"""
import json
import re
from typing import List, Dict, Tuple
from llm import ask_llm
from config import EMBEDDING_MODEL
import numpy as np

# ===================== 配置参数 =====================
# 话题分段配置
TOPIC_SEGMENT_METHOD = "hybrid"  # "vector" / "llm" / "hybrid"
VECTOR_SIMILARITY_THRESHOLD = 0.7  # 向量相似度阈值（低于此值认为话题转换）
TIME_GAP_THRESHOLD = 300  # 时间间隔阈值（秒），超过此值强制切分
MIN_SEGMENT_LENGTH = 15  # 最小段落长度（字符数）
MAX_SEGMENT_LENGTH = 200  # 最大段落长度（字符数）

# 滑动窗口配置
CONTEXT_WINDOW_SIZE = 2  # 前后各保留N个话题段落
MAX_CONTEXT_CHARS = 150  # 上下文最大字符数
INCLUDE_TOPIC_LABEL = True  # 是否包含话题标签

# 向量化策略
VECTORIZE_STRATEGY = "full"  # "core" / "full" / "dual"

# ===================== 数据结构 =====================
class ShortMessage:
    """短文本消息"""
    def __init__(self, index: int, text: str, timestamp: str = None, speaker: str = None):
        self.index = index
        self.text = text.strip()
        self.timestamp = timestamp
        self.speaker = speaker
        self.char_count = len(self.text)
    
    def __repr__(self):
        return f"Message({self.index}: '{self.text}')"


class TopicSegment:
    """话题段落"""
    def __init__(self, segment_id: int, messages: List[ShortMessage], topic: str = None):
        self.segment_id = segment_id
        self.messages = messages
        self.topic = topic or "未分类话题"
        
        # 合并文本
        self.text = " ".join([msg.text for msg in messages])
        
        # 统计信息
        self.start_index = messages[0].index
        self.end_index = messages[-1].index
        self.message_count = len(messages)
        self.char_count = len(self.text)
        
        # 时间信息
        self.start_time = messages[0].timestamp if messages[0].timestamp else None
        self.end_time = messages[-1].timestamp if messages[-1].timestamp else None
    
    def __repr__(self):
        return f"Segment({self.segment_id}: '{self.topic}', {self.message_count} msgs, {self.char_count} chars)"


class EnhancedChunk:
    """增强型Chunk（带上下文）"""
    def __init__(self, chunk_id: int, core_segment: TopicSegment,
                 prev_segments: List[TopicSegment] = None,
                 next_segments: List[TopicSegment] = None):
        self.chunk_id = chunk_id
        self.core_segment = core_segment
        self.prev_segments = prev_segments or []
        self.next_segments = next_segments or []
        
        # 构建上下文文本
        self.prev_context = " ".join([seg.text for seg in self.prev_segments])
        self.next_context = " ".join([seg.text for seg in self.next_segments])
        
        # 构建完整文本（用于向量化）
        if INCLUDE_TOPIC_LABEL:
            self.full_text = (
                f"{self.prev_context} "
                f"[话题:{core_segment.topic}] {core_segment.text} "
                f"{self.next_context}"
            ).strip()
        else:
            self.full_text = f"{self.prev_context} {core_segment.text} {self.next_context}".strip()
        
        # 元数据
        self.has_prev = len(self.prev_segments) > 0
        self.has_next = len(self.next_segments) > 0
        self.context_dependency = self._calculate_dependency()
    
    def _calculate_dependency(self) -> str:
        """计算对上下文的依赖程度"""
        core_len = len(self.core_segment.text)
        
        if core_len < 20:
            return "high"  # 核心内容很短，高度依赖上下文
        elif core_len < 50:
            return "medium"
        else:
            return "low"
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        # 提取核心段落的说话人
        core_speakers = list(set([msg.speaker for msg in self.core_segment.messages if msg.speaker]))
        
        return {
            "chunk_id": self.chunk_id,
            "core_segment_id": self.core_segment.segment_id,
            "core_text": self.core_segment.text,
            "core_topic": self.core_segment.topic,
            "core_speakers": core_speakers,  # 新增：核心段落的说话人
            "prev_context": self.prev_context,
            "prev_topics": [seg.topic for seg in self.prev_segments],
            "next_context": self.next_context,
            "next_topics": [seg.topic for seg in self.next_segments],
            "full_text": self.full_text,
            "start_index": self.core_segment.start_index,
            "end_index": self.core_segment.end_index,
            "message_count": self.core_segment.message_count,
            "char_count": self.core_segment.char_count,
            "has_prev": self.has_prev,
            "has_next": self.has_next,
            "context_dependency": self.context_dependency,
            "start_time": self.core_segment.start_time,
            "end_time": self.core_segment.end_time
        }
    
    def __repr__(self):
        return f"EnhancedChunk({self.chunk_id}: {self.core_segment.topic}, dep={self.context_dependency})"


# ===================== 工具函数 =====================
def load_embedding_model():
    """加载向量模型"""
    try:
        from sentence_transformers import SentenceTransformer
        print(f"🔄 加载向量模型: {EMBEDDING_MODEL}")
        model = SentenceTransformer(EMBEDDING_MODEL)
        print("✅ 向量模型加载成功")
        return model
    except Exception as e:
        print(f"❌ 向量模型加载失败: {e}")
        return None


def calculate_similarity(embedding1, embedding2) -> float:
    """计算余弦相似度"""
    return np.dot(embedding1, embedding2) / (
        np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
    )


# ===================== 阶段1：话题分段 =====================
class TopicSegmenter:
    """话题分段器"""
    
    def __init__(self, method: str = TOPIC_SEGMENT_METHOD):
        self.method = method
        self.embedding_model = None
        
        if method in ["vector", "hybrid"]:
            self.embedding_model = load_embedding_model()
    
    def segment_by_vector(self, messages: List[ShortMessage]) -> List[TopicSegment]:
        """基于向量相似度的话题分段"""
        if not self.embedding_model:
            print("⚠️ 向量模型未加载，回退到LLM方法")
            return self.segment_by_llm(messages)
        
        print(f"\n🔄 使用向量相似度进行话题分段...")
        print(f"   阈值: {VECTOR_SIMILARITY_THRESHOLD}")
        
        # 编码所有消息
        texts = [msg.text for msg in messages]
        embeddings = self.embedding_model.encode(texts)
        
        # 计算相邻消息的相似度
        segments = []
        current_segment_messages = [messages[0]]
        segment_id = 1
        
        for i in range(1, len(messages)):
            similarity = calculate_similarity(embeddings[i-1], embeddings[i])
            
            # 判断是否需要切分
            should_split = False
            
            # 条件1：相似度低于阈值
            if similarity < VECTOR_SIMILARITY_THRESHOLD:
                should_split = True
                reason = f"相似度={similarity:.3f}"
            
            # 条件2：时间间隔过大
            if messages[i].timestamp and messages[i-1].timestamp:
                time_gap = self._calculate_time_gap(messages[i-1].timestamp, messages[i].timestamp)
                if time_gap > TIME_GAP_THRESHOLD:
                    should_split = True
                    reason = f"时间间隔={time_gap}秒"
            
            # 条件3：当前段落过长
            current_length = sum(msg.char_count for msg in current_segment_messages)
            if current_length > MAX_SEGMENT_LENGTH:
                should_split = True
                reason = f"段落过长={current_length}字符"
            
            if should_split:
                # 创建段落
                segment = TopicSegment(segment_id, current_segment_messages)
                segments.append(segment)
                print(f"   切分点 {i}: {reason}")
                
                # 开始新段落
                segment_id += 1
                current_segment_messages = [messages[i]]
            else:
                current_segment_messages.append(messages[i])
        
        # 添加最后一个段落
        if current_segment_messages:
            segment = TopicSegment(segment_id, current_segment_messages)
            segments.append(segment)
        
        print(f"✅ 向量分段完成: {len(segments)} 个段落")
        return segments
    
    def segment_by_llm(self, messages: List[ShortMessage], batch_size: int = 50) -> List[TopicSegment]:
        """基于LLM的话题分段（批量处理）"""
        print(f"\n🔄 使用LLM进行话题分段...")
        print(f"   批量大小: {batch_size}")
        
        # 导入prompt构建函数
        from prompts import build_topic_segmentation_prompt
        
        all_segments = []
        segment_id = 1
        
        # 分批处理
        for batch_start in range(0, len(messages), batch_size):
            batch_end = min(batch_start + batch_size, len(messages))
            batch_messages = messages[batch_start:batch_end]
            
            print(f"   处理批次: {batch_start+1}-{batch_end}/{len(messages)}")
            
            # 使用统一的prompt构建函数
            prompt = build_topic_segmentation_prompt(
                messages=batch_messages,
                batch_size=len(batch_messages),
                max_segment_length=MAX_SEGMENT_LENGTH
            )
            
            try:
                response = ask_llm(prompt)
                
                # 解析JSON
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = response[json_start:json_end]
                    result = json.loads(json_str)
                    
                    # 构建段落
                    for seg_info in result.get("segments", []):
                        start_idx = seg_info["start_index"]
                        end_idx = seg_info["end_index"]
                        topic = seg_info.get("topic", "未分类话题")
                        
                        # 获取对应的消息
                        seg_messages = [
                            msg for msg in batch_messages
                            if start_idx <= msg.index <= end_idx
                        ]
                        
                        if seg_messages:
                            segment = TopicSegment(segment_id, seg_messages, topic)
                            all_segments.append(segment)
                            segment_id += 1
                            print(f"      段落{segment_id-1}: {topic} ({len(seg_messages)}条)")
                
            except Exception as e:
                print(f"   ⚠️ LLM分段失败: {e}，使用默认分段")
                # 默认分段：每5条消息一个段落
                for i in range(0, len(batch_messages), 5):
                    seg_messages = batch_messages[i:i+5]
                    segment = TopicSegment(segment_id, seg_messages, "默认段落")
                    all_segments.append(segment)
                    segment_id += 1
        
        print(f"✅ LLM分段完成: {len(all_segments)} 个段落")
        return all_segments
    
    def segment_hybrid(self, messages: List[ShortMessage]) -> List[TopicSegment]:
        """混合方法：向量粗筛 + LLM精调"""
        print(f"\n🔄 使用混合方法进行话题分段...")
        
        # 步骤1：向量粗筛
        vector_segments = self.segment_by_vector(messages)
        
        # 步骤2：LLM精调（只处理边界模糊的段落）
        refined_segments = []
        
        for i, segment in enumerate(vector_segments):
            # 如果段落过短或过长，用LLM重新分析
            if segment.char_count < MIN_SEGMENT_LENGTH or segment.char_count > MAX_SEGMENT_LENGTH:
                print(f"   精调段落 {segment.segment_id}: {segment.char_count}字符")
                
                # 用LLM重新分析这个段落
                llm_segments = self.segment_by_llm(segment.messages, batch_size=len(segment.messages))
                refined_segments.extend(llm_segments)
            else:
                refined_segments.append(segment)
        
        # 重新编号
        for i, seg in enumerate(refined_segments, 1):
            seg.segment_id = i
        
        print(f"✅ 混合分段完成: {len(refined_segments)} 个段落")
        return refined_segments
    
    def segment(self, messages: List[ShortMessage]) -> List[TopicSegment]:
        """执行话题分段"""
        if self.method == "vector":
            return self.segment_by_vector(messages)
        elif self.method == "llm":
            return self.segment_by_llm(messages)
        elif self.method == "hybrid":
            return self.segment_hybrid(messages)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _calculate_time_gap(self, time1: str, time2: str) -> float:
        """计算时间间隔（秒）"""
        try:
            from datetime import datetime
            t1 = datetime.fromisoformat(time1)
            t2 = datetime.fromisoformat(time2)
            return abs((t2 - t1).total_seconds())
        except:
            return 0


# ===================== 阶段2：滑动窗口增强 =====================
class ContextEnhancer:
    """上下文增强器"""
    
    def __init__(self, window_size: int = CONTEXT_WINDOW_SIZE,
                 max_context_chars: int = MAX_CONTEXT_CHARS):
        self.window_size = window_size
        self.max_context_chars = max_context_chars
    
    def enhance(self, segments: List[TopicSegment]) -> List[EnhancedChunk]:
        """为每个段落添加上下文"""
        print(f"\n🔄 添加滑动窗口上下文...")
        print(f"   窗口大小: 前后各{self.window_size}个段落")
        print(f"   最大上下文: {self.max_context_chars}字符")
        
        enhanced_chunks = []
        
        for i, segment in enumerate(segments):
            # 获取前文
            prev_start = max(0, i - self.window_size)
            prev_segments = segments[prev_start:i]
            prev_segments = self._trim_context(prev_segments, self.max_context_chars)
            
            # 获取后文
            next_end = min(len(segments), i + self.window_size + 1)
            next_segments = segments[i+1:next_end]
            next_segments = self._trim_context(next_segments, self.max_context_chars)
            
            # 创建增强chunk
            chunk = EnhancedChunk(
                chunk_id=i + 1,
                core_segment=segment,
                prev_segments=prev_segments,
                next_segments=next_segments
            )
            enhanced_chunks.append(chunk)
            
            if (i + 1) % 10 == 0:
                print(f"   处理进度: {i+1}/{len(segments)}")
        
        print(f"✅ 上下文增强完成: {len(enhanced_chunks)} 个chunks")
        return enhanced_chunks
    
    def _trim_context(self, segments: List[TopicSegment], max_chars: int) -> List[TopicSegment]:
        """裁剪上下文到指定长度"""
        if not segments:
            return []
        
        total_chars = sum(seg.char_count for seg in segments)
        
        if total_chars <= max_chars:
            return segments
        
        # 从后往前裁剪（保留最近的上下文）
        trimmed = []
        current_chars = 0
        
        for seg in reversed(segments):
            if current_chars + seg.char_count <= max_chars:
                trimmed.insert(0, seg)
                current_chars += seg.char_count
            else:
                break
        
        return trimmed


# ===================== 主处理流程 =====================
class ShortTextProcessor:
    """短文本处理器（主类）"""
    
    def __init__(self):
        self.segmenter = TopicSegmenter(method=TOPIC_SEGMENT_METHOD)
        self.enhancer = ContextEnhancer(
            window_size=CONTEXT_WINDOW_SIZE,
            max_context_chars=MAX_CONTEXT_CHARS
        )
    
    def process_txt_file(self, txt_file_path: str, 
                        output_jsonl_path: str = None,
                        line_separator: str = "\n") -> List[EnhancedChunk]:
        """
        处理TXT文件
        
        Args:
            txt_file_path: 输入TXT文件路径（每行一条短文本）
            output_jsonl_path: 输出JSONL文件路径（可选）
            line_separator: 行分隔符
        
        Returns:
            增强型chunks列表
        """
        print("=" * 80)
        print("🚀 短文本处理器")
        print("=" * 80)
        print(f"输入文件: {txt_file_path}")
        if output_jsonl_path:
            print(f"输出文件: {output_jsonl_path}")
        print("=" * 80)
        
        # 步骤1：读取文件
        messages = self._load_txt_file(txt_file_path, line_separator)
        
        # 步骤2：话题分段
        segments = self.segmenter.segment(messages)
        
        # 步骤3：滑动窗口增强
        enhanced_chunks = self.enhancer.enhance(segments)
        
        # 步骤4：保存结果
        if output_jsonl_path:
            self._save_to_jsonl(enhanced_chunks, output_jsonl_path)
        
        # 打印统计信息
        self._print_statistics(messages, segments, enhanced_chunks)
        
        return enhanced_chunks
    
    def _load_txt_file(self, file_path: str, separator: str) -> List[ShortMessage]:
        """读取TXT文件"""
        print(f"\n🔄 读取TXT文件...")
        
        messages = []
        
        try:
            # 尝试多种编码
            for encoding in ["utf-8", "gbk", "gb2312"]:
                try:
                    with open(file_path, "r", encoding=encoding) as f:
                        content = f.read()
                    print(f"✅ 使用编码: {encoding}")
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise Exception("无法识别文件编码")
            
            # 按行分割
            lines = content.split(separator)
            
            # 检测格式类型
            has_speaker = self._detect_speaker_format(lines)
            
            if has_speaker:
                print(f"✅ 检测到说话人格式")
            else:
                print(f"✅ 检测到纯文本格式")
            
            # 清洗和过滤
            message_index = 1
            for line in lines:
                line = line.strip()
                
                # 跳过空行
                if not line:
                    continue
                
                # 跳过过短的行（可能是噪音）
                if len(line) < 2:
                    continue
                
                # 解析说话人和文本
                speaker, text = self._parse_line(line, has_speaker)
                
                # 跳过解析失败的行
                if not text:
                    continue
                
                message = ShortMessage(
                    index=message_index,
                    text=text,
                    speaker=speaker
                )
                messages.append(message)
                message_index += 1
            
            print(f"✅ 读取完成: {len(messages)} 条消息")
            print(f"   平均长度: {sum(msg.char_count for msg in messages) / len(messages):.1f} 字符")
            print(f"   最短: {min(msg.char_count for msg in messages)} 字符")
            print(f"   最长: {max(msg.char_count for msg in messages)} 字符")
            
            if has_speaker:
                speakers = set(msg.speaker for msg in messages if msg.speaker)
                print(f"   说话人数: {len(speakers)} 人")
                print(f"   说话人: {', '.join(list(speakers)[:5])}")
            
            return messages
            
        except Exception as e:
            print(f"❌ 读取文件失败: {e}")
            raise
    
    def _detect_speaker_format(self, lines: List[str]) -> bool:
        """
        检测是否包含说话人格式
        支持格式：
        - "说话人: 文本"
        - "说话人：文本"（中文冒号）
        """
        speaker_count = 0
        valid_lines = 0
        
        for line in lines[:20]:  # 检查前20行
            line = line.strip()
            if len(line) < 3:
                continue
            
            valid_lines += 1
            
            # 检查是否包含冒号分隔符
            if ':' in line or '：' in line:
                speaker_count += 1
        
        # 如果超过50%的行包含冒号，认为是说话人格式
        if valid_lines > 0 and speaker_count / valid_lines > 0.5:
            return True
        
        return False
    
    def _parse_line(self, line: str, has_speaker: bool) -> tuple:
        """
        解析一行文本，提取说话人和内容
        
        Args:
            line: 原始行文本
            has_speaker: 是否包含说话人
        
        Returns:
            (speaker, text) 元组
        """
        if not has_speaker:
            return None, line
        
        # 尝试用英文冒号分割
        if ':' in line:
            parts = line.split(':', 1)
            if len(parts) == 2:
                speaker = parts[0].strip()
                text = parts[1].strip()
                
                # 清理表情符号（如 [旺柴]、[Emm]）
                text = self._clean_emojis(text)
                
                return speaker, text
        
        # 尝试用中文冒号分割
        if '：' in line:
            parts = line.split('：', 1)
            if len(parts) == 2:
                speaker = parts[0].strip()
                text = parts[1].strip()
                
                # 清理表情符号
                text = self._clean_emojis(text)
                
                return speaker, text
        
        # 如果没有冒号，返回整行作为文本
        return None, line
    
    def _clean_emojis(self, text: str) -> str:
        """
        清理文本中的表情符号
        支持格式：[表情名称]
        """
        # 检查配置是否启用表情清理
        try:
            from config import CLEAN_EMOJIS
            if not CLEAN_EMOJIS:
                return text
        except:
            pass  # 如果配置不存在，默认清理
        
        import re
        # 移除方括号表情
        text = re.sub(r'\[.*?\]', '', text)
        return text.strip()
    
    def _save_to_jsonl(self, chunks: List[EnhancedChunk], output_path: str):
        """保存为JSONL格式"""
        print(f"\n🔄 保存结果到: {output_path}")
        
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                for chunk in chunks:
                    chunk_dict = chunk.to_dict()
                    f.write(json.dumps(chunk_dict, ensure_ascii=False) + "\n")
            
            print(f"✅ 保存完成: {len(chunks)} 个chunks")
            
        except Exception as e:
            print(f"❌ 保存失败: {e}")
            raise
    
    def _print_statistics(self, messages: List[ShortMessage],
                         segments: List[TopicSegment],
                         chunks: List[EnhancedChunk]):
        """打印统计信息"""
        print("\n" + "=" * 80)
        print("📊 处理统计")
        print("=" * 80)
        
        print(f"\n原始消息:")
        print(f"  总数: {len(messages)}")
        print(f"  总字符数: {sum(msg.char_count for msg in messages)}")
        
        print(f"\n话题段落:")
        print(f"  总数: {len(segments)}")
        print(f"  平均长度: {sum(seg.char_count for seg in segments) / len(segments):.1f} 字符")
        print(f"  平均消息数: {sum(seg.message_count for seg in segments) / len(segments):.1f} 条")
        
        print(f"\n增强Chunks:")
        print(f"  总数: {len(chunks)}")
        
        # 统计上下文依赖程度
        dep_stats = {"high": 0, "medium": 0, "low": 0}
        for chunk in chunks:
            dep_stats[chunk.context_dependency] += 1
        
        print(f"  上下文依赖:")
        for dep, count in dep_stats.items():
            ratio = count / len(chunks) * 100
            print(f"    {dep}: {count} ({ratio:.1f}%)")
        
        print("=" * 80)


# ===================== 使用示例 =====================
if __name__ == "__main__":
    # 示例：处理短文本TXT文件
    processor = ShortTextProcessor()
    
    # 输入输出路径
    input_txt = r"D:\rag\venv\短文本对话.txt"  # 每行一条短文本
    output_jsonl = r"D:\rag\venv\短文本_增强版.jsonl"
    
    # 处理
    chunks = processor.process_txt_file(
        txt_file_path=input_txt,
        output_jsonl_path=output_jsonl
    )
    
    # 查看前3个chunks
    print("\n" + "=" * 80)
    print("📋 示例Chunks（前3个）")
    print("=" * 80)
    for chunk in chunks[:3]:
        print(f"\nChunk {chunk.chunk_id}:")
        print(f"  核心话题: {chunk.core_segment.topic}")
        print(f"  核心内容: {chunk.core_segment.text}")
        print(f"  前文: {chunk.prev_context[:50]}...")
        print(f"  后文: {chunk.next_context[:50]}...")
        print(f"  依赖程度: {chunk.context_dependency}")
