# Prompt快速参考

> 所有prompt统一在 `prompts.py` 中管理

## 快速导入

```python
from prompts import (
    # RAG核心
    build_query_rewrite_prompt,
    build_rag_prompt,
    build_query_expansion_prompt,
    build_semantic_annotation_prompt,
    build_intent_classification_prompt,
    build_semantic_relation_extraction_prompt,
    build_semantic_relation_matching_prompt,
    
    # 短文本处理
    build_topic_segmentation_prompt,
    build_short_text_annotation_prompt,
    build_short_text_annotation_template,
    
    # 辅助函数
    struct_evidence_by_type,
    list_available_prompts
)
```

## RAG核心Prompts

| Prompt | 函数 | 用途 | 自动适配 |
|--------|------|------|----------|
| 查询改写 | `build_query_rewrite_prompt(query)` | 优化用户问题 | ✅ |
| RAG回答 | `build_rag_prompt(evidence, question)` | 基于证据生成回答 | ✅ |
| 查询扩展 | `build_query_expansion_prompt(query, num=2)` | 生成问题变体 | ❌ |
| 语义标注 | `build_semantic_annotation_prompt(chunk)` | 提取语义信息 | ✅ |
| 意图分类 | `build_intent_classification_prompt(query)` | 识别问题意图 | ✅ |
| 关系提取 | `build_semantic_relation_extraction_prompt(query)` | 提取语义关系 | ❌ |
| 关系匹配 | `build_semantic_relation_matching_prompt(...)` | 判断关系相关性 | ❌ |

## 短文本处理Prompts

| Prompt | 函数 | 用途 | 参数 |
|--------|------|------|------|
| 话题分段 | `build_topic_segmentation_prompt(messages, batch_size, max_length)` | LLM识别话题转换 | messages列表 |
| 短文本标注 | `build_short_text_annotation_prompt(chunk_data)` | 标注带上下文的chunk | chunk_data字典 |
| 标注模板 | `build_short_text_annotation_template()` | 返回PromptTemplate对象 | 无 |

## 使用示例

### 查询改写
```python
prompt = build_query_rewrite_prompt("他为什么心情不好？")
optimized = ask_llm(prompt)
```

### RAG回答
```python
prompt = build_rag_prompt(
    evidence="检索到的对话内容...",
    question="他为什么心情不好？"
)
answer = ask_llm(prompt)
```

### 话题分段
```python
from short_text_processor import ShortMessage

messages = [
    ShortMessage(1, "在吗"),
    ShortMessage(2, "在的"),
    # ...
]

prompt = build_topic_segmentation_prompt(
    messages=messages,
    batch_size=50,
    max_segment_length=200
)
segments = ask_llm(prompt)
```

### 短文本标注
```python
chunk_data = {
    "prev_topics": ["寒暄"],
    "prev_context": "在吗 在的",
    "core_topic": "情绪倾诉",
    "core_text": "今天心情不好 怎么了 被老板骂了",
    "next_context": "太过分了 别难过",
    "next_topics": ["安慰"]
}

prompt = build_short_text_annotation_prompt(chunk_data)
annotation = ask_llm(prompt)
```

## 自动适配说明

标记为 ✅ 的prompt会根据 `text_type_config` 自动适配：

```python
from text_type_config import set_text_type, TEXT_TYPE_NOVEL, TEXT_TYPE_CHAT

# 小说模式
set_text_type(TEXT_TYPE_NOVEL)
prompt = build_query_rewrite_prompt("萧澈的性格？")
# → 使用小说检索优化助手的prompt

# 聊天记录模式
set_text_type(TEXT_TYPE_CHAT)
prompt = build_query_rewrite_prompt("他为什么生气？")
# → 使用情感文本检索优化助手的prompt
```

## 查看所有Prompts

```python
from prompts import list_available_prompts
list_available_prompts()
```

输出：
```
============================================================
📋 可用的Prompt列表
============================================================
  query_rewrite: 查询改写
  rag_answer: RAG回答生成
  query_expansion: 查询扩展
  semantic_annotation: 语义标注
  semantic_relation_extraction: 语义关系提取
  semantic_relation_matching: 语义关系匹配
  intent_classification: 意图分类
  topic_segmentation: 话题分段（LLM）
  short_text_annotation: 短文本标注（带上下文）
============================================================
```

## 修改Prompt

直接编辑 `prompts.py` 文件，所有使用该prompt的模块会自动更新。

## 详细文档

查看 `文档需求/PROMPT_MANAGEMENT.md` 获取完整文档。
