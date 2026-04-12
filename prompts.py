"""
RAG系统 - Prompt配置中心
支持长文本（小说）和短文本（聊天记录）两种模式
包含所有RAG相关的prompt：
- 查询改写、RAG回答、语义标注、意图分类等
- 短文本处理：话题分段、上下文标注等
"""

try:
    from langchain_core.prompts import PromptTemplate
except ImportError:
    try:
        from langchain.prompts import PromptTemplate
    except ImportError as e:
        raise ImportError(
            "无法导入 PromptTemplate，请确认已安装兼容版本的 `langchain-core` 或 `langchain`。"
        ) from e

from text_type_config import get_config


# ===================== 查询改写Prompt =====================

def get_query_rewrite_prompt() -> str:
    """获取查询改写prompt模板"""
    config = get_config()
    
    if config.text_type == "novel":
        return """
你是小说检索优化助手，请优化用户问题，提升向量检索精度。

规则：
1. 自动补充关键词：人物/剧情/动机/关系
2. 识别隐含的查询意图（如：性格分析、剧情发展、人物关系等）
3. 补充小说上下文线索（如：哪个角色、什么事件）
4. 不改变问题原意
5. 只输出优化后的问题，无多余内容

原始问题：{}
优化后：""".strip()
    elif config.text_type == "manual":
        return """
你是工业机器说明书检索优化助手，请优化用户问题，提升向量检索精度。

规则：
1. 自动补充关键词：设备型号/部件名称/操作步骤/参数设置/故障码/安全注意
2. 识别隐含的查询意图（如：开机流程、参数调节、报警排查、维护保养）
3. 保留原始约束条件（如温度、压力、电流、转速等）
4. 不改变问题原意
5. 只输出优化后的问题，无多余内容

原始问题：{}
优化后：""".strip()
    else:
        return """
你是情感文本检索优化助手，请优化用户问题，提升向量检索精度。

规则：
1. 自动补充关键词：情感/态度/语气/时间/人物关系
2. 识别隐含的情感意图（如：生气、开心、失落、期待等）
3. 补充对话上下文线索（如：谁对谁说、什么场景）
4. 不改变问题原意
5. 只输出优化后的问题，无多余内容

原始问题：{}
优化后：""".strip()

QUERY_REWRITE_PROMPT = get_query_rewrite_prompt()


# ===================== RAG回答生成Prompt =====================

def get_rag_answer_prompt_template() -> str:
    """获取RAG回答prompt模板"""
    config = get_config()
    
    if config.text_type == "novel":
        return """你是专业的小说内容分析助手，请根据以下小说片段证据回答问题。

规则：
1. 只使用给出的小说片段证据
2. 分析剧情发展、人物性格、动机原因
3. 回答要结合小说情节和人物关系
4. 如果证据不足，明确说明
5. 回答简洁、有理有据

小说片段证据：
{evidence}

问题：{question}

回答："""
    elif config.text_type == "manual":
        return """你是专业的工业机器说明书问答助手，请根据以下说明书证据回答问题。

规则：
1. 只使用给出的说明书证据，不要编造
2. 优先给出可执行步骤，按顺序表达（步骤1/2/3）
3. 涉及参数时给出参数名、推荐值或范围；证据不足时明确标注
4. 涉及故障时给出现象、可能原因、排查步骤和恢复动作
5. 必要时补充安全注意事项和禁止操作
6. 回答简洁、明确、可直接执行

说明书证据：
{evidence}

问题：{question}

回答："""
    else:
        return """你是专业的情感文本分析助手，请根据以下聊天记录证据回答问题。

规则：
1. 只使用给出的聊天记录证据
2. 分析对话中的情感倾向、态度变化、关系动态
3. 回答要结合对话语境和情感色彩
4. 如果证据不足，明确说明
5. 回答简洁、有理有据

聊天记录证据：
{evidence}

问题：{question}

回答："""

RAG_ANSWER_PROMPT = PromptTemplate(
    input_variables=["evidence", "question"],
    template=get_rag_answer_prompt_template()
)


# ===================== Query扩展Prompt =====================

QUERY_EXPANSION_PROMPT = """请从不同角度改写以下问题，生成{num_variants}个变体。

要求：
1. 保持原意，但换不同表达方式
2. 可以从情感角度、关系角度、时间角度等多维度改写
3. 可以拆解复杂问题为子问题
4. 可以补充情感关键词或对话背景
5. 每个变体一行，不要编号

原始问题：{original_query}

变体："""


# ===================== 语义标注Prompt =====================

def get_semantic_annotation_prompt_template() -> str:
    """获取语义标注prompt模板"""
    config = get_config()
    content_name = config.get_content_name()
    types_str = ",".join(config.get_types())
    
    if config.text_type == "novel":
        return f"""你是专业的小说语义标注助手，只输出JSON，无任何解释。

任务：分析以下小说片段，提取关键信息。

小说片段：
{{chunk}}

输出JSON格式：
{{{{
    "summary": "剧情核心内容概括（30字内）",
    "emotion": "主要情感倾向（如：愤怒/喜悦/悲伤/平静等）",
    "participants": ["角色1", "角色2"],
    "key_topics": ["话题1", "话题2"],
    "relationship_dynamic": "关系动态描述（如：对立/合作/暧昧/冷漠等）",
    "types": "内容类型（用逗号分隔，如：{types_str}）"
}}}}

只输出JSON："""
    elif config.text_type == "manual":
        return f"""你是专业的工业说明书语义标注助手，只输出JSON，无任何解释。

任务：分析以下说明书段落，提取关键信息。

说明书段落：
{{chunk}}

输出JSON格式：
{{{{
    "summary": "核心操作或参数说明（30字内）",
    "emotion": "固定输出：中性",
    "participants": ["设备或模块1", "设备或模块2"],
    "key_topics": ["操作主题1", "参数主题1"],
    "relationship_dynamic": "流程关系描述（如：前置/并行/互锁/依赖）",
    "types": "内容类型（用逗号分隔，如：{types_str}）"
}}}}

只输出JSON："""
    else:
        return f"""你是专业的情感对话标注助手，只输出JSON，无任何解释。

任务：分析以下聊天记录片段，提取关键信息。

聊天记录：
{{chunk}}

输出JSON格式：
{{{{
    "summary": "对话核心内容概括（30字内）",
    "emotion": "主要情感倾向（如：开心/生气/失落/期待/平静等）",
    "participants": ["参与者1", "参与者2"],
    "key_topics": ["话题1", "话题2"],
    "relationship_dynamic": "关系动态描述（如：亲密/疏远/冲突/和解等）",
    "types": "对话类型（用逗号分隔，如：{types_str}）"
}}}}

只输出JSON："""

SEMANTIC_ANNOTATION_PROMPT = PromptTemplate(
    input_variables=["chunk"],
    template=get_semantic_annotation_prompt_template()
)


# ===================== 语义关系提取Prompt =====================

def get_semantic_relation_extraction_prompt_template() -> str:
    """获取语义关系提取prompt模板"""
    config = get_config()

    if config.text_type == "manual":
        return """分析以下问题，提取其中的工业语义关系。

问题：{query}

请识别问题中的关系类型，从以下类型中选择：
1. 操作关系：设备/模块到执行动作与结果
2. 参数关系：参数名、取值范围、默认值、阈值
3. 故障关系：故障现象、可能原因、排查路径、恢复动作
4. 安全关系：风险点、禁止操作、防护要求
5. 维护关系：维护项目、周期、标准

输出JSON格式：
{{
    "relation_type": "关系类型",
    "entities": ["实体1", "实体2"],
    "description": "关系描述"
}}

只输出JSON："""

    if config.text_type == "novel":
        return """分析以下问题，提取其中的语义关系。

问题：{query}

请识别问题中的关系类型，从以下类型中选择：
1. 人物关系：人物之间的互动、态度、情感变化
2. 因果关系：事件原因、结果、动机
3. 时间关系：时间顺序、前后发展
4. 剧情关系：事件关联、冲突与转折
5. 主题关系：主题线索与情节指向

输出JSON格式：
{{
    "relation_type": "关系类型",
    "entities": ["实体1", "实体2"],
    "description": "关系描述"
}}

只输出JSON："""

    return """分析以下问题，提取其中的语义关系。

问题：{query}

请识别问题中的关系类型，从以下类型中选择：
1. 情感关系：表达情感、情绪变化、态度转变
2. 因果关系：原因、结果、动机
3. 时间关系：时间顺序、事件发展
4. 人物关系：对话双方、关系变化、互动模式
5. 话题关系：讨论主题、话题转换

输出JSON格式：
{{
    "relation_type": "关系类型",
    "entities": ["实体1", "实体2"],
    "description": "关系描述"
}}

只输出JSON："""


SEMANTIC_RELATION_EXTRACTION_PROMPT = get_semantic_relation_extraction_prompt_template()


# ===================== 语义关系匹配Prompt =====================

def get_semantic_relation_matching_prompt_template() -> str:
    """获取语义关系匹配prompt模板"""
    config = get_config()

    if config.text_type == "manual":
        return """判断以下说明书段落是否包含目标语义关系。

目标关系类型：{relation_type}
目标实体：{entities}
关系描述：{description}

说明书段落：
{content}

请判断该段落是否与目标关系相关，输出JSON：
{{
    "has_relation": true/false,
    "confidence": 0.0-1.0,
    "extracted_info": "提取的关键步骤/参数/故障信息",
    "reason": "判断理由"
}}

只输出JSON："""

    if config.text_type == "novel":
        return """判断以下小说片段是否包含目标语义关系。

目标关系类型：{relation_type}
目标实体：{entities}
关系描述：{description}

小说片段：
{content}

请判断该片段是否与目标关系相关，输出JSON：
{{
    "has_relation": true/false,
    "confidence": 0.0-1.0,
    "extracted_info": "提取的关系信息",
    "reason": "判断理由"
}}

只输出JSON："""

    return """判断以下聊天记录片段是否包含目标语义关系。

目标关系类型：{relation_type}
目标实体：{entities}
关系描述：{description}

聊天记录片段：
{content}

请判断该片段是否与目标关系相关，输出JSON：
{{
    "has_relation": true/false,
    "confidence": 0.0-1.0,
    "extracted_info": "提取的关系信息",
    "reason": "判断理由"
}}

只输出JSON："""


SEMANTIC_RELATION_MATCHING_PROMPT = get_semantic_relation_matching_prompt_template()


# ===================== 意图分类Prompt =====================

def get_intent_classification_prompt_template() -> str:
    """获取意图分类prompt模板"""
    config = get_config()
    
    if config.text_type == "novel":
        return """你是小说内容意图分类专家。请分析用户问题的意图类型。

问题：{query}

意图类型（选择最匹配的）：
1. 人物查询：询问角色性格、特点、背景
2. 剧情查询：询问故事发展、事件经过
3. 动机查询：询问为什么、原因、目的
4. 关系查询：询问人物关系、感情变化
5. 综合查询：包含多个意图

输出JSON格式：
{{
    "intent": "意图类型",
    "confidence": 0.0-1.0,
    "keywords": ["关键词1", "关键词2"]
}}

只输出JSON："""
    elif config.text_type == "manual":
        return """你是工业机器说明书意图分类专家。请分析用户问题的意图类型。

问题：{query}

意图类型（选择最匹配的）：
1. 操作查询：询问开机/停机/切换/执行步骤
2. 参数查询：询问参数含义、设置方法、推荐范围
3. 故障查询：询问报警、异常、报错、无法运行
4. 安全查询：询问风险、注意事项、防护、联锁条件
5. 维护查询：询问保养、点检、清洁、更换周期
6. 综合查询：包含多个意图

输出JSON格式：
{{
    "intent": "意图类型",
    "confidence": 0.0-1.0,
    "keywords": ["关键词1", "关键词2"]
}}

只输出JSON："""
    else:
        return """你是情感对话意图分类专家。请分析用户问题的意图类型。

问题：{query}

意图类型（选择最匹配的）：
1. 情感查询：询问某人的情感状态、心情、态度
2. 关系查询：询问人物关系、互动模式、关系变化
3. 事件查询：询问发生了什么事、对话内容
4. 原因查询：询问为什么、原因、动机
5. 时间查询：询问什么时候、时间顺序
6. 综合查询：包含多个意图

输出JSON格式：
{{
    "intent": "意图类型",
    "confidence": 0.0-1.0,
    "keywords": ["关键词1", "关键词2"]
}}

只输出JSON："""

INTENT_CLASSIFICATION_PROMPT = get_intent_classification_prompt_template()


# ===================== 辅助函数 =====================

def build_rag_prompt(evidence: str, question: str) -> str:
    """构建RAG回答prompt"""
    template = PromptTemplate(
        input_variables=["evidence", "question"],
        template=get_rag_answer_prompt_template()
    )
    return template.format(evidence=evidence, question=question)


def build_query_rewrite_prompt(original_query: str) -> str:
    """构建查询改写prompt"""
    return get_query_rewrite_prompt().format(original_query)


def build_query_expansion_prompt(original_query: str, num_variants: int = 2) -> str:
    """构建查询扩展prompt"""
    return QUERY_EXPANSION_PROMPT.format(
        num_variants=num_variants,
        original_query=original_query
    )


def build_semantic_annotation_prompt(chunk: str) -> str:
    """构建语义标注prompt"""
    return SEMANTIC_ANNOTATION_PROMPT.format(chunk=chunk)


def build_semantic_relation_extraction_prompt(query: str) -> str:
    """构建语义关系提取prompt"""
    return get_semantic_relation_extraction_prompt_template().format(query=query)


def build_semantic_relation_matching_prompt(
    relation_type: str,
    entities: list,
    description: str,
    content: str
) -> str:
    """构建语义关系匹配prompt"""
    return get_semantic_relation_matching_prompt_template().format(
        relation_type=relation_type,
        entities=", ".join(entities),
        description=description,
        content=content
    )


def build_intent_classification_prompt(query: str) -> str:
    """构建意图分类prompt"""
    return get_intent_classification_prompt_template().format(query=query)


# ===================== 证据结构化函数 =====================

def struct_evidence_by_type(chunks: list, max_tokens: int = 2000) -> str:
    """
    结构化证据组织（自动适配文本类型）
    - 按类型分类
    - 去重相似内容
    - 按相关性排序
    - 限制总token数
    """
    config = get_config()
    content_name = config.get_content_name()
    categories = config.get_evidence_categories()
    
    if not chunks:
        return f"无相关{content_name}"
    
    # 按类型分类
    struct_data = {cat: [] for cat in categories}
    seen_contents = set()
    
    for chunk in chunks:
        content = chunk.get("content", "")
        
        # 去重
        content_hash = content[:50]
        if content_hash in seen_contents:
            continue
        seen_contents.add(content_hash)
        
        # 构建证据条目
        rerank_score = chunk.get('rerank_score', 0)
        chunk_id = chunk.get('chunk_id', 'unknown')
        
        if rerank_score:
            evidence_item = f"【{content_name}{chunk_id} | 相关度:{rerank_score:.3f}】\n{content}"
        else:
            evidence_item = f"【{content_name}{chunk_id}】\n{content}"
        
        # 按类型分类
        types = chunk.get("types", "其他")
        if isinstance(types, list):
            types = ",".join(types)
        types_list = types.split(",")
        
        classified = False
        for t in types_list:
            t = t.strip()
            if t in struct_data:
                struct_data[t].append(evidence_item)
                classified = True
                break
        
        if not classified:
            struct_data["其他"].append(evidence_item)
    
    # 组装证据，控制长度
    evidence = ""
    current_tokens = 0
    
    for type_name, contents in struct_data.items():
        if contents and current_tokens < max_tokens:
            if config.text_type == "novel":
                type_label = "类剧情"
            elif config.text_type == "manual":
                type_label = "类说明"
            else:
                type_label = "类对话"
            type_section = f"\n【{type_name}{type_label}】\n" + "\n\n".join(contents) + "\n"
            type_tokens = len(type_section)
            
            if current_tokens + type_tokens <= max_tokens:
                evidence += type_section
                current_tokens += type_tokens
            else:
                # 截断
                remaining = max_tokens - current_tokens
                evidence += type_section[:remaining] + "\n...(证据过长，已截断)\n"
                break
    
    return evidence.strip() if evidence else f"无相关{content_name}"


# ===================== 短文本处理 - 话题分段Prompt =====================

def build_topic_segmentation_prompt(messages: list, batch_size: int = 50, 
                                   max_segment_length: int = 200) -> str:
    """
    构建话题分段prompt（用于LLM识别话题转换点）
    
    Args:
        messages: 消息列表，每个消息包含 index 和 text
        batch_size: 批量大小
        max_segment_length: 最大段落长度
    
    Returns:
        格式化的prompt字符串
    """
    # 构建消息列表
    message_list = "\n".join([
        f"{msg.index}. \"{msg.text}\""
        for msg in messages[:batch_size]
    ])
    
    prompt = f"""你是对话话题分析专家。请分析以下短文本序列，识别话题转换点并进行分段。

短文本序列：
{message_list}

规则：
1. 识别话题转换点（话题变化、时间跳跃、情感转变等）
2. 每个段落至少包含2条消息
3. 每个段落不超过{max_segment_length}字符
4. 为每个段落提取话题标签（10字以内）

请输出JSON格式：
{{
  "segments": [
    {{
      "segment_id": 1,
      "start_index": 起始消息索引,
      "end_index": 结束消息索引,
      "topic": "话题标签",
      "reason": "切分原因"
    }}
  ]
}}

只输出JSON，无其他内容："""
    
    return prompt


# ===================== 短文本处理 - 上下文标注Prompt =====================

def build_short_text_annotation_prompt(chunk_data: dict) -> str:
    """
    构建短文本标注prompt（带上下文）
    
    Args:
        chunk_data: 包含以下字段的字典
            - prev_topics: 前文话题列表
            - prev_context: 前文内容
            - core_topic: 核心话题
            - core_text: 核心内容
            - next_context: 后文内容
            - next_topics: 后文话题列表
    
    Returns:
        格式化的prompt字符串
    """
    config = get_config()
    
    types_str = '", "'.join(config.get_types())
    type_descriptions = config.get_type_descriptions()
    type_desc_text = "\n   - ".join([f"{k}：{v}" for k, v in type_descriptions.items()])
    
    importance_criteria = config.get_importance_criteria()
    importance_text = "\n   - ".join([f"{k}：{v}" for k, v in importance_criteria.items()])
    
    prev_topics_str = ", ".join(chunk_data.get("prev_topics", [])) if chunk_data.get("prev_topics") else "无"
    next_topics_str = ", ".join(chunk_data.get("next_topics", [])) if chunk_data.get("next_topics") else "无"
    
    prompt = f"""你是专业的对话内容标注专家。请分析以下对话片段（已包含上下文）。

【上文话题】
{prev_topics_str}

【上文内容】
{chunk_data.get("prev_context", "无")}

【核心内容】（重点标注这部分）
话题：{chunk_data.get("core_topic", "未知")}
内容：{chunk_data.get("core_text", "")}

【下文内容】
{chunk_data.get("next_context", "无")}

【下文话题】
{next_topics_str}

请输出标注结果（JSON格式）：
{{{{
  "core_topic": "核心话题（20字内）",
  "emotion": "主要情感（如：开心/生气/难过/平静/期待等）",
  "participants": ["参与者1", "参与者2"],
  "key_events": ["关键事件1", "关键事件2"],
  "relationship_dynamic": "关系动态（如：亲密/疏远/冲突/和解/倾诉/安慰等）",
  "types": ["{types_str}"],  // 从中选择1-2个
  "keywords": ["关键词1", "关键词2", "关键词3"],
  "summary": "核心内容总结（30字内）",
  "importance": "high/middle/low",  // 标准：{importance_text}
  "context_dependency": "high/medium/low"  // 对上下文的依赖程度
}}}}

规则：
1. types只能从 ["{types_str}"] 中选择1-2个
   - {type_desc_text}
2. 重点标注【核心内容】，上下文仅用于理解
3. importance要严格区分，不要都标为high
4. context_dependency根据核心内容的完整性判断：
   - high: 核心内容很短（<20字），高度依赖上下文才能理解
   - medium: 核心内容中等（20-50字），部分依赖上下文
   - low: 核心内容较长（>50字），相对独立完整

只输出JSON，无其他内容："""
    
    return prompt


# ===================== 短文本处理 - 辅助函数 =====================

def build_topic_segmentation_prompt_simple(messages: list, max_length: int = 200) -> str:
    """
    构建简化版话题分段prompt（用于小批量处理）
    
    Args:
        messages: 消息列表
        max_length: 最大段落长度
    
    Returns:
        格式化的prompt字符串
    """
    return build_topic_segmentation_prompt(
        messages=messages,
        batch_size=len(messages),
        max_segment_length=max_length
    )


def build_short_text_annotation_template() -> PromptTemplate:
    """
    构建短文本标注的PromptTemplate对象
    
    Returns:
        PromptTemplate对象
    """
    config = get_config()
    
    types_str = '", "'.join(config.get_types())
    type_descriptions = config.get_type_descriptions()
    type_desc_text = "\n   - ".join([f"{k}：{v}" for k, v in type_descriptions.items()])
    
    importance_criteria = config.get_importance_criteria()
    importance_text = "\n   - ".join([f"{k}：{v}" for k, v in importance_criteria.items()])
    
    template = f"""你是专业的对话内容标注专家。请分析以下对话片段（已包含上下文）。

【上文话题】
{{prev_topics}}

【上文内容】
{{prev_context}}

【核心内容】（重点标注这部分）
话题：{{core_topic}}
内容：{{core_text}}

【下文内容】
{{next_context}}

【下文话题】
{{next_topics}}

请输出标注结果（JSON格式）：
{{{{
  "core_topic": "核心话题（20字内）",
  "emotion": "主要情感（如：开心/生气/难过/平静/期待等）",
  "participants": ["参与者1", "参与者2"],
  "key_events": ["关键事件1", "关键事件2"],
  "relationship_dynamic": "关系动态（如：亲密/疏远/冲突/和解/倾诉/安慰等）",
  "types": ["{types_str}"],  // 从中选择1-2个
  "keywords": ["关键词1", "关键词2", "关键词3"],
  "summary": "核心内容总结（30字内）",
  "importance": "high/middle/low",
  "context_dependency": "high/medium/low"
}}}}

规则：
1. types只能从 ["{types_str}"] 中选择1-2个
   - {type_desc_text}
2. 重点标注【核心内容】，上下文仅用于理解
3. importance要严格区分，不要都标为high
4. context_dependency根据核心内容的完整性判断

只输出JSON，无其他内容："""
    
    return PromptTemplate(
        template=template,
        input_variables=["prev_topics", "prev_context", "core_topic", 
                        "core_text", "next_context", "next_topics"]
    )


# ===================== Prompt索引（方便查找） =====================

PROMPT_INDEX = {
    # RAG核心prompts
    "query_rewrite": "查询改写",
    "rag_answer": "RAG回答生成",
    "query_expansion": "查询扩展",
    "semantic_annotation": "语义标注",
    "semantic_relation_extraction": "语义关系提取",
    "semantic_relation_matching": "语义关系匹配",
    "intent_classification": "意图分类",
    
    # 短文本处理prompts
    "topic_segmentation": "话题分段（LLM）",
    "short_text_annotation": "短文本标注（带上下文）",
}


def list_available_prompts():
    """列出所有可用的prompt"""
    print("=" * 60)
    print("📋 可用的Prompt列表")
    print("=" * 60)
    for key, desc in PROMPT_INDEX.items():
        print(f"  {key}: {desc}")
    print("=" * 60)


# ===================== 小说专用证据结构化 =====================

def struct_evidence_by_type_novel(chunks: list, max_tokens: int = 2000) -> str:
    """
    小说专用证据结构化
    - 按类型分类：事件/心理/原因/转折
    - 去重相似内容
    - 按相关性排序
    - 限制总token数
    """
    if not chunks:
        return "无相关小说片段"
    
    # 按类型分类
    categories = ["事件", "心理", "原因", "转折", "其他"]
    struct_data = {cat: [] for cat in categories}
    seen_contents = set()
    
    for chunk in chunks:
        content = chunk.get("content", "")
        
        # 去重
        content_hash = content[:50]
        if content_hash in seen_contents:
            continue
        seen_contents.add(content_hash)
        
        # 构建证据条目
        rerank_score = chunk.get('rerank_score', 0)
        chunk_id = chunk.get('chunk_id', 'unknown')
        characters = chunk.get('characters', [])
        char_str = "、".join(characters[:3]) if characters else "未知"
        
        if rerank_score:
            evidence_item = f"【片段{chunk_id} | 人物:{char_str} | 相关度:{rerank_score:.3f}】\n{content}"
        else:
            evidence_item = f"【片段{chunk_id} | 人物:{char_str}】\n{content}"
        
        # 按类型分类
        types = chunk.get("types", "其他")
        if isinstance(types, list):
            types = ",".join(types)
        types_list = types.split(",")
        
        classified = False
        for t in types_list:
            t = t.strip()
            if t in struct_data:
                struct_data[t].append(evidence_item)
                classified = True
                break
        
        if not classified:
            struct_data["其他"].append(evidence_item)
    
    # 组装证据，控制长度
    evidence = ""
    current_tokens = 0
    
    for type_name, contents in struct_data.items():
        if contents and current_tokens < max_tokens:
            evidence += f"\n【{type_name}类剧情】\n"
            for content in contents:
                content_tokens = len(content)
                if current_tokens + content_tokens > max_tokens:
                    break
                evidence += content + "\n\n"
                current_tokens += content_tokens
    
    return evidence.strip() if evidence else "无相关小说片段"


# ===================== 短对话专用证据结构化 =====================

def struct_evidence_by_type_chat(chunks: list, max_tokens: int = 2500) -> str:
    """
    短对话专用证据结构化
    - 按类型分类：情感表达/问题讨论/日常闲聊/冲突争执
    - 保留说话人信息
    - 按话题分组
    - 限制总token数
    """
    if not chunks:
        return "无相关对话片段"
    
    # 按类型分类
    categories = ["情感表达", "问题讨论", "日常闲聊", "冲突争执", "其他"]
    struct_data = {cat: [] for cat in categories}
    seen_contents = set()
    
    for chunk in chunks:
        content = chunk.get("content", "")
        
        # 去重
        content_hash = content[:50]
        if content_hash in seen_contents:
            continue
        seen_contents.add(content_hash)
        
        # 构建证据条目
        rerank_score = chunk.get('rerank_score', 0)
        chunk_id = chunk.get('chunk_id', 'unknown')
        topic_id = chunk.get('topic_id', 0)
        speakers = chunk.get('speakers', chunk.get('characters', []))
        speaker_str = "、".join(speakers[:3]) if speakers else "未知"
        
        if rerank_score:
            evidence_item = f"【话题{topic_id} | 说话人:{speaker_str} | 相关度:{rerank_score:.3f}】\n{content}"
        else:
            evidence_item = f"【话题{topic_id} | 说话人:{speaker_str}】\n{content}"
        
        # 按类型分类
        types = chunk.get("types", "其他")
        if isinstance(types, list):
            types = ",".join(types)
        types_list = types.split(",")
        
        classified = False
        for t in types_list:
            t = t.strip()
            if t in struct_data:
                struct_data[t].append(evidence_item)
                classified = True
                break
        
        if not classified:
            struct_data["其他"].append(evidence_item)
    
    # 组装证据，控制长度
    evidence = ""
    current_tokens = 0
    
    for type_name, contents in struct_data.items():
        if contents and current_tokens < max_tokens:
            evidence += f"\n【{type_name}类对话】\n"
            for content in contents:
                content_tokens = len(content)
                if current_tokens + content_tokens > max_tokens:
                    break
                evidence += content + "\n\n"
                current_tokens += content_tokens
    
    return evidence.strip() if evidence else "无相关对话片段"
