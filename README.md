# 企业级语义检索与问答引擎（RAG）

本项目不是一个“演示脚本集合”，而是一套可直接包装为业务系统的语义检索中台。

本项目依赖本地的哦llama部署qwen3.5 9b 模型
项目中还用到一些小模型，比如reranker模型，jieba等以及一些小的语义模型和向量模型（这些均在代码中通过hugging face下载）
参考配置rtx4060 ,i5-14450hx
需要保证cpu内心大于等于16gb（小模型大部分在cpu上运行）

如果你需要对你的项目进行定制和修改prompt.py 以及config.py文件



它具备完整闭环：
- 数据标注与切分
- 混合检索（向量 + BM25）
- 重排序（Reranker）
- 意图感知与语义关系验证
- 基于证据的回答生成
- 评估与可观测输出

---

## 一句话定位

将非结构化文本（聊天、长文、工业说明书）转化为可问、可查、可解释的业务知识系统。

## 完整优化流程

所有优化已实现并集成，系统可以投入使用。
Query → 改写 → 混合检索 → Reranker → 意图感知 → 语义验证 → 答案
         ↓        ↓          ↓         ↓          ↓
       补充别名  向量+BM25   精准排序  类型过滤   关系验证 ⭐

```
用户问题
    ↓
【1. Query处理】
    ├─ Query改写（补充角色别名、剧情关键词）
    └─ Query扩展（可选，生成多个查询变体）
    ↓
【2. 混合检索】
    ├─ 向量检索（语义相似度）
    └─ BM25检索（关键词匹配）
    ↓ 合并去重
【3. Reranker重排序】
    └─ 使用bge-reranker-base精准排序
    ↓ 保留Top-20
【4. 意图感知优化】
    ├─ 识别问题意图（5种类型）
    ├─ 根据意图过滤chunk类型
    └─ 意图感知评分
    ↓ 保留Top-20
【5. 语义关系验证】⭐ 核心创新
    ├─ 提取抽象语义关系
    ├─ LLM验证chunk是否包含目标关系
    └─ 优先返回验证通过的chunks
    ↓ 保留Top-10
【6. 证据组织】
    ├─ 按类型结构化组织
    └─ 添加语义关系增强信息
    ↓
【7. 答案生成】
    └─ 基于高质量证据生成答案
    ↓
【8. 评估报告】
    ├─ 召回指标（精准率、距离分数）
    └─ LLM答案质量评估
```

## 业务方案

### 1) 客服情感理解系统（短文本场景）

方向：
- 客诉识别与情绪预警
- 会话主题追踪
- 关系与态度变化分析
- 坐席质检辅助

项目内对应能力：
- 短文本分段与上下文增强：short_text_processor.py
- 短文本标注：annotate_short_text_complete.py / annotate_text.py
- 情感与意图检索：intent_classifier.py + intent_aware_search.py

典型价值：
- 降低人工质检成本
- 提升投诉升级前的预警率
- 缩短客服复盘时间

### 2) 内容理解与角色关系洞察（长文本场景）

方向：
- 影视/网文剧情理解助手
- 内容运营检索系统
- 角色关系图谱问答

项目内对应能力：
- 长文本标注和切分：qiefen_final.py
- 长文本问答入口：main_novel.py
- 语义关系提取与验证：semantic_relation_extractor.py

典型价值：
- 降低内容检索门槛
- 提升复杂问题回答准确率
- 支持运营与编辑快速定位证据

### 3) 工业机器说明书智能助手（工业场景）

方向：
- 设备操作指导
- 参数设置助手
- 故障排查助手
- 安全规程问答

项目内对应能力：
- 工业模式入口：main_manual.py
- 工业意图与关键词策略：text_type_config.py
- 工业Prompt与证据组织：prompts.py

典型价值：
- 缩短新员工上手周期
- 降低误操作风险
- 提升一线故障处理效率

---

## 系统架构（已落地）

```text
用户问题
  -> Query改写/扩展
  -> 混合检索（Vector + BM25）
  -> Reranker重排
  -> 意图感知优化
  -> 语义关系验证（可开关）
  -> 结构化证据组织
  -> LLM回答生成
  -> 评估报告输出
```

关键模块：
- 检索层：vector_db.py, bm25_retriever.py, reranker.py
- 语义层：intent_classifier.py, intent_aware_search.py, semantic_relation_extractor.py
- 生成层：prompts.py, utils.py, llm.py
- 数据层：data_loader.py, qiefen_final.py, annotate_text.py
- 主程序：main.py, main_novel.py, main_manual.py

---

## 快速启动

### 1. 安装依赖

```bash
pip install -r requirements_fixed.txt
```

### 2. 进入目录

```bash
cd lang_chain/rag代码
```

### 3. 按业务场景启动

```bash
python main.py          # 通用模式（默认配置）
python main_novel.py    # 长文本/小说分析模式
python main_manual.py   # 工业说明书问答模式
```

### 4. 交互命令

- eval on：开启评估输出
- eval off：关闭评估输出
- quit：退出

---

## 数据要求（JSONL）

系统复用同一套加载逻辑，建议每条至少包含以下字段：

```json
{
  "chunk_id": 1,
  "content": "文本内容",
  "types": ["类型1", "类型2"],
  "keywords": ["关键词1", "关键词2"],
  "summary": "摘要",
  "importance": "high"
}
```

工业说明书建议 types 使用：
- 操作步骤
- 参数设置
- 故障排查
- 安全规程

默认工业数据路径：
- D:\\rag\\venv\\工业机器说明书_标注版.jsonl

---

## 核心卖点（对业务方）

- 可解释：回答基于证据片段，不是黑盒输出
- 可扩展：同一套框架覆盖多行业文本
- 可配置：检索策略、权重、验证开关可调
- 可落地：已有脚本入口与处理链路，无需从零开发

---

## 典型落地方式

### 方式A：内部知识助手

把企业文档（SOP、FAQ、说明书）接入后，作为内部问答门户。

### 方式B：客服辅助中台

接入历史对话与工单，支持“情绪识别 + 原因定位 + 应答建议”。

### 方式C：设备运维助手

接入设备说明书与维护手册，提供“操作步骤 + 参数建议 + 故障排查”联动回答。

---

## 进阶建议

- 增加业务指标看板：命中率、首答解决率、平均处理时长
- 增加数据闭环：把人工修正结果反哺标注集
- 增加权限控制：按角色限制可见文档范围

---

## 相关文档

- 快速开始：快速开始.md
- 短文本方案：README_SHORT_TEXT.md
- 数据源切换：数据源切换指南.md
- Prompt索引：PROMPT_QUICK_REFERENCE.md
