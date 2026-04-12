"""
文本类型配置模块
支持长文本、短文本（聊天记录）和工业说明书三种模式
"""

from typing import List, Literal
from pydantic import BaseModel, Field

# ===================== 文本类型枚举 =====================
TEXT_TYPE_NOVEL = "novel"  # 长文本（
TEXT_TYPE_CHAT = "chat"    # 短文本（聊天记录）
TEXT_TYPE_MANUAL = "manual"  # 工业说明书

# ===================== 全局配置 =====================
class TextTypeConfig:
    """文本类型配置类"""
    
    def __init__(self, text_type: str = TEXT_TYPE_CHAT):
        """
        初始化配置
        
        Args:
            text_type: 文本类型，"novel" / "chat" / "manual"
        """
        self.text_type = text_type
        self._custom_data_file = None
        self._custom_vector_db_path = None
    
    def set_text_type(self, text_type: str):
        """设置文本类型"""
        if text_type not in [TEXT_TYPE_NOVEL, TEXT_TYPE_CHAT, TEXT_TYPE_MANUAL]:
            raise ValueError(f"Invalid text_type: {text_type}")
        self.text_type = text_type
    
    def set_data_file(self, file_path: str):
        """设置自定义数据文件路径"""
        self._custom_data_file = file_path
    
    def get_data_file(self) -> str:
        """获取数据文件路径"""
        if self._custom_data_file:
            return self._custom_data_file
        
        # 默认路径
        if self.text_type == TEXT_TYPE_NOVEL:
            return r"D:\rag\venv\逆天邪神_自动标注版_最终.jsonl"
        elif self.text_type == TEXT_TYPE_MANUAL:
            return r"D:\rag\venv\工业机器说明书_标注版.jsonl"
        else:
            return r"D:\rag\venv\对话记录_标注版.jsonl"
    
    def set_vector_db_path(self, db_path: str):
        """设置自定义向量库路径"""
        self._custom_vector_db_path = db_path
    
    def get_vector_db_path(self) -> str:
        """获取向量库路径"""
        if self._custom_vector_db_path:
            return self._custom_vector_db_path
        
        # 默认路径
        if self.text_type == TEXT_TYPE_NOVEL:
            return "./chroma_rag_db"
        elif self.text_type == TEXT_TYPE_MANUAL:
            return "./chroma_manual_db"
        else:
            return "./chroma_chat_db"
    
    def get_types(self) -> List[str]:
        """获取当前文本类型的类型列表"""
        if self.text_type == TEXT_TYPE_NOVEL:
            return ["事件", "心理", "原因", "转折"]
        elif self.text_type == TEXT_TYPE_MANUAL:
            return ["操作步骤", "参数设置", "故障排查", "安全规程"]
        else:  # TEXT_TYPE_CHAT
            return ["情感表达", "问题讨论", "日常闲聊", "冲突争执"]
    
    def get_default_type(self) -> str:
        """获取默认类型"""
        if self.text_type == TEXT_TYPE_NOVEL:
            return "事件"
        elif self.text_type == TEXT_TYPE_MANUAL:
            return "操作步骤"
        else:
            return "日常闲聊"
    
    def get_character_description(self) -> str:
        """获取角色字段描述"""
        if self.text_type == TEXT_TYPE_NOVEL:
            return "最多3个核心人类角色"
        elif self.text_type == TEXT_TYPE_MANUAL:
            return "最多3个核心设备实体（机型/模块/部件）"
        else:
            return "最多3个核心对话参与者"
    
    def get_keywords_description(self) -> str:
        """获取关键词字段描述"""
        if self.text_type == TEXT_TYPE_NOVEL:
            return "2-5个核心关键词（人名、地名、关键动作）"
        elif self.text_type == TEXT_TYPE_MANUAL:
            return "3-6个核心关键词（设备名、参数名、故障码、关键操作）"
        else:
            return "2-5个核心关键词（人名、情感词、关键话题）"
    
    def get_summary_description(self) -> str:
        """获取摘要字段描述"""
        if self.text_type == TEXT_TYPE_NOVEL:
            return "20字以内剧情总结"
        elif self.text_type == TEXT_TYPE_MANUAL:
            return "30字以内操作/参数/故障总结"
        else:
            return "20字以内对话总结"
    
    def get_content_name(self) -> str:
        """获取内容名称"""
        if self.text_type == TEXT_TYPE_NOVEL:
            return "小说片段"
        elif self.text_type == TEXT_TYPE_MANUAL:
            return "说明书段落"
        else:
            return "对话片段"
    
    def get_task_description(self) -> str:
        """获取任务描述"""
        if self.text_type == TEXT_TYPE_NOVEL:
            return "提取剧情价值信息，忽略无用描写"
        elif self.text_type == TEXT_TYPE_MANUAL:
            return "提取可执行的设备操作、参数和故障处理信息"
        else:
            return "提取对话价值信息，分析情感和关系动态"
    
    def get_type_descriptions(self) -> dict:
        """获取类型详细描述"""
        if self.text_type == TEXT_TYPE_NOVEL:
            return {
                "事件": "具体动作、对话、场景变化",
                "心理": "内心想法、情绪变化",
                "原因": "解释为什么、因果关系",
                "转折": "剧情转折、意外发生"
            }
        elif self.text_type == TEXT_TYPE_MANUAL:
            return {
                "操作步骤": "设备启动、停机、切换模式、标准作业流程",
                "参数设置": "参数定义、推荐值、阈值范围、调参方法",
                "故障排查": "报警现象、可能原因、排查路径、恢复方法",
                "安全规程": "风险提示、禁用操作、联锁条件、防护要求"
            }
        else:
            return {
                "情感表达": "表达情绪、感受、态度",
                "问题讨论": "讨论问题、寻求建议、分析情况",
                "日常闲聊": "日常对话、寒暄、闲聊",
                "冲突争执": "争吵、冲突、矛盾"
            }
    
    def get_importance_criteria(self) -> dict:
        """获取重要性判断标准"""
        if self.text_type == TEXT_TYPE_NOVEL:
            return {
                "high": "主线剧情推进、重要角色出场、关键转折、核心冲突",
                "middle": "次要剧情、日常对话、场景描写、一般事件",
                "low": "环境描写、无关紧要的细节、过渡性内容"
            }
        elif self.text_type == TEXT_TYPE_MANUAL:
            return {
                "high": "关键操作步骤、安全联锁、故障恢复、停机保护",
                "middle": "常规参数说明、维护建议、一般注意事项",
                "low": "背景介绍、营销描述、重复性说明"
            }
        else:
            return {
                "high": "重要情感表达、关键冲突、关系转折、核心问题",
                "middle": "一般对话、日常交流、普通话题",
                "low": "无关紧要的闲聊、过渡性内容"
            }
    
    def get_assistant_role(self) -> str:
        """获取助手角色描述"""
        if self.text_type == TEXT_TYPE_NOVEL:
            return "专业小说语义标注助手"
        elif self.text_type == TEXT_TYPE_MANUAL:
            return "专业工业设备说明书标注助手"
        else:
            return "专业情感文本标注助手"
    
    def get_evidence_categories(self) -> List[str]:
        """获取证据分类类别"""
        if self.text_type == TEXT_TYPE_NOVEL:
            return ["事件", "心理", "原因", "转折", "其他"]
        elif self.text_type == TEXT_TYPE_MANUAL:
            return ["操作步骤", "参数设置", "故障排查", "安全规程", "其他"]
        else:
            return ["情感表达", "问题讨论", "日常闲聊", "冲突争执", "其他"]
    
    def get_intent_patterns(self) -> dict:
        """获取意图模式配置"""
        if self.text_type == TEXT_TYPE_NOVEL:
            return {
                'character_query': {
                    'keywords': ['性格', '人物', '角色', '特点', '是什么样的人'],
                    'preferred_types': ['心理', '事件'],
                    'boost_summary': True,
                    'description': '人物性格查询'
                },
                'plot_query': {
                    'keywords': ['剧情', '发生', '经过', '事件', '做了什么'],
                    'preferred_types': ['事件', '转折'],
                    'boost_summary': False,
                    'description': '剧情查询'
                },
                'motivation_query': {
                    'keywords': ['为什么', '原因', '动机', '目的'],
                    'preferred_types': ['原因', '心理'],
                    'boost_summary': True,
                    'description': '动机查询'
                },
                'relationship_query': {
                    'keywords': ['关系', '喜欢', '爱', '恨', '感情', '对.*态度'],
                    'preferred_types': ['心理', '事件'],
                    'boost_summary': True,
                    'description': '关系查询'
                }
            }
        elif self.text_type == TEXT_TYPE_MANUAL:
            return {
                'operation_query': {
                    'keywords': ['怎么', '如何', '步骤', '操作', '使用', '启动', '停机', '开机', '关机'],
                    'preferred_types': ['操作步骤', '安全规程'],
                    'boost_summary': False,
                    'description': '操作步骤查询'
                },
                'parameter_query': {
                    'keywords': ['参数', '设置', '取值', '范围', '配置', '调节', '阈值', '校准'],
                    'preferred_types': ['参数设置', '操作步骤'],
                    'boost_summary': True,
                    'description': '参数设置查询'
                },
                'troubleshooting_query': {
                    'keywords': ['故障', '报警', '异常', '报错', '错误', '无法', '不工作', '失效', '怎么办'],
                    'preferred_types': ['故障排查', '安全规程'],
                    'boost_summary': True,
                    'description': '故障排查查询'
                },
                'safety_query': {
                    'keywords': ['安全', '风险', '注意事项', '防护', '禁止', '危险', '联锁'],
                    'preferred_types': ['安全规程', '操作步骤'],
                    'boost_summary': True,
                    'description': '安全规程查询'
                },
                'maintenance_query': {
                    'keywords': ['维护', '保养', '润滑', '清洁', '点检', '检修', '更换'],
                    'preferred_types': ['操作步骤', '参数设置'],
                    'boost_summary': False,
                    'description': '维护保养查询'
                }
            }
        else:
            return {
                'emotion_query': {
                    'keywords': ['情感', '心情', '感觉', '态度', '情绪', '开心', '生气', '难过', '失落'],
                    'preferred_types': ['情感表达'],
                    'boost_summary': True,
                    'description': '情感查询类问题'
                },
                'relationship_query': {
                    'keywords': ['关系', '喜欢', '爱', '恨', '感情', '对.*态度', '看法', '互动', '相处'],
                    'preferred_types': ['情感表达', '问题讨论'],
                    'boost_summary': True,
                    'description': '关系查询类问题'
                },
                'event_query': {
                    'keywords': ['发生', '经过', '事件', '做了什么', '说了什么', '聊了什么', '讨论'],
                    'preferred_types': ['问题讨论', '日常闲聊'],
                    'boost_summary': False,
                    'description': '事件查询类问题'
                },
                'causality': {
                    'keywords': ['为什么', '原因', '导致', '因为', '怎么会', '如何'],
                    'preferred_types': ['情感表达', '冲突争执'],
                    'boost_summary': True,
                    'description': '因果推理类问题'
                },
                'time_query': {
                    'keywords': ['什么时候', '时间', '何时', '最近', '之前', '后来'],
                    'preferred_types': ['问题讨论', '日常闲聊'],
                    'boost_summary': False,
                    'description': '时间查询类问题'
                }
            }


# ===================== 全局配置实例 =====================
_global_config = None

def set_text_type(text_type: str):
    """设置全局文本类型"""
    global _global_config
    if text_type not in [TEXT_TYPE_NOVEL, TEXT_TYPE_CHAT, TEXT_TYPE_MANUAL]:
        raise ValueError(
            f"Invalid text_type: {text_type}. Must be '{TEXT_TYPE_NOVEL}' / '{TEXT_TYPE_CHAT}' / '{TEXT_TYPE_MANUAL}'"
        )
    _global_config = TextTypeConfig(text_type)
    type_desc_map = {
        TEXT_TYPE_NOVEL: '长文本（小说）',
        TEXT_TYPE_CHAT: '短文本（聊天记录）',
        TEXT_TYPE_MANUAL: '工业说明书'
    }
    print(f"✅ 文本类型已设置为: {type_desc_map.get(text_type, text_type)}")

def get_config() -> TextTypeConfig:
    """获取全局配置实例"""
    global _global_config
    if _global_config is None:
        # 默认使用聊天记录模式
        _global_config = TextTypeConfig(TEXT_TYPE_CHAT)
    return _global_config

def prompt_user_for_text_type() -> str:
    """提示用户选择文本类型"""
    print("\n" + "=" * 60)
    print("📝 请选择文本类型:")
    print("=" * 60)
    print("1. 长文本（小说）- 使用类型: 事件/心理/原因/转折")
    print("2. 短文本（聊天记录）- 使用类型: 情感表达/问题讨论/日常闲聊/冲突争执")
    print("3. 工业说明书 - 使用类型: 操作步骤/参数设置/故障排查/安全规程")
    print("=" * 60)
    
    while True:
        choice = input("请输入选项 (1 / 2 / 3): ").strip()
        if choice == "1":
            return TEXT_TYPE_NOVEL
        elif choice == "2":
            return TEXT_TYPE_CHAT
        elif choice == "3":
            return TEXT_TYPE_MANUAL
        else:
            print("❌ 无效选项，请输入 1 / 2 / 3")
