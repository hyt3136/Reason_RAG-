"""
数据源切换工具
快速切换RAG系统使用的数据源（小说 / 短对话 / 工业说明书）
"""
import os
import shutil
from text_type_config import TextTypeConfig

def list_jsonl_files():
    """列出当前目录下所有JSONL文件"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    jsonl_files = [f for f in os.listdir(current_dir) if f.endswith('.jsonl')]
    return jsonl_files

def switch_to_novel():
    """切换到小说模式"""
    config = TextTypeConfig()
    config.set_text_type("novel")
    print("✅ 已切换到小说模式")
    print(f"   - 数据文件: {config.get_data_file()}")
    print(f"   - 向量库路径: {config.get_vector_db_path()}")
    print(f"   - 内容名称: {config.get_content_name()}")

def switch_to_chat():
    """切换到短对话模式"""
    config = TextTypeConfig()
    config.set_text_type("chat")
    print("✅ 已切换到短对话模式")
    print(f"   - 数据文件: {config.get_data_file()}")
    print(f"   - 向量库路径: {config.get_vector_db_path()}")
    print(f"   - 内容名称: {config.get_content_name()}")

def switch_to_manual():
    """切换到工业说明书模式"""
    config = TextTypeConfig()
    config.set_text_type("manual")
    print("✅ 已切换到工业说明书模式")
    print(f"   - 数据文件: {config.get_data_file()}")
    print(f"   - 向量库路径: {config.get_vector_db_path()}")
    print(f"   - 内容名称: {config.get_content_name()}")

def set_custom_data_file(jsonl_file: str):
    """设置自定义数据文件"""
    if not os.path.exists(jsonl_file):
        print(f"❌ 文件不存在: {jsonl_file}")
        return False
    
    config = TextTypeConfig()
    
    # 根据文件名判断类型
    lower_name = jsonl_file.lower()
    if "说明书" in jsonl_file or "manual" in lower_name or "工业" in jsonl_file:
        config.set_text_type("manual")
        print("✅ 检测到工业说明书文件，已切换到manual模式")
    elif "对话" in jsonl_file or "chat" in lower_name:
        config.set_text_type("chat")
        print("✅ 检测到短对话文件，已切换到chat模式")
    else:
        config.set_text_type("novel")
        print("✅ 检测到小说文件，已切换到novel模式")
    
    # 更新配置文件中的数据路径
    config.set_data_file(jsonl_file)
    
    print(f"   - 数据文件: {jsonl_file}")
    print(f"   - 向量库路径: {config.get_vector_db_path()}")
    return True

def show_current_config():
    """显示当前配置"""
    config = TextTypeConfig()
    print("\n" + "=" * 60)
    print("📋 当前配置:")
    print("=" * 60)
    print(f"文本类型: {config.text_type}")
    print(f"数据文件: {config.get_data_file()}")
    print(f"向量库路径: {config.get_vector_db_path()}")
    print(f"内容名称: {config.get_content_name()}")
    print("=" * 60 + "\n")

def main():
    """主函数"""
    print("=" * 60)
    print("🔄 RAG数据源切换工具")
    print("=" * 60)
    
    # 显示当前配置
    show_current_config()
    
    # 列出可用的JSONL文件
    jsonl_files = list_jsonl_files()
    if jsonl_files:
        print("📁 当前目录下的JSONL文件:")
        for i, f in enumerate(jsonl_files, 1):
            size = os.path.getsize(f) / 1024 / 1024  # MB
            print(f"   {i}. {f} ({size:.2f} MB)")
        print()
    
    # 菜单
    print("请选择操作:")
    print("  1. 切换到小说模式（使用默认小说数据）")
    print("  2. 切换到短对话模式（使用默认对话数据）")
    print("  3. 切换到工业说明书模式（使用默认说明书数据）")
    print("  4. 指定自定义JSONL文件")
    print("  5. 查看当前配置")
    print("  6. 退出")
    print()
    
    choice = input("请输入选项 (1-6): ").strip()
    
    if choice == "1":
        switch_to_novel()
        print("\n💡 提示: 运行 python main.py 启动RAG系统")
    
    elif choice == "2":
        switch_to_chat()
        print("\n💡 提示: 运行 python main.py 启动RAG系统")
    
    elif choice == "3":
        switch_to_manual()
        print("\n💡 提示: 运行 python main_manual.py 启动工业说明书RAG系统")

    elif choice == "4":
        if jsonl_files:
            print("\n选择文件:")
            for i, f in enumerate(jsonl_files, 1):
                print(f"  {i}. {f}")
            print(f"  {len(jsonl_files) + 1}. 输入自定义路径")
            
            file_choice = input(f"\n请输入选项 (1-{len(jsonl_files) + 1}): ").strip()
            
            try:
                file_idx = int(file_choice)
                if 1 <= file_idx <= len(jsonl_files):
                    selected_file = jsonl_files[file_idx - 1]
                    set_custom_data_file(selected_file)
                    print("\n💡 提示: 运行 python main.py 启动RAG系统")
                elif file_idx == len(jsonl_files) + 1:
                    custom_path = input("请输入JSONL文件完整路径: ").strip()
                    if set_custom_data_file(custom_path):
                        print("\n💡 提示: 运行 python main.py 启动RAG系统")
                else:
                    print("❌ 无效选项")
            except ValueError:
                print("❌ 请输入数字")
        else:
            custom_path = input("请输入JSONL文件完整路径: ").strip()
            if set_custom_data_file(custom_path):
                print("\n💡 提示: 运行 python main.py 启动RAG系统")
    
    elif choice == "5":
        show_current_config()
    
    elif choice == "6":
        print("👋 再见！")
    
    else:
        print("❌ 无效选项")

if __name__ == "__main__":
    main()
