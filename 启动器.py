"""
RAG系统启动器
让用户选择运行小说RAG还是短对话RAG
"""
import sys
import subprocess

def main():
    print("=" * 60)
    print("🚀 RAG系统启动器")
    print("=" * 60)
    print("\n请选择要运行的系统:\n")
    print("  1. 📚 小说RAG系统")
    print("     - 处理长文本小说内容")
    print("     - 支持人物/剧情/动机/关系查询")
    print("     - 数据文件: 逆天邪神_自动标注版_最终.jsonl")
    print()
    print("  2. 💬 短对话RAG系统")
    print("     - 处理聊天记录和对话内容")
    print("     - 支持情感/关系/事件查询")
    print("     - 数据文件: 对话记录_标注版.jsonl")
    print()
    print("  3. 🔄 数据源切换工具")
    print("     - 管理和切换数据文件")
    print()
    print("  4. 退出")
    print()
    
    choice = input("请输入选项 (1-4): ").strip()
    
    if choice == "1":
        print("\n" + "=" * 60)
        print("启动小说RAG系统...")
        print("=" * 60 + "\n")
        # 使用当前Python解释器（保持虚拟环境）
        subprocess.run([sys.executable, "main_novel.py"])
    
    elif choice == "2":
        print("\n" + "=" * 60)
        print("启动短对话RAG系统...")
        print("=" * 60 + "\n")
        # 使用当前Python解释器（保持虚拟环境）
        subprocess.run([sys.executable, "main_chat.py"])
    
    elif choice == "3":
        print("\n" + "=" * 60)
        print("启动数据源切换工具...")
        print("=" * 60 + "\n")
        # 使用当前Python解释器（保持虚拟环境）
        subprocess.run([sys.executable, "switch_data_source.py"])
    
    elif choice == "4":
        print("\n👋 再见！")
        sys.exit(0)
    
    else:
        print("\n❌ 无效选项，请输入 1-4")
        main()

if __name__ == "__main__":
    main()
