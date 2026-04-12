"""
短对话TXT文件上传和处理工具
用户可以上传TXT格式的短对话文件，自动进行话题分段、上下文增强和标注
"""
import os
import sys
from typing import Optional
from short_text_processor import ShortTextProcessor
from annotate_short_text import annotate_short_text_file
from text_type_config import set_text_type, TEXT_TYPE_CHAT
import config


def upload_and_process_txt(
    txt_file_path: str,
    output_dir: Optional[str] = None,
    auto_annotate: bool = True,
    auto_vectorize: bool = True
) -> dict:
    """
    上传并处理TXT短对话文件
    
    Args:
        txt_file_path: TXT文件路径（每行一条短对话）
        output_dir: 输出目录（可选，默认与输入文件同目录）
        auto_annotate: 是否自动标注（默认True）
        auto_vectorize: 是否自动向量化并存入数据库（默认True）
    
    Returns:
        处理结果字典
    """
    print("\n" + "=" * 80)
    print("📤 短对话TXT文件上传和处理工具")
    print("=" * 80)
    
    # 验证文件存在
    if not os.path.exists(txt_file_path):
        raise FileNotFoundError(f"文件不存在: {txt_file_path}")
    
    # 设置文本类型为聊天记录
    set_text_type(TEXT_TYPE_CHAT)
    
    # 确定输出目录
    if output_dir is None:
        output_dir = os.path.dirname(txt_file_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成输出文件名
    base_name = os.path.splitext(os.path.basename(txt_file_path))[0]
    enhanced_jsonl = os.path.join(output_dir, f"{base_name}_增强版.jsonl")
    annotated_jsonl = os.path.join(output_dir, f"{base_name}_标注版.jsonl")
    
    result = {
        "input_file": txt_file_path,
        "enhanced_file": None,
        "annotated_file": None,
        "vectorized": False,
        "message_count": 0,
        "segment_count": 0,
        "chunk_count": 0
    }
    
    try:
        # 步骤1：话题分段 + 上下文增强
        print("\n" + "=" * 80)
        print("📝 步骤1：话题分段和上下文增强")
        print("=" * 80)
        
        processor = ShortTextProcessor()
        chunks = processor.process_txt_file(
            txt_file_path=txt_file_path,
            output_jsonl_path=enhanced_jsonl,
            line_separator=config.TXT_LINE_SEPARATOR
        )
        
        result["enhanced_file"] = enhanced_jsonl
        result["chunk_count"] = len(chunks)
        
        print(f"\n✅ 增强版文件已保存: {enhanced_jsonl}")
        
        # 步骤2：自动标注（可选）
        if auto_annotate and config.AUTO_ANNOTATE_ENABLED:
            print("\n" + "=" * 80)
            print("🏷️ 步骤2：自动标注")
            print("=" * 80)
            
            try:
                annotate_short_text_file(
                    input_jsonl=enhanced_jsonl,
                    output_jsonl=annotated_jsonl,
                    batch_size=config.AUTO_ANNOTATE_BATCH_SIZE
                )
                
                result["annotated_file"] = annotated_jsonl
                print(f"\n✅ 标注版文件已保存: {annotated_jsonl}")
                
            except Exception as e:
                print(f"\n⚠️ 自动标注失败: {e}")
                print("   可以稍后手动运行标注脚本")
        
        # 步骤3：向量化并存入数据库（可选）
        if auto_vectorize and result["annotated_file"]:
            print("\n" + "=" * 80)
            print("🗄️ 步骤3：向量化并存入数据库")
            print("=" * 80)
            
            try:
                from data_loader import load_and_store_data
                
                # 更新config中的JSONL路径
                original_jsonl = config.ANNOTATED_JSONL
                config.ANNOTATED_JSONL = annotated_jsonl
                
                # 加载数据
                load_and_store_data()
                
                # 恢复原始配置
                config.ANNOTATED_JSONL = original_jsonl
                
                result["vectorized"] = True
                print(f"\n✅ 数据已存入向量数据库")
                
            except Exception as e:
                print(f"\n⚠️ 向量化失败: {e}")
                print("   可以稍后手动运行 data_loader.py")
        
        # 打印最终结果
        print("\n" + "=" * 80)
        print("✅ 处理完成")
        print("=" * 80)
        print(f"输入文件: {result['input_file']}")
        print(f"增强版文件: {result['enhanced_file']}")
        if result['annotated_file']:
            print(f"标注版文件: {result['annotated_file']}")
        print(f"Chunk数量: {result['chunk_count']}")
        if result['vectorized']:
            print(f"向量化: ✅ 已完成")
        print("=" * 80)
        
        return result
        
    except Exception as e:
        print(f"\n❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()
        raise


def interactive_upload():
    """交互式上传模式"""
    print("\n" + "=" * 80)
    print("📤 短对话TXT文件上传工具（交互模式）")
    print("=" * 80)
    
    # 获取文件路径
    print("\n请输入TXT文件路径:")
    print("（每行一条短对话，支持UTF-8/GBK/GB2312编码）")
    txt_file_path = input("文件路径: ").strip().strip('"').strip("'")
    
    if not os.path.exists(txt_file_path):
        print(f"❌ 文件不存在: {txt_file_path}")
        return
    
    # 询问是否自动标注
    print("\n是否自动标注？（推荐）")
    print("1. 是（自动提取类型、角色、关键词等）")
    print("2. 否（仅进行话题分段和上下文增强）")
    choice = input("请选择 (1/2，默认1): ").strip() or "1"
    auto_annotate = (choice == "1")
    
    # 询问是否自动向量化
    if auto_annotate:
        print("\n是否自动向量化并存入数据库？（推荐）")
        print("1. 是（可立即使用RAG检索）")
        print("2. 否（稍后手动向量化）")
        choice = input("请选择 (1/2，默认1): ").strip() or "1"
        auto_vectorize = (choice == "1")
    else:
        auto_vectorize = False
    
    # 执行处理
    try:
        result = upload_and_process_txt(
            txt_file_path=txt_file_path,
            auto_annotate=auto_annotate,
            auto_vectorize=auto_vectorize
        )
        
        print("\n🎉 处理成功！")
        
        if auto_vectorize and result["vectorized"]:
            print("\n💡 提示：现在可以使用 main.py 进行RAG检索了")
            print("   示例: python main.py")
        elif result["annotated_file"]:
            print("\n💡 提示：如需向量化，请运行:")
            print(f"   python data_loader.py")
        
    except Exception as e:
        print(f"\n❌ 处理失败: {e}")


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='短对话TXT文件上传和处理工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 交互模式
  python upload_txt_chat.py
  
  # 命令行模式（完整处理）
  python upload_txt_chat.py --file "对话记录.txt"
  
  # 仅话题分段，不标注
  python upload_txt_chat.py --file "对话记录.txt" --no-annotate
  
  # 指定输出目录
  python upload_txt_chat.py --file "对话记录.txt" --output "./output"
        """
    )
    
    parser.add_argument(
        '--file',
        type=str,
        help='TXT文件路径'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='输出目录（可选）'
    )
    
    parser.add_argument(
        '--no-annotate',
        action='store_true',
        help='不进行自动标注'
    )
    
    parser.add_argument(
        '--no-vectorize',
        action='store_true',
        help='不进行自动向量化'
    )
    
    args = parser.parse_args()
    
    if args.file:
        # 命令行模式
        try:
            result = upload_and_process_txt(
                txt_file_path=args.file,
                output_dir=args.output,
                auto_annotate=not args.no_annotate,
                auto_vectorize=not args.no_vectorize
            )
            print("\n✅ 处理完成")
        except Exception as e:
            print(f"\n❌ 处理失败: {e}")
            sys.exit(1)
    else:
        # 交互模式
        interactive_upload()


if __name__ == "__main__":
    main()
