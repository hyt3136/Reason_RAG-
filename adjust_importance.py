"""
后处理脚本：调整现有数据的importance
不需要重新标注，基于规则快速调整
"""
import json
import re
from collections import Counter

def adjust_importance(chunk):
    """
    基于规则调整importance
    
    降级规则：
    1. 过短的内容（<150字符）→ middle
    2. 纯对话且无转折/原因 → middle
    3. 环境描写、过渡性内容 → low
    
    保持high的条件：
    1. 包含"转折"或"原因"类型
    2. 关键词中有"决定"、"发现"、"突破"等关键动作
    3. 摘要中有"重要"、"关键"等词
    """
    content = chunk['content']
    types = chunk['types']
    keywords = chunk.get('keywords', [])
    summary = chunk.get('summary', '')
    
    # 关键词列表
    high_keywords = ['决定', '发现', '突破', '觉醒', '死亡', '重伤', '战斗', 
                     '修炼', '提升', '获得', '失去', '离开', '到达']
    low_keywords = ['看着', '走着', '想着', '说着', '环境', '描写', '景色']
    
    # 规则1: 过短内容降级
    if len(content) < 150:
        return 'middle'
    
    # 规则2: 包含转折或原因，保持high
    if '转折' in types or '原因' in types:
        return 'high'
    
    # 规则3: 检查关键词
    has_high_keyword = any(kw in ''.join(keywords) or kw in summary for kw in high_keywords)
    has_low_keyword = any(kw in content for kw in low_keywords)
    
    if has_high_keyword:
        return 'high'
    
    if has_low_keyword:
        return 'low'
    
    # 规则4: 纯对话且无特殊类型
    if ('说' in content or '道' in content) and len(types) <= 1:
        dialogue_count = content.count('说') + content.count('道')
        if dialogue_count >= 2:  # 多次对话
            return 'middle'
    
    # 规则5: 只有"事件"类型，且内容较短
    if types == ['事件'] and len(content) < 200:
        return 'middle'
    
    # 默认保持原值
    return chunk['importance']


def process_file(input_file, output_file, dry_run=False):
    """
    处理JSONL文件
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
        dry_run: 是否只统计不写入
    """
    print("=" * 80)
    print("🔄 开始处理...")
    print("=" * 80)
    
    chunks = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    
    print(f"✅ 读取 {len(chunks)} 个chunk")
    
    # 统计原始分布
    original_dist = Counter(c['importance'] for c in chunks)
    print("\n【原始importance分布】")
    for imp in ['high', 'middle', 'low']:
        count = original_dist.get(imp, 0)
        ratio = count / len(chunks) * 100
        print(f"  {imp}: {count} ({ratio:.1f}%)")
    
    # 调整
    adjusted_count = 0
    for chunk in chunks:
        original = chunk['importance']
        adjusted = adjust_importance(chunk)
        if original != adjusted:
            chunk['importance'] = adjusted
            adjusted_count += 1
    
    # 统计调整后分布
    new_dist = Counter(c['importance'] for c in chunks)
    print("\n【调整后importance分布】")
    for imp in ['high', 'middle', 'low']:
        count = new_dist.get(imp, 0)
        ratio = count / len(chunks) * 100
        change = count - original_dist.get(imp, 0)
        change_str = f"({change:+d})" if change != 0 else ""
        print(f"  {imp}: {count} ({ratio:.1f}%) {change_str}")
    
    print(f"\n📊 共调整 {adjusted_count} 个chunk ({adjusted_count/len(chunks)*100:.1f}%)")
    
    # 写入文件
    if not dry_run:
        with open(output_file, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
        print(f"\n✅ 已保存到: {output_file}")
    else:
        print("\n💡 这是预览模式，未实际写入文件")
        print(f"   如需写入，请运行: python adjust_importance.py --write")
    
    # 建议
    print("\n" + "=" * 80)
    print("💡 建议:")
    high_ratio = new_dist['high'] / len(chunks)
    if high_ratio > 0.6:
        print("  ⚠️ high占比仍然较高，建议:")
        print("     1. 检查是否需要更严格的降级规则")
        print("     2. 考虑使用LLM重新判断部分数据")
    elif high_ratio < 0.3:
        print("  ⚠️ high占比较低，建议:")
        print("     1. 检查是否降级过度")
        print("     2. 适当放宽降级条件")
    else:
        print("  ✅ importance分布合理")
    print("=" * 80)


def show_examples(input_file, num_examples=5):
    """展示调整示例"""
    print("\n" + "=" * 80)
    print("📋 调整示例:")
    print("=" * 80)
    
    chunks = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    
    shown = 0
    for chunk in chunks:
        original = chunk['importance']
        adjusted = adjust_importance(chunk)
        
        if original != adjusted and shown < num_examples:
            print(f"\nChunk {chunk['chunk_id']}:")
            print(f"  原importance: {original} → 调整后: {adjusted}")
            print(f"  类型: {', '.join(chunk['types'])}")
            print(f"  关键词: {', '.join(chunk.get('keywords', []))}")
            print(f"  摘要: {chunk.get('summary', '')}")
            print(f"  内容: {chunk['content'][:80]}...")
            shown += 1
        
        if shown >= num_examples:
            break
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    import sys
    
    input_file = r"D:\rag\venv\逆天邪神_自动标注版.jsonl"
    output_file = r"D:\rag\venv\逆天邪神_自动标注版_调整后.jsonl"
    
    # 检查命令行参数
    if len(sys.argv) > 1 and sys.argv[1] == "--write":
        # 实际写入
        process_file(input_file, output_file, dry_run=False)
    elif len(sys.argv) > 1 and sys.argv[1] == "--examples":
        # 展示示例
        show_examples(input_file, num_examples=10)
    else:
        # 预览模式
        print("💡 预览模式（不会修改文件）")
        print("   如需实际写入，请运行: python adjust_importance.py --write")
        print("   如需查看示例，请运行: python adjust_importance.py --examples\n")
        process_file(input_file, output_file, dry_run=True)
