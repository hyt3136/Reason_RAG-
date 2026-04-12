"""分析切分数据质量"""
import json
from collections import Counter

def analyze_jsonl(file_path, sample_size=50):
    """分析JSONL文件的数据质量"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= sample_size:
                break
            if line.strip():
                data.append(json.loads(line))
    
    if not data:
        print("❌ 文件为空或无法读取")
        return
    
    print("=" * 80)
    print(f"📊 数据质量分析报告（样本数: {len(data)}）")
    print("=" * 80)
    
    # 1. 基本统计
    print("\n【基本统计】")
    print(f"  总chunk数: {len(data)}")
    avg_len = sum(len(d['content']) for d in data) / len(data)
    print(f"  平均长度: {avg_len:.1f} 字符")
    min_len = min(len(d['content']) for d in data)
    max_len = max(len(d['content']) for d in data)
    print(f"  长度范围: {min_len} - {max_len} 字符")
    
    # 2. Importance分布
    print("\n【重要度分布】")
    importance_dist = Counter(d['importance'] for d in data)
    total = len(data)
    for imp, count in importance_dist.most_common():
        print(f"  {imp}: {count} ({count/total*100:.1f}%)")
    
    # 3. Types分布
    print("\n【类型分布】")
    all_types = []
    for d in data:
        all_types.extend(d['types'])
    type_dist = Counter(all_types)
    for typ, count in type_dist.most_common():
        print(f"  {typ}: {count} ({count/len(data)*100:.1f}%)")
    
    # 4. Characters统计
    print("\n【角色统计】")
    all_chars = []
    for d in data:
        all_chars.extend(d['characters'])
    char_dist = Counter(all_chars)
    print(f"  总角色数: {len(char_dist)}")
    print(f"  Top 10角色:")
    for char, count in char_dist.most_common(10):
        print(f"    {char}: {count}次")
    
    # 5. 数据质量问题
    print("\n【潜在问题】")
    issues = []
    
    # 检查过短/过长
    too_short = sum(1 for d in data if len(d['content']) < 100)
    too_long = sum(1 for d in data if len(d['content']) > 300)
    if too_short > 0:
        issues.append(f"  ⚠️ {too_short} 个chunk过短（<100字符）")
    if too_long > 0:
        issues.append(f"  ⚠️ {too_long} 个chunk过长（>300字符）")
    
    # 检查默认值
    default_chars = sum(1 for d in data if d['characters'] == ['未知'])
    if default_chars > 0:
        issues.append(f"  ⚠️ {default_chars} 个chunk角色为默认值'未知'")
    
    # 检查importance分布
    high_ratio = importance_dist.get('high', 0) / total
    if high_ratio > 0.7:
        issues.append(f"  ⚠️ high重要度占比过高（{high_ratio*100:.1f}%），可能标注不准确")
    elif high_ratio < 0.2:
        issues.append(f"  ⚠️ high重要度占比过低（{high_ratio*100:.1f}%），可能遗漏重要内容")
    
    if issues:
        for issue in issues:
            print(issue)
    else:
        print("  ✅ 未发现明显问题")
    
    # 6. 示例展示
    print("\n【示例chunk】")
    for i in range(min(3, len(data))):
        d = data[i]
        print(f"\n  Chunk {d['chunk_id']}:")
        print(f"    角色: {', '.join(d['characters'])}")
        print(f"    类型: {', '.join(d['types'])}")
        print(f"    关键词: {', '.join(d['keywords'])}")
        print(f"    重要度: {d['importance']}")
        print(f"    摘要: {d['summary']}")
        print(f"    内容: {d['content'][:80]}...")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    file_path = r"D:\rag\venv\逆天邪神_自动标注版.jsonl"
    analyze_jsonl(file_path, sample_size=100)
