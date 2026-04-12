"""检查实际生成的数据是否有overlap"""
import json

file_path = r"D:\rag\venv\逆天邪神_自动标注版_优化.jsonl"

# 读取前10个chunk
chunks = []
with open(file_path, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 10:
            break
        if line.strip():
            chunks.append(json.loads(line))

print("=" * 80)
print("检查实际生成数据的overlap情况")
print("=" * 80)

# 显示前5个chunk的内容
print("\n【前5个chunk内容】\n")
for chunk in chunks[:5]:
    print(f"Chunk {chunk['chunk_id']} (长度: {len(chunk['content'])}):")
    print(f"  开头: {chunk['content'][:50]}...")
    print(f"  结尾: ...{chunk['content'][-50:]}")
    print()

# 检查overlap
print("=" * 80)
print("【Overlap检查】\n")

has_overlap_count = 0
total_pairs = len(chunks) - 1

for i in range(1, len(chunks)):
    prev_content = chunks[i-1]['content']
    curr_content = chunks[i]['content']
    
    print(f"Chunk {i} → Chunk {i+1}:")
    
    # 查找重叠
    max_overlap = 0
    overlap_text = ""
    
    # 从长到短查找重叠
    for length in range(min(len(prev_content), len(curr_content)), 0, -1):
        if prev_content[-length:] == curr_content[:length]:
            max_overlap = length
            overlap_text = curr_content[:length]
            break
    
    if max_overlap > 0:
        print(f"  ✅ 有 {max_overlap} 字符重叠")
        print(f"  重叠内容: {overlap_text[:60]}...")
        has_overlap_count += 1
    else:
        print(f"  ❌ 无重叠")
        print(f"  Chunk {i} 结尾: ...{prev_content[-60:]}")
        print(f"  Chunk {i+1} 开头: {curr_content[:60]}...")
    print()

print("=" * 80)
print("【统计】")
print(f"  总chunk对数: {total_pairs}")
print(f"  有overlap的对数: {has_overlap_count}")
print(f"  overlap覆盖率: {has_overlap_count/total_pairs*100:.1f}%")
print("=" * 80)
