"""
依赖诊断脚本
检查当前环境中的包版本和冲突
"""
import subprocess
import sys

def get_package_version(package_name):
    """获取包版本"""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", package_name],
            capture_output=True,
            text=True
        )
        for line in result.stdout.split('\n'):
            if line.startswith('Version:'):
                return line.split(':')[1].strip()
        return "未安装"
    except:
        return "未安装"

def check_import(module_name, display_name=None):
    """检查模块是否可以导入"""
    if display_name is None:
        display_name = module_name
    
    try:
        __import__(module_name)
        return f"✅ {display_name}: 可以导入"
    except Exception as e:
        return f"❌ {display_name}: {str(e)[:50]}"

print("=" * 80)
print("📊 依赖诊断报告")
print("=" * 80)

# 检查关键包版本
packages = {
    "protobuf": "4.25.3",
    "chromadb": "0.4.22",
    "langchain-chroma": "0.1.2",
    "opentelemetry-api": "1.21.0",
    "opentelemetry-sdk": "1.21.0",
    "opentelemetry-proto": "1.21.0",
    "blackboxprotobuf": None,  # 可能导致冲突
    "googleapis-common-protos": None,
}

print("\n📦 已安装的包版本:")
print("-" * 80)

conflicts = []
for package, expected in packages.items():
    current = get_package_version(package)
    status = "✅" if current == expected or expected is None else "⚠️"
    
    if expected:
        print(f"{status} {package:30} 当前: {current:15} 期望: {expected}")
        if current != expected and current != "未安装":
            conflicts.append((package, current, expected))
    else:
        print(f"ℹ️  {package:30} 当前: {current:15} (可选)")

# 检查导入
print("\n🔍 导入测试:")
print("-" * 80)

imports = [
    ("google.protobuf.internal.builder", "protobuf.builder"),
    ("chromadb", "chromadb"),
    ("langchain_chroma", "langchain_chroma"),
    ("opentelemetry.proto.common.v1.common_pb2", "opentelemetry-proto"),
]

for module, display in imports:
    print(check_import(module, display))

# 冲突报告
if conflicts:
    print("\n⚠️ 发现版本冲突:")
    print("-" * 80)
    for package, current, expected in conflicts:
        print(f"  {package}: {current} → {expected}")
    
    print("\n💡 修复建议:")
    print("  运行: .\\fix_dependencies.bat")
    print("  或手动执行:")
    for package, current, expected in conflicts:
        print(f"    pip install {package}=={expected}")
else:
    print("\n✅ 所有包版本正确！")

print("\n" + "=" * 80)
