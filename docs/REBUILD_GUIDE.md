# 虚拟环境重建指南

## 🎯 快速开始

### 方法1：使用批处理脚本（推荐）

```powershell
cd D:\rag\venv\lang_chain\rag代码
.\rebuild_env.bat
```

### 方法2：使用 PowerShell 脚本

```powershell
cd D:\rag\venv\lang_chain\rag代码
powershell -ExecutionPolicy Bypass -File .\rebuild_env.ps1
```

### 方法3：手动执行

```powershell
# 1. 备份数据
cd D:\rag\venv
xcopy /E /I /Y "lang_chain\rag代码\chroma_rag_db" "chroma_rag_db_backup"

# 2. 重命名旧环境
ren venv venv_old

# 3. 创建新环境
python -m venv venv
.\venv\Scripts\activate

# 4. 安装依赖
cd lang_chain\rag代码
pip install -r requirements_fixed.txt

# 5. 测试
python main.py
```

## 📋 脚本功能

自动化脚本会执行以下操作：

1. ✅ **备份向量数据库** - 保存到 `chroma_rag_db_backup`
2. ✅ **保留旧环境** - 重命名为 `venv_old`（可以随时恢复）
3. ✅ **创建新环境** - 全新的虚拟环境
4. ✅ **安装依赖** - 使用固定版本避免冲突
5. ✅ **验证安装** - 确保所有包正常工作

## ⏱️ 预计时间

- 备份数据：10秒
- 创建环境：30秒
- 安装依赖：3-5分钟（取决于网速）
- 验证：10秒

**总计：约5分钟**

## 🔍 验证成功的标志

脚本运行成功后，你会看到：

```
✅ ChromaDB: 0.4.22
✅ Protobuf: 4.25.3
✅ LangChain-Chroma: OK
✅ Protobuf builder: OK
```

## 🚀 运行程序

环境重建完成后：

```powershell
# 确保在正确的目录
cd D:\rag\venv\lang_chain\rag代码

# 激活虚拟环境
..\..\venv\Scripts\activate

# 运行程序
python main.py
```

## 🔄 如何恢复旧环境

如果新环境有问题，可以恢复旧环境：

```powershell
cd D:\rag\venv

# 删除新环境
rmdir /S /Q venv

# 恢复旧环境
ren venv_old venv

# 恢复数据（如果需要）
xcopy /E /I /Y "chroma_rag_db_backup" "lang_chain\rag代码\chroma_rag_db"
```

## ❓ 常见问题

### Q1: 脚本运行失败怎么办？

**A**: 检查以下几点：
1. 是否有管理员权限
2. Python 是否在 PATH 中
3. 是否有足够的磁盘空间（至少2GB）

### Q2: 安装依赖时出现错误？

**A**: 可能是网络问题，尝试：
```powershell
pip install -r requirements_fixed.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Q3: 验证失败怎么办？

**A**: 查看具体的错误信息，可能需要：
1. 检查 Python 版本（需要 3.10+）
2. 重新运行脚本
3. 手动安装失败的包

### Q4: 旧环境可以删除吗？

**A**: 确认新环境完全正常后，可以删除：
```powershell
rmdir /S /Q D:\rag\venv\venv_old
```

## 📦 依赖版本说明

`requirements_fixed.txt` 使用以下固定版本：

- **chromadb==0.4.22** - 稳定版本
- **protobuf==4.25.3** - 与 chromadb 兼容
- **opentelemetry-*==1.21.0** - 兼容版本系列
- **langchain-chroma==0.1.2** - 最新稳定版

这些版本经过测试，互相兼容，不会产生冲突。

## 💡 提示

1. **备份重要数据** - 脚本会自动备份，但建议手动再备份一次
2. **关闭其他程序** - 确保没有其他程序在使用虚拟环境
3. **网络稳定** - 安装依赖需要下载约500MB的包
4. **耐心等待** - 安装过程可能需要几分钟

## 📞 需要帮助？

如果遇到问题，请查看：
- `FINAL_FIX.md` - 详细的故障排除指南
- `FIX_PROTOBUF_ERROR.md` - Protobuf 错误说明
