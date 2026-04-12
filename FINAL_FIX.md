# 最终解决方案 - 重新创建虚拟环境

由于依赖冲突太复杂，最可靠的方法是重新创建虚拟环境。

## 步骤1：备份当前数据

```powershell
# 备份向量数据库
cp -r D:\rag\venv\lang_chain\rag代码\chroma_rag_db D:\rag\venv\lang_chain\rag代码\chroma_rag_db_backup

# 备份标注文件
cp D:\rag\venv\逆天邪神_自动标注版_最终.jsonl D:\rag\venv\逆天邪神_自动标注版_最终_backup.jsonl
```

## 步骤2：创建新的虚拟环境

```powershell
# 进入项目目录
cd D:\rag\venv

# 重命名旧环境（保留备份）
Rename-Item venv venv_old

# 创建新环境
python -m venv venv_new
Rename-Item venv_new venv

# 激活新环境
.\venv\Scripts\activate
```

## 步骤3：按正确顺序安装依赖

```powershell
# 1. 先安装基础依赖
pip install numpy==1.24.0
pip install pydantic==2.10.6

# 2. 安装 protobuf（固定版本）
pip install protobuf==4.25.3

# 3. 安装 opentelemetry（固定版本）
pip install opentelemetry-api==1.21.0
pip install opentelemetry-sdk==1.21.0
pip install opentelemetry-proto==1.21.0
pip install opentelemetry-exporter-otlp-proto-grpc==1.21.0

# 4. 安装其他依赖
pip install deprecated==1.3.1
pip install "importlib-metadata<7.0,>=6.0"

# 5. 安装 chromadb（不升级已安装的包）
pip install chromadb==0.4.22 --no-deps
pip install chromadb==0.4.22

# 6. 安装 langchain 相关
pip install langchain==0.1.20
pip install langchain-core==0.1.52
pip install langchain-chroma==0.1.2
pip install langchain-huggingface==0.0.1
pip install langchain-community==0.0.38
pip install langchain-text-splitters==0.0.1

# 7. 安装 embedding 和其他工具
pip install transformers==4.30.0
pip install torch==2.0.0
pip install sentence-transformers==2.2.0
pip install jieba==0.42.1
pip install requests==2.28.0
```

## 步骤4：验证安装

```powershell
python -c "import chromadb; print('ChromaDB:', chromadb.__version__)"
python -c "import protobuf; print('Protobuf:', protobuf.__version__)"
python -c "from langchain_chroma import Chroma; print('LangChain-Chroma: OK')"
```

## 步骤5：测试运行

```powershell
cd lang_chain\rag代码
python main.py
```

## 如果仍然失败

使用 requirements_fixed.txt：

```powershell
pip install -r requirements_fixed.txt
```

## 恢复数据

```powershell
# 如果需要恢复向量数据库
cp -r D:\rag\venv\lang_chain\rag代码\chroma_rag_db_backup\* D:\rag\venv\lang_chain\rag代码\chroma_rag_db\
```
