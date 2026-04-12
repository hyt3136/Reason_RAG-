# 重建虚拟环境 - PowerShell 脚本

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "重建虚拟环境 - 完整自动化脚本" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "警告：此脚本将重新创建虚拟环境" -ForegroundColor Yellow
Write-Host "当前虚拟环境将被重命名为 venv_old" -ForegroundColor Yellow
Write-Host ""
$confirm = Read-Host "是否继续？(y/n)"
if ($confirm -ne 'y') {
    Write-Host "已取消" -ForegroundColor Red
    exit
}

# 步骤 1: 备份数据
Write-Host ""
Write-Host "[步骤 1/5] 备份数据..." -ForegroundColor Green
Write-Host "================================================================================"
Set-Location D:\rag\venv

if (Test-Path "lang_chain\rag代码\chroma_rag_db") {
    Write-Host "备份向量数据库..."
    Copy-Item -Path "lang_chain\rag代码\chroma_rag_db" -Destination "chroma_rag_db_backup" -Recurse -Force
    Write-Host "✅ 向量数据库已备份" -ForegroundColor Green
} else {
    Write-Host "ℹ️  未找到向量数据库，跳过备份" -ForegroundColor Yellow
}

# 步骤 2: 重命名旧环境
Write-Host ""
Write-Host "[步骤 2/5] 重命名旧环境..." -ForegroundColor Green
Write-Host "================================================================================"
if (Test-Path "venv") {
    if (Test-Path "venv_old") {
        Write-Host "删除旧的 venv_old..."
        Remove-Item -Path "venv_old" -Recurse -Force
    }
    Write-Host "重命名 venv 为 venv_old..."
    Rename-Item -Path "venv" -NewName "venv_old"
    Write-Host "✅ 旧环境已保存为 venv_old" -ForegroundColor Green
} else {
    Write-Host "⚠️  未找到 venv 目录" -ForegroundColor Yellow
}

# 步骤 3: 创建新环境
Write-Host ""
Write-Host "[步骤 3/5] 创建新虚拟环境..." -ForegroundColor Green
Write-Host "================================================================================"
python -m venv venv
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ 创建虚拟环境失败" -ForegroundColor Red
    exit 1
}
Write-Host "✅ 新虚拟环境已创建" -ForegroundColor Green

# 步骤 4: 安装依赖
Write-Host ""
Write-Host "[步骤 4/5] 安装依赖包..." -ForegroundColor Green
Write-Host "================================================================================"

Write-Host "升级 pip..."
& ".\venv\Scripts\python.exe" -m pip install --upgrade pip

Write-Host ""
Write-Host "安装依赖（这可能需要几分钟）..."
Set-Location "lang_chain\rag代码"
& "..\..\venv\Scripts\pip.exe" install -r requirements_fixed.txt

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ 依赖安装失败" -ForegroundColor Red
    exit 1
}

# 步骤 5: 验证
Write-Host ""
Write-Host "[步骤 5/5] 验证安装..." -ForegroundColor Green
Write-Host "================================================================================"

& "..\..\venv\Scripts\python.exe" -c "import chromadb; print('✅ ChromaDB:', chromadb.__version__)"
& "..\..\venv\Scripts\python.exe" -c "import protobuf; print('✅ Protobuf:', protobuf.__version__)"
& "..\..\venv\Scripts\python.exe" -c "from langchain_chroma import Chroma; print('✅ LangChain-Chroma: OK')"
& "..\..\venv\Scripts\python.exe" -c "from google.protobuf.internal import builder; print('✅ Protobuf builder: OK')"

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ 验证失败" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "✅ 环境重建完成！" -ForegroundColor Green
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "现在可以运行：" -ForegroundColor Yellow
Write-Host "  python main.py" -ForegroundColor White
Write-Host ""
Write-Host "如果需要恢复旧环境：" -ForegroundColor Yellow
Write-Host "  1. 删除新的 venv 目录" -ForegroundColor White
Write-Host "  2. 将 venv_old 重命名为 venv" -ForegroundColor White
Write-Host ""
