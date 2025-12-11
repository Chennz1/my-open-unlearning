#!/bin/bash

# 这个脚本用于将当前目录转换为你自己的 git 仓库
# 使用方法: ./setup_new_repo.sh <你的GitHub用户名> <新仓库名>

set -e

if [ $# -ne 2 ]; then
    echo "使用方法: $0 <GitHub用户名> <新仓库名>"
    echo "例如: $0 myusername my-open-unlearning"
    exit 1
fi

GITHUB_USER=$1
REPO_NAME=$2

echo "================================================"
echo "准备将当前目录设置为新的 git 仓库"
echo "GitHub 用户名: $GITHUB_USER"
echo "新仓库名: $REPO_NAME"
echo "================================================"

# 1. 备份当前的 .git 目录（可选，以防万一）
if [ -d ".git" ]; then
    echo "步骤 1: 备份原 .git 目录..."
    mv .git .git.backup.$(date +%Y%m%d_%H%M%S)
    echo "✓ 已备份原 .git 目录"
else
    echo "步骤 1: 未发现 .git 目录，跳过备份"
fi

# 2. 初始化新的 git 仓库
echo "步骤 2: 初始化新的 git 仓库..."
git init
echo "✓ 已初始化新仓库"

# 3. 添加所有文件
echo "步骤 3: 添加所有文件到暂存区..."
git add .
echo "✓ 已添加文件"

# 4. 创建初始提交
echo "步骤 4: 创建初始提交..."
git commit -m "Initial commit: Fork from locuslab/open-unlearning"
echo "✓ 已创建初始提交"

# 5. 创建 main 分支（如果不存在）
echo "步骤 5: 确保在 main 分支..."
git branch -M main
echo "✓ 已设置为 main 分支"

# 6. 添加你的远程仓库
echo "步骤 6: 添加远程仓库..."
git remote add origin "https://github.com/$GITHUB_USER/$REPO_NAME.git"
echo "✓ 已添加远程仓库: https://github.com/$GITHUB_USER/$REPO_NAME.git"

echo ""
echo "================================================"
echo "设置完成！"
echo "================================================"
echo ""
echo "接下来的步骤："
echo "1. 在 GitHub 上创建新仓库: https://github.com/new"
echo "   - 仓库名: $REPO_NAME"
echo "   - 不要初始化 README、.gitignore 或 license"
echo ""
echo "2. 创建仓库后，运行以下命令推送代码："
echo "   git push -u origin main"
echo ""
echo "3. (可选) 如果想要保留对原仓库的引用，可以添加 upstream:"
echo "   git remote add upstream https://github.com/locuslab/open-unlearning.git"
echo ""
echo "当前远程仓库配置:"
git remote -v
echo ""
