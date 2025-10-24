+++
date = '2025-10-23'
title = 'Brew 换源'
+++

原文：

- https://mirrors.ustc.edu.cn/help/brew.git.html
- https://mirrors.ustc.edu.cn/help/homebrew-bottles.html
- https://mirrors.ustc.edu.cn/help/homebrew-core.git.html
- https://mirrors.ustc.edu.cn/help/homebrew-cask.git.html 

## ✅ 一、设置环境变量（永久生效，针对 fish）
在 fish 中，永久设置环境变量应使用   set -Ux （全局、导出、持久化）：
（set文档： https://fishshell.com/docs/current/cmds/set.html ）

```sh
# Homebrew 主程序仓库
set -Ux HOMEBREW_BREW_GIT_REMOTE https://mirrors.ustc.edu.cn/brew.git

# Homebrew 核心公式仓库
set -Ux HOMEBREW_CORE_GIT_REMOTE https://mirrors.ustc.edu.cn/homebrew-core.git

# 预编译二进制包（bottles）域名
set -Ux HOMEBREW_BOTTLE_DOMAIN https://mirrors.ustc.edu.cn/homebrew-bottles

# 元数据 API 域名（Brew 4.0+ 必需）
set -Ux HOMEBREW_API_DOMAIN https://mirrors.ustc.edu.cn/homebrew-bottles/api
```


## ✅ 二、配置 Homebrew Cask 使用镜像（Git 仓库方式）
由于Brew 4.0+ 默认使用 JSON API，大多数情况下不需要手动设置 cask 的 Git 镜像。但如果你仍希望显式使用 USTC 的 cask Git 仓库（例如离线环境或调试），请运行：
```sh
brew tap --custom-remote homebrew/cask https://mirrors.ustc.edu.cn/homebrew-cask.git
```


> ⚠️ 注意：如果你以后想恢复官方源，可运行：
    ```
    brew tap --custom-remote homebrew/cask https://github.com/Homebrew/homebrew-cask
    ```

## ✅ 三、验证设置是否生效
检查环境变量：
```sh
echo $HOMEBREW_BREW_GIT_REMOTE
echo $HOMEBREW_CORE_GIT_REMOTE
echo $HOMEBREW_BOTTLE_DOMAIN
echo $HOMEBREW_API_DOMAIN
```


更新 Homebrew：
```sh
brew update
```

如果下载速度很快（且无 GitHub 超时），说明 bottles 已走镜像。



（如果首次安装Homebrew）
如果你尚未安装 Homebrew，可先设置上述环境变量，再运行安装脚本：

```sh
set -Ux HOMEBREW_BREW_GIT_REMOTE https://mirrors.ustc.edu.cn/brew.git
set -Ux HOMEBREW_CORE_GIT_REMOTE https://mirrors.ustc.edu.cn/homebrew-core.git
set -Ux HOMEBREW_BOTTLE_DOMAIN https://mirrors.ustc.edu.cn/homebrew-bottles
set -Ux HOMEBREW_API_DOMAIN https://mirrors.ustc.edu.cn/homebrew-bottles/api
```


## 使用官方安装脚本（若无法访问 GitHub，改用 USTC 备份脚本）

```sh
/bin/bash -c "$(curl -fsSL https://mirrors.ustc.edu.cn/misc/brew-install.sh)"
```









