# Git 维护与发布

`py/just_etc/` 保存 Python 包；`examples/` 保存科研工作流；`doc/` 保存文档。
修改仪器参数时同步检查 `etc/spec.dat` 与包内默认配置。



## 创建分支

建议每次按“**创建分支 → 修改 → 测试 → 提交 → 推送 → 合并**”操作。

**1. 更新主分支，然后创建工作分支**

在终端执行：

```
cd /Users/rain/just_etc

git switch main
git pull --ff-only

git switch -c branch-name
```

`branch-name` 是示例名称，可换成能描述本次工作的名称。创建后会自动切换到该分支。

**2. 修改文件并测试**

直接用编辑器修改这个目录里的文件。主要位置是：

| 想修改的内容         | 位置                                                   |
| -------------------- | ------------------------------------------------------ |
| Python API、计算代码 | `py/just_etc/`                                         |
| 科研示例             | `examples/`                                            |
| 文档                 | `doc/`                                                 |
| 默认仪器参数         | `py/just_etc/data/spec.dat`，并同步更新 `etc/spec.dat` |

首次在这个克隆目录开发时，在科研环境中做一次“可编辑安装”：

```
conda activate base
python -m pip install -e . --no-deps
```

之后修改 Python 源码通常不需要重新安装。完成修改后运行：

```
python -m pytest py/just_etc/test -q
```

**3. 检查并提交修改**

```
git status
git diff

# 将本次修改的文件加入暂存区，按实际文件调整
git add py/just_etc/just_etc_api.py doc/usage.md

# 将暂存内容保存为一个本地提交
git commit -m "Improve SNR calculation and update documentation"
```

`git commit` 只保存到本地，尚未上传 GitHub。

**4. 上传工作分支**

第一次推送这个分支：

```
git push -u origin branch-name
```

以后在同一个分支继续修改，重复 `git add`、`git commit` 后，只需：

```
git push
```

此时 GitHub 上会出现并更新 `branch-name` 分支，`main` 还没有改变。

**5. 在 GitHub 合并，再同步本地**

打开[仓库](https://github.com/RainW7/just_etc)，点击 **Compare & pull request**，确认：

- **base：`main`**
- **compare：`branch-name`**

创建 Pull Request，检查差异和自动测试结果，通过后点击 **Merge pull request**。随后回到终端：

```
git switch main
git pull --ff-only
```

下一项工作再从这个更新后的 `main` 创建新分支即可。



## 日常修改

```sh
git status
git pull --ff-only
git switch -c improve-etc
python -m pip install -e '.[test]'
# 修改代码和文档后：
python -m pytest py/just_etc/test
git diff
git add py/just_etc doc
git commit -m "Describe the concrete change"
git push -u origin improve-etc
```

然后在 GitHub 创建 Pull Request，检查修改和测试后合并。每次新工作从最新
`main` 建立分支；已有同名分支时用 `git switch` 切换。`git add` 只添加本次
需要的文件，避免把数据输出或缓存一起提交。不要用强制推送覆盖远端历史。



## 打包

在有构建工具的环境中：

```sh
python -m pip install build
python -m build
```

版本号位于 `py/just_etc/_version.py`。发布新版本时更新版本和变更说明，完成
测试后再创建版本标签；不要将 GitHub 上传与 PyPI 发布混为一谈。

## 自动创建 GitHub Release

`.github/workflows/release.yml` 在推送符合 `v*.*.*` 格式的 tag 时自动执行：

1. 检查 tag 版本与 `just_etc.__version__` 完全一致；
2. 在 Python 3.10、3.11 和 3.12 上运行测试；
3. 构建 source distribution 和 wheel；
4. 验证 wheel 中的 `just-etc` 命令；
5. 创建 GitHub Release、自动生成 release notes，并上传 `dist/` 中的构建产物。

例如发布 1.2.0 时，先在工作分支中将 `py/just_etc/_version.py` 更新为：

```python
__version__ = "1.2.0"
```

提交、测试并通过 Pull Request 合并到 `main` 后执行：

```sh
git switch main
git pull --ff-only
git tag -a v1.2.0 -m "JUST ETC v1.2.0"
git push origin v1.2.0
```

只有最后一条 tag push 会触发正式发布。普通 branch push 和 Pull Request 不会创建
Release。tag 和包版本不一致、测试失败或构建失败时，workflow 会停止且不会发布。
不要移动或覆盖已经发布的版本 tag；修复后应使用新的 patch 版本，例如 1.1.1。

macOS 默认文件系统通常不区分大小写： `JUST_ETC` 与 `just_etc` 可能是
同一目录。克隆到新目录时选择明确不同的名称，避免混入历史工作文件。
