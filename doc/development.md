# Git 维护与发布

`py/just_etc/` 保存 Python 包；`examples/` 保存科研工作流；`doc/` 保存文档。
修改仪器参数时同步检查 `etc/spec.dat` 与包内默认配置。

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

原 Python 3.6 环境也可以使用已有 setuptools/wheel：

```sh
python setup.py sdist bdist_wheel
```

版本号位于 `py/just_etc/_version.py`。发布新版本时更新版本和变更说明，完成
测试后再创建版本标签；不要将 GitHub 上传与 PyPI 发布混为一谈。

macOS 默认文件系统通常不区分大小写：本机的 `JUST_ETC` 与 `just_etc` 可能是
同一目录。克隆到新目录时选择明确不同的名称，避免混入历史工作文件。
