# 摩尔线程 MUSA 平台支持模型列表

本页收录本仓库的摩尔线程 MUSA 模型教程。目前提供 Qwen3-4B 的 Transformers 推理教程和配套脚本，涵盖思考模式、非思考模式及流式输出。

支持范围按具体模型、任务和后端列出。模型训练、其他模型尺寸、量化方式及推理后端，需要各自的教程和运行记录。

## 目录

- [硬件与软件环境](#硬件与软件环境)
- [模型教程列表](#模型教程列表)
- [环境配置与运行](#环境配置与运行)
- [训练与其他推理后端](#训练与其他推理后端)
- [常见问题](#常见问题)
- [贡献教程](#贡献教程)

## 硬件与软件环境

现有教程记录的环境如下，详细配置见 [MUSA 环境说明](./models_musa/README.md)。

| 项目 | 教程环境 |
| --- | --- |
| GPU | MTT S5000，单卡显存 80 GB，机器共 8 卡 |
| 操作系统 | Ubuntu 22.04.5 LTS |
| 驱动 | 3.3.5-server |
| MUSA SDK | 4.3.8 |
| 基础镜像 | `registry.mthreads.com/mcconline/training-suite:v2.1.7` |
| Python | 3.10.12 |
| PyTorch / TorchMUSA | `torch 2.7.1.post1` / `torch_musa 2.7.1.post1` |
| Transformers | `4.57.0+musa.p34` |

表中是教程使用的整机配置。实际显存需求取决于模型尺寸、数据类型、输入长度和生成长度；运行时按需选择可见设备。

## 模型教程列表

### Qwen3

[Qwen3 官方仓库](https://github.com/QwenLM/Qwen3)

| 模型 | 任务 | 后端 | 教程 | 配套脚本 |
| --- | --- | --- | --- | --- |
| Qwen3-4B | 文本生成推理 | Transformers，BF16 | [Qwen3-4B Transformers 推理教程](./models_musa/qwen3/01-Qwen3-4B-transformers推理教程.md) | [qwen3_transformers.py](./models_musa/qwen3/qwen3_transformers.py) |

教程中记录了模型加载和生成输出。本页按该教程收录支持范围；更换环境或模型后，应重新验证。

## 环境配置与运行

### 1. 检查设备并启动容器

在宿主机执行 `mthreads-gmi`，确认 GPU 和驱动状态。Docker 需配置摩尔线程容器运行时，容器启动命令见 [MUSA 环境说明：启动容器](./models_musa/README.md#启动容器)。

使用教程指定的镜像及其内置依赖；更换 PyTorch、TorchMUSA 或 Transformers 版本后，需要重新检查兼容性。

### 2. 检查 Python 环境

在容器内执行：

```python
import torch
import torch_musa

print("torch:", torch.__version__)
print("torch_musa:", torch_musa.__version__)
print("MUSA available:", torch.musa.is_available())
print("MUSA device count:", torch.musa.device_count())
```

确认 `MUSA available` 为 `True`，且可见设备数量大于 0，再运行模型。

### 3. 准备权重并运行示例

将本仓库放到容器可访问的目录，从仓库根目录执行：

```bash
cd models_musa/qwen3

python3 - <<'PY'
from modelscope import snapshot_download
snapshot_download('Qwen/Qwen3-4B', cache_dir='.')
PY

python3 qwen3_transformers.py
```

脚本使用相对路径 `Qwen/Qwen3-4B`，按当前工作目录解析。已有权重可参考 [模型权重约定](./models_musa/README.md#模型权重) 设置软链接。

## 训练与其他推理后端

当前 `models_musa/` 提供的是 Qwen3-4B Transformers 推理教程，尚未收录 MUSA 训练、vLLM 或 SGLang 模型教程。

基础镜像中安装了训练框架或推理引擎，只能说明依赖已提供。增加 LoRA、全量微调、强化学习、量化或服务化支持时，需要补充对应入口和运行结果。

## 常见问题

### 无法识别 MUSA 设备怎么办？

先检查宿主机的 `mthreads-gmi`，再检查容器运行时、`MTHREADS_VISIBLE_DEVICES` 和 Python 环境。容器中的程序可通过 `MUSA_VISIBLE_DEVICES` 进一步限制可见设备。

### 可以直接把 CUDA 教程当作 MUSA 教程使用吗？

迁移时需要检查设备 API、依赖包及自定义算子的兼容性，并在 MUSA 环境中运行验证。CUDA 教程的运行结果不能作为 MUSA 支持记录。

## 贡献教程

新增教程请放在 `models_musa/<模型目录>/`，并在本页补充链接。提交时写明模型名称、任务、硬件、镜像与依赖版本、权重和数据准备步骤、运行命令及结果。

训练教程还应记录训练步数、loss、检查点保存与加载情况；推理教程应给出输入和生成结果。未完成的验证请注明。
