# 仓库指南

## 项目结构与模块组织
- `dias/` — 核心包：`train.py`、`test.py`、`eval.py`，以及子模块 `model/`（网络结构）与 `dataIO/`（加载、预处理、后处理）。
- `dias_main.py` — 命令行入口；通过 YAML 配置分派到训练/测试/评估。
- `dataset/` — 数据根目录，包含 `train.lst` 与 `test.lst` 索引文件。
- `result/` — 输出目录：模型权重、可视化图片、评估指标等。
- `example_config.yaml`、`my_config.yaml` — 配置模板。配置中优先使用相对项目的路径。
- `convert_GRM_to_input/` — 用于将原始 GRM 数据转换的脚本/笔记本。

## 构建、测试与开发命令
- 创建环境（Python 3.6+、TensorFlow 2）：`pip install tensorflow segmentation-models pyyaml matplotlib`
- 训练：`python dias_main.py --train --gpu_id 0 --config-file my_config.yaml`
- 测试：`python dias_main.py --test --gpu_id 0 --config-file my_config.yaml`
- 评估：`python dias_main.py --eval --config-file my_config.yaml`
说明：通过 `--gpu_id` 设置 `CUDA_VISIBLE_DEVICES`。确保配置中的目录（如 `ModelSaveDir`、`ImgSaveDir`）存在或可创建。

## 代码风格与命名约定
- Python 风格：PEP 8；4 空格缩进；建议行宽 ~100。
- 命名：函数/变量用 `snake_case`，类用 `CamelCase`，常量用 `UPPER_CASE`。
- 配置键遵循现有 YAML 风格（分组用 TitleCase，字段名清晰）。避免硬编码绝对路径，优先相对项目路径。

## 测试指南
- 功能测试以场景为驱动：
  - 使用 `--test` 生成预测与产物到 `Test.SavePath`/`Test.ImgSaveDir`。
  - 对保存结果（如 `MinHMaxF.npy`）使用 `--eval` 输出精度/召回与误差。
- 新增代码时，先在小样本列表（`dataset/test.lst`）上验证；如需提交示例产物，仅保留具有代表性的小文件（勿提交大二进制）。

## 提交与合并请求规范
- 提交信息：使用祈使句并聚焦单一改动（如：“train: fix save interval”、“dataIO: add scaler”）。
- PR 必需内容：
  - 变更摘要与动机。
  - 使用的配置（附 YAML 片段）与完整运行命令。
  - 相关的 `result/` 示例输出或截图（如适用）。
  - 关联 Issue 与向后兼容性说明。

## 安全与配置提示
- 不要提交数据集或模型权重；大文件请排除在 git 之外。
- 运行前请检查 `my_config.yaml` 路径；默认路径可能依赖本机环境。
