# Ionogram Interpreter 用户指南

## 1. 快速安装与启动

### 1.1 获取文件
- CLI 版：`dist/IonogramCLI/`（内含 `IonogramCLI.exe` 及默认配置/模型）。
- GUI 版：`dist/IonogramGUI/`（内含 `IonogramGUI.exe`）。
- 将对应文件夹复制到任何 Windows 10/11 64 位电脑即可使用，无需安装 Python。

### 1.2 运行环境
- CPU 即可；若本机有 NVIDIA GPU，推理速度更快。
- 打包目录中预置 `my_config.minimal.yaml` 与 `result/minimal/models/STEP_10.model`，如需换模型可替换这些文件或在启动时通过 `--config` 指定。

## 2. CLI（批处理）

### 2.1 基本命令
```
IonogramCLI.exe --input D:\\ionograms --out D:\\dias_output --max-files 20 --threshold 0.35
```
常用参数：
- `--input`：单个文件或目录（支持 .png/.jpg/.pickle）。
- `--out`：结果输出目录（包含 CSV/JSON/叠加图）。
- `--threshold`：描迹阈值，默认 0.3。
- `--max-files`：限制处理数量，便于抽样。
- `--no-overlays` / `--no-project`：跳过叠加图或工程文件。
- `--config`：自定义配置；留空则自动加载随包配置。

### 2.2 输出
- `params.csv`：每个样本的 f0/h’ 与置信度。
- `summary.json`：总览信息与逐样本耗时/描迹/置信度。
- `project.ionproj`：包含描迹点、参数，可在 GUI 中继续编辑。
- `overlays/*.png`：原图叠加识别结果方便审核。

## 3. GUI（交互编辑）

### 3.1 启动方式
- 双击 `IonogramGUI.exe`，或在命令行运行 `IonogramGUI.exe --input D:\\ionograms --max-files 20`。
- `--check` 仅做依赖检查，不启动界面。

### 3.2 主要区域
- **文件列表**：左侧列出已导入文件，支持多选导入。
- **画布**：居中显示原图与描迹。操作：
  - `Add` 模式：左键点击添加控制点。
  - `Select` 模式：左键拖动移动节点，右键删除。
- **侧栏**：切换层（E/F1/F2）、工具模式、导出叠加图，参数表实时显示各层 f0/h’、置信度与来源（auto/edited）。
- **工具栏**：`Open` 导入、`Run Inference` 自动判读、`Save/Load Project` 保存/加载 `.ionproj`、`Export CSV`、撤销/重做、阈值调节。

### 3.3 标准流程
1. 点击 `Open` 选择原始图片或 `.pickle`。
2. 选中目标文件后点击 `Run Inference`，待状态栏提示完成。
3. 如需修正，切换到 `Add`/`Select` 模式对描迹节点进行增删/拖拽；可使用 `Undo/Redo` 回溯。
4. 通过 `Save Project` 保存当前工程，或 `Export CSV` / `Overlay PNG` 输出参数与可视化。

### 3.4 常见问题
- **Run Inference 无响应**：
  1. 确认随包的 `result/minimal/models/STEP_10.model` 未被移动。
  2. TensorFlow 首次加载需数秒，请耐心等待；出错会弹出对话框并在终端打印日志。
- **描迹质量不佳**：尝试调整阈值（0.2~0.6），或手动添加/拖拽节点。
- **保存失败**：确保对本地可写路径操作，网络盘/只读介质可能导致写入异常。

## 4. 自定义模型与配置
1. 将新的 `STEP_xxx.model`（含 `.index`、`.data-*`）放入某个目录，如 `models/custom/`。
2. 复制 `my_config.minimal.yaml`，修改 `Test.ModelPath`（例如 `models/custom/STEP_200000.model`）。
3. 启动 CLI/GUI 时用 `--config 新配置`；若将新配置命名为 `my_config.minimal.yaml` 并与 exe 同目录，程序会自动加载。

## 5. 日志与排障
- CLI/GUI 在控制台输出日志；若需要图像界面但无终端，可运行 `IonogramGUI.exe > log.txt 2>&1` 收集日志。
- 若 GPU 显存不足，可在 CLI 加 `--gpu-id -1`，GUI 可暂时禁用显卡或在配置中设置相同参数。
- 遇到问题，建议提供输入文件、`summary.json`/`project.ionproj` 以及日志，方便定位。

祝使用顺利！
