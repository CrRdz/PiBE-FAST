# PiBE-FAST

基于树莓派与本地电脑的多模态 BE-FAST 脑卒中警示体征筛查研究原型。

> **医疗安全提示**
>
> 本项目不是医疗器械，不能诊断或排除脑卒中。若突然出现失衡、视力异常、
> 面部歪斜、单侧无力或言语不清，请立即联系当地急救服务，不要等待软件完成检测。

## 论文

**Multimodal Embodied AI for Stroke Warning-Sign Assessment**

- [英文论文 PDF](docs/pibefast-paper/pibefast-conference-paper.pdf)
- [英文 LaTeX 源码与复现说明](docs/pibefast-paper/README.md)
- [论文数据索引](research_data/FINAL_DATA_TABLES.md)

## 系统概览

PiBE-FAST 将 BE-FAST 的五类警示体征拆成独立任务：

| 模块 | 输入 | 作用 |
|---|---|---|
| B — Balance | MoveNet 姿态序列、人工报告 | 失衡、跌倒序列与个人基线变化 |
| E — Eyes | MediaPipe 面部/虹膜关键点 | 标准化眼动与视力异常报告 |
| F — Face | MediaPipe 面部关键点 | 中性脸与微笑的不对称变化 |
| A — Arms | MoveNet 姿态序列 | 双臂抬举、下垂与侧别 |
| S — Speech | 本地麦克风、whisper.cpp、MDSC 表征 | 固定句质量、转写与构音异常证据 |

系统默认在本地处理摄像头与麦克风数据。树莓派可低频运行 B/S 被动变化监测；
用户主动开始筛查后，本地电脑执行 E/F/A/S 引导任务。被动信号只用于提示主动确认，
不等于卒中诊断或临床风险概率。

## 快速开始

要求 Python 3.10+、Node.js 18+，以及与平台匹配的 TensorFlow Lite 运行时。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
npm install
npm run build
```

将运行模型放入 `models/`（大型模型已由 Git 忽略）：

- `movenet_lightning.tflite`
- `face_landmarker.task`
- `ggml-base.bin`（启用言语转写时）

开发机启动：

```bash
python -m app.main \
  --model models/movenet_lightning.tflite \
  --face-model models/face_landmarker.task
```

浏览器打开 <http://localhost:8080>。视频文件可通过
`--source /path/to/video.mp4` 代替摄像头；完整参数见：

```bash
python -m app.main --help
```

树莓派安装脚本：

```bash
./scripts/install_pi.sh
./scripts/install_speech_pi.sh   # 可选：本地语音能力
```

手机浏览器调用摄像头需要可信 HTTPS，可通过 `--web-cert` 与 `--web-key` 配置。
采集记录、历史图片/音频、模型权重、构建缓存和缩略图默认不会进入 Git。

## 测试

```bash
python -m unittest discover -s tests
npm run build
```

测试覆盖 BE-FAST 会话状态、融合与紧急度、被动监测、语音、历史记录、Web API
和结果渲染。硬件摄像头、麦克风和目标设备性能仍需在部署环境单独验证。

## 论文与数据复现

主要入口：

```bash
python scripts/analyze_human_repeatability.py
python scripts/analyze_pipc_pilot.py
python scripts/analyze_validation_workbook.py
python scripts/generate_paper_evidence_figures.py
```

数据边界、样本限制和来源完整性记录在各数据目录的 README 与
[`docs/pibefast-paper/supporting-notes.md`](docs/pibefast-paper/supporting-notes.md)。
仓库中的参与者数据为去标识化特征级记录，不包含原始视频或音频。

英文论文构建：

```bash
cd docs/pibefast-paper
latexmk -interaction=nonstopmode -halt-on-error main.tex
```

LaTeX 中间文件写入 `build/` 并由 Git 忽略；命名后的交付 PDF 保留在仓库中。

## 目录

```text
app/                         Python 服务、推理与 API
frontend/                    React 前端源码
tests/                       单元与工作流回归测试
models/                      小型模型元数据；大型权重忽略
training/                    语音与多模态训练代码
research_data/               去标识化论文数据与来源记录
experiments/                 保留的最终验证证据
docs/pibefast-paper/          英文论文、正式图与补充说明
scripts/                     部署、分析与复现入口
```

## 重要限制

- 当前阈值是研究和工程阈值，不是临床决策阈值。
- MDSC 模型识别构音障碍表型，不是急性卒中诊断器。
- 健康参与者、脚本化事件和小规模 pilot 不能证明临床敏感度或特异度。
- 系统不会自动联系急救服务；任何告警都需要人员确认和既定处置流程。

## English

PiBE-FAST is a local-first research prototype for multimodal BE-FAST warning-sign
assessment. It combines MoveNet pose sequences, MediaPipe facial landmarks,
guided speech capture, and explicit symptom/onset reports. It is not a medical
device and must not delay emergency care.

Read the [English paper](docs/pibefast-paper/pibefast-conference-paper.pdf) or
the [paper build notes](docs/pibefast-paper/README.md). Setup, commands, data
boundaries, and repository layout are documented above; command-line options are
available through `python -m app.main --help`.
