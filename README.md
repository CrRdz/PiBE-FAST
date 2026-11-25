# PiBE-FAST

运行于树莓派与本地电脑的多模态健康监测研究原型。树莓派持续监测平衡与言语变化，本地电脑引导用户完成 BE-FAST 评估，并汇总测量结果、紧急程度和完成状态。

[论文 PDF](docs/pibefast-paper/pibefast-conference-paper.pdf) · [论文源码与构建](docs/pibefast-paper/README.md) · [数据索引](research_data/FINAL_DATA_TABLES.md)

## 快速开始

环境：Python 3.10+、Node.js 18+，以及适配平台的 TensorFlow Lite 运行时。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
npm install
npm run build
```

在 `models/` 中放入 `movenet_lightning.tflite` 和 `face_landmarker.task`；启用本地语音转写时另需 `ggml-base.bin`。

```bash
python -m app.main \
  --model models/movenet_lightning.tflite \
  --face-model models/face_landmarker.task
```

打开 [localhost:8080](http://localhost:8080)。使用视频文件时添加 `--source /path/to/video.mp4`。

## 常用命令

- 参数说明：`python -m app.main --help`
- 树莓派安装：`./scripts/install_pi.sh`
- 测试：`python -m unittest discover -s tests`
- 前端构建：`npm run build`
