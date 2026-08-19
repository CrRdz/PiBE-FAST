# AISHELL-6B/MDSC 构音障碍语音表征训练与 M5 验证

更新日期：2026-08-17

## 定位

这条管线训练普通话构音障碍语音表征与二分类识别器。正类是 MDSC 的
`Uncontrol/Dysarthria`，负类是 `Control`。它不学习急性卒中标签，运行时只能输出
`shadow_dysarthria_*`，不会改变现有 BE-FAST S 项的状态或紧急决策。

AISHELL 官方页说明 MDSC 包含 18,630 条、17 小时录音：21 名构音障碍说话人和
25 名对照说话人，16 kHz、安静室内、手机麦克风约 20 cm。许可为 CC BY-NC 4.0。
官方数据包已于 2026-08-17 完成本地审计与训练。原始数据、本地 manifest、特征归档
和训练报告均由 `.gitignore` 排除；项目不会把数据或真实语音提交到 Git。

官方申请包解压后的数据结构为：

```text
lrdwws/
├── Control/{train,dev,test}/{transcript,wav}/...
└── Uncontrol/
    ├── train/{transcript,wav}/...
    ├── dev/{enrollment,eval}/{transcript,wav}/...
    └── test/{enrollment,eval}/{transcript,wav}/...
```

## 模型

- 16 kHz、16-bit PCM；运行端对多通道取平均，非 16 kHz 输入线性重采样。
- 与官方 LRDWWS 基线一致的 40-bin log-Mel、25 ms窗、10 ms移。
- 80 个逐频带均值/标准差，加10个时序、能量、停顿和谱统计，共90维。
- 开发集执行说话人分组交叉验证；部署模型只拟合 train 说话人，最终阈值只使用独立
  validation 说话人的平均概率选择。
- 完全独立的测试说话人只用于最终报告。
- 导出标准化逻辑回归 JSON；树莓派只依赖 NumPy，不依赖 PyTorch。

轻量线性模型是第一条可审计基线，不应被描述成 SOTA。后续若引入 TCN/DS-CNN，必须
保留同一 manifest、说话人切分和持出测试集，才能做公平比较。

## 当前真实训练结果

2026-08-17 的 v1 使用官方 18,630 条录音、46 名说话人。固定研究切分为
33 名训练、7 名验证和 6 名完全独立测试说话人。审计发现 18,616 条单声道和
14 条立体声录音，未丢弃样本。

| 评估层级 | ROC-AUC | 敏感度 | 特异度 | 平衡准确率 | Brier score |
|---|---:|---:|---:|---:|---:|
| 开发集折外样本（16,200 条） | 0.8241 | 0.7956 | 0.6655 | 0.7306 | 0.1942 |
| 开发集折外说话人（40 人） | 0.9015 | 0.9444 | 0.5909 | 0.7677 | 0.1272 |
| 独立测试样本（2,430 条） | 0.9573 | 0.6444 | 0.9926 | 0.8185 | 0.0983 |
| 独立测试说话人（6 人） | 1.0000 | 0.6667 | 1.0000 | 0.8333 | 0.0577 |

部署阈值只由 7 名验证说话人校准，为 `0.807859`。说话人级独立测试只有
3 名阳性和 3 名阴性，敏感度的说话人 Bootstrap 95% 区间为 `0.0–1.0`；
因此表中数字只是内部研究结果，不是临床性能。完整数值见
`training/reports/mdsc_dysarthria_v1.json`。
当前产物为 `models/mdsc_dysarthria_v1.json`，已通过运行时加载和真实 WAV 推理检查。
开发机 100 个测试窗口的推理中位时间为 0.96 ms、p95 为 1.33 ms；这不是
树莓派或目标麦克风的 M5 性能数据。当次仓库验证为 128 项测试全部通过。

## 1. 数据审计与固定切分

```bash
python training/speech/prepare_mdsc.py \
  --dataset-root data/training/lrdwws \
  --output training/manifests/mdsc.local.csv \
  --report training/reports/mdsc_audit.json
```

脚本兼容上述官方分类压缩包布局和归一化后的
`train/{Control,Uncontrol}` 布局，验证 WAV 与 transcript 一一对应以及16位 PCM 格式。
官方包中少量立体声异常样本会在共享读取器中以通道平均转为单声道，并在
`audio_formats` 中计数，不会静默丢弃。脚本按类别分别划分说话人。官方的 train/dev/test 来源记录在
manifest 的 `source_partition` 字段；研究切分仍由 `split` 字段固定。
报告保存各 split 的说话人集合哈希，便于发现意外重切分。所有音频路径只出现在被
`.gitignore` 排除的本地 manifest 中。

## 2. 提取训练与运行共享表征

```bash
python training/speech/extract_mdsc_features.py \
  --manifest training/manifests/mdsc.local.csv \
  --output training/processed/mdsc_features.npz
```

提取器直接导入 `app.speech_representation.extract_speech_representation`，避免训练和
树莓派推理的特征漂移。`--limit` 仅用于冒烟测试，正式训练不能设置。

## 3. 分组训练、校准与持出测试

```bash
python training/speech/train_mdsc.py \
  --features training/processed/mdsc_features.npz \
  --output models/mdsc_dysarthria_v1.json \
  --report training/reports/mdsc_dysarthria_v1.json \
  --folds 5 \
  --target-sensitivity 0.90 \
  --steps 500
```

正式报告必须同时查看：

- `oof_sample_metrics` 与 `oof_speaker_metrics`；
- `held_out_test_sample_metrics` 与 `held_out_test_speaker_metrics`；
- ROC-AUC、Brier score、敏感度、特异度、F1和平衡准确率；
- 以说话人为重采样单位的95% bootstrap区间；
- 测试说话人列表和 feature archive SHA-256。

不得根据持出测试结果调整阈值。若要调整特征或超参数，应产生新模型版本，并保留新的
最终测试说话人。

## 4. 运行时 shadow 接入

服务默认查找：

```text
models/mdsc_dysarthria_v1.json
```

也可显式指定：

```bash
python -m app.main \
  --speech-representation-model models/mdsc_dysarthria_v1.json
```

模型存在时，固定句结果增加：

- `metrics.shadow_dysarthria_probability`
- `metrics.shadow_dysarthria_threshold`
- `details.shadow_dysarthria_model`
- `details.shadow_dysarthria_prediction`
- `details.shadow_medical_role=dysarthria_representation_not_acute_stroke`

模型缺失、格式无效或推理失败时，现有 Whisper、质量门控和规则结果保持不变。

启动后可检查模型状态：

```bash
curl -s http://localhost:8080/api/speech/status | python3 -m json.tool
```

正常时 `representation_ready=true`、`representation_model=mdsc-dysarthria-v1`、
`representation_mode=shadow_only`。固定句完成后，概率和阈值会出现在 S 项报告的
原始数据区。当前模型只应用于固定句，不参与长期自然语音基线的异常投票。

## 5. M5 目标麦克风影子验证

在树莓派运行，不保存临时原始音频：

```bash
python scripts/run_speech_shadow.py \
  --model models/mdsc_dysarthria_v1.json \
  --device default \
  --windows 20 \
  --window-seconds 5 \
  --session-id participant-001-quiet \
  --label control \
  --environment quiet-20cm \
  --microphone usb-mic-v1
```

`participant-*` 必须是去标识化研究编号，不能写姓名、病历号或手机号。建议每名参与者
至少完成三类环境：

1. 安静、20 cm；
2. 安静、约1 m；
3. 家庭电视或风扇背景声。

汇总：

```bash
python training/speech/evaluate_shadow.py \
  --log data/speech/mdsc-shadow.jsonl \
  --output training/reports/mdsc_shadow.json
```

M5 工程完成条件：

- 树莓派连续30分钟没有崩溃或持续内存增长；
- 单窗口推理 p95 小于500 ms；
- 麦克风模式日志中 `raw_audio_retained=false`；
- 同一说话人同一环境的概率分布和阳性比例可复查；
- 至少覆盖安静、距离变化和家庭噪声；
- 模型仍为 shadow-only，阈值未根据影子参与者标签反复调节。

只有完成预先冻结、具有独立参与者的前瞻性方案，才能报告目标麦克风性能。项目自带的
合成冒烟数据只能证明代码可运行，不能作为模型准确率。

## 参考

- [AISHELL-6B 官方页](https://www.aishelltech.com/AISHELL_6B)
- [AISHELL Open Data 申请页](https://opendata.aishelltech.com/aishell-6b)
- [官方 LRDWWS 基线](https://github.com/greeeenmouth/LRDWWS)
- [MDSC 论文](https://arxiv.org/abs/2406.10304)
