# 长期语音监测：证据、实现与限制

更新日期：2026-08-17

实现版本：`stroke-speech-evidence-v2`

## 结论先行

当前长期语音监测是一个**论文启发的个体内变化触发器**，不是经过临床验证的构音障碍
检测器，更不是脑卒中诊断器。论文支持“卒中后言语可能在时序、发声和构音/共振等多个
子系统发生变化”；论文**没有**直接验证本项目的 8 秒窗口、6 窗口基线、稳健 z 分数
3.5、两个异常域或 3/5 投票规则。

因此，代码只在多个声学域持续偏离个人基线时建议执行保留的固定句 S 检查。无论长期
监测或固定句结果如何，只要言语困难突然出现，都不应等待本系统，应立即呼叫急救。

固定句和合格的长期语音窗口会加载已用官方 AISHELL-6B/MDSC 数据训练的 90 维
log-Mel 模型。概率达到冻结阈值 `0.735282` 时，长期窗口直接建议固定句确认；固定句
则直接把 S 判为阳性。模型学习的是 MDSC 慢性构音障碍与对照说话人的区分，不是
急性卒中标签，因此不能单独诊断卒中。其训练数据、内部持出结果和尚未完成的 M5
目标麦克风验证见 [`mdsc-speech-training.md`](mdsc-speech-training.md)。

## 证据到代码的映射

| 证据域 | 原始研究观察 | v2 本地特征 | 不能声称的内容 |
|---|---|---|---|
| 时序 / 韵律 | 构音速率、无声间隔、持续时间和 F0 等特征与神经疾病或构音障碍严重度有关；2026 年卒中语料研究也报告了 F0 与过渡区持续时间差异 [2, 7] | `pause_fraction`、`mean_pause_duration_seconds`、`syllable_nuclei_rate_hz` | 能量峰只是音节核速率代理，不等于人工标注或强制对齐得到的构音速率 |
| 发声 | 急性缺血性卒中构音障碍常见粗糙音质、可闻吸气，最大持续发声时间和最大响度偏离明显；卒中研究也使用 F0、传统声学和声门特征 [1, 5, 6, 7] | `intensity_range_db`、`f0_median_hz`、`f0_std_semitones`、`jitter_local`、`shimmer_local`、`hnr_db` | 当前是轻量自相关估计，不等同于临床仪器、Praat、YAGA 或 openSMILE/eGeMAPS |
| 构音 / 共振 | 普通话卒中后构音障碍研究发现元音空间缩小、F1/F2 偏离或分布改变；连接语音研究也支持 F2 范围、VSA、VAI、FCR [3, 4] | `formant_f1_iqr_hz`、`formant_f2_iqr_hz` | 自由语音没有元音标签，当前 LPC 分布 IQR 不是 VSA/VAI/FCR，只是探索性个人基线代理 |

研究本身的适用边界也必须保留：

| 文献 | 人群与任务 | 对本项目的直接性 |
|---|---|---|
| De Cock 2021 [1] | 151 名首次急性缺血性卒中患者，67 名由言语语言治疗师确认构音障碍；入院 72 小时内标准化评估 | 直接支持急性卒中构音障碍的临床表现，但不是家庭连续监听研究 |
| Kim 2011 [2] | 107 名构音障碍说话者，病因包括卒中、帕金森病、TBI、MSA | 支持多子系统声学特征，不支持卒中特异阈值 |
| Mou 2018 [3] | 31 名普通话卒中后构音障碍患者与 38 名健康对照；标准化单音节元音 | 直接支持 F1/F2 与元音空间变化，不直接支持无文本自由语音 IQR |
| Ge 2021 [4] | 25 名普通话卒中后痉挛型构音障碍患者与 25 名对照；连接语音中的元音标注 | 比单元音更接近自然言语，但仍使用元音标签，且只覆盖痉挛型 |
| Wong 2022 [5] | 9 名慢性卒中后构音障碍患者；持续元音、朗读和快速音节重复的治疗试验 | 支持 jitter/shimmer/HNR 等发声量的可测性，不能证明筛查性能 |
| Sanguedolce 2025 [6] | 专门卒中语料的图片描述，组合传统声学与声门参数做分类 | 最接近连接语音自动分析；同库分类结果不能替代家庭外部验证 |
| Jyothi 2026 [7] | 50 名卒中患者与 50 名健康对照；朗读及 5 个持续元音 | 支持 F0 与时长特征，仍非开放麦克风纵向监测 |

代码映射：

- 特征提取与三域判定：`app/passive_speech.py`
- 本地无摄像头测试入口：`scripts/run_passive_speech_local.py`
- 固定句录音、质量门控与离线转写：`app/speech_audio.py`
- Web 状态与长期监测控制：`app/web.py`

## v2 触发逻辑

1. 每个窗口先执行时长、响度、有效语音时长和削波质量门控。
2. 默认收集 6 个有效自然语音窗口作为个人基线；一项特征至少在其中 3 个窗口可测，
   才进入基线比较；v1 基线不会被 v2 复用。
3. 对每项可用特征计算相对个人基线中位数的稳健偏离：

   ```text
   score = abs(current - baseline_median) / max(1.4826 × MAD, scale_floor)
   ```

4. 某个证据域内至少一项 `score >= 3.5`，该域才记为改变。
5. 同一窗口至少两个证据域改变，才投一张异常票。单一域变化不会投票。
6. 最近 5 个有效窗口至少 3 票，页面显示“建议进行固定句确认”；MDSC 达阈值时则
   无需等待多窗口投票，立即建议确认，且该窗口不写入个人基线。
7. 日常临时 WAV 在分析后删除；长期层不运行 Whisper。个人基线只保存数值特征。

API 状态显式返回：

- `evidence_version: "stroke-speech-evidence-v2"`
- `clinical_validation: false`
- `medical_role: "change_detection_trigger_only"`
- `changed_domains`、`domain_scores` 和 `trigger_policy`

## 哪些仍是工程假设

以下参数来自边缘设备资源、误触发控制和可测试性取舍，并非论文给出的临床阈值：

- 8 秒窗口与 1 秒间隔；
- 6 个基线窗口与最多 48 个历史基线窗口；
- 各特征的最小稳健尺度；
- `3.5` 的偏离阈值；
- “至少两个域”的窗口规则；
- “最近 5 个窗口至少 3 票”的持续变化规则；
- 最低响度、有效语音时长和削波门槛；
- 自由语音 LPC 共振峰分布的使用方式；
- 固定句的提示文本、CER、停顿和语速工程阈值。

这些值必须在目标人群数据上冻结预注册方案后再验证，不能在论文引用后被描述为“有临床
依据的卒中阈值”。

## 为什么不直接复刻论文分类器

纳入研究的任务和人群并不相同：有标准元音、持续发声、朗读、图片描述，也有急性、
亚急性或既往卒中患者。部分研究还混合了卒中、帕金森病、创伤性脑损伤等病因。最新
分类结果通常来自受控录音和同一数据库，尚不能直接外推到家庭中持续开放麦克风、不同
说话人、电视噪声和树莓派麦克风。

本项目因此采用：

- 个体内基线，而不是把研究中的组间均值当患者阈值；
- 多域一致性，而不是单一声学值；
- 多窗口持续性，而不是一次录音；
- 固定句作二次标准化确认，而不是自动给出卒中结论。

这仍只是降低误触发的工程设计，不等于临床有效性证据。

## 临床验证前必须完成

1. 前瞻性收集目标麦克风、真实家庭噪声和树莓派上的数据。
2. 纳入急性卒中、非卒中急症、既往构音障碍、健康对照及卒中模拟疾病。
3. 以言语语言治疗师盲法评估、神经科诊断和发病时间为独立参考标准。
4. 按患者而非录音切分训练/验证/测试集，并进行外部中心验证。
5. 预先规定主要终点，报告灵敏度、特异度、PPV、NPV、置信区间与校准。
6. 单独测试不同性别、年龄、方言、麦克风距离、背景说话人、电视和呼吸道疾病。
7. 验证长期个人基线的漂移、缺失语音、换人说话和隐私/知情同意处理。

## 主要文献

1. De Cock E, et al. *Dysarthria following acute ischemic stroke:
   Prospective evaluation of characteristics, type and severity.* International
   Journal of Language & Communication Disorders. 2021;56(3):549–557.
   [DOI 10.1111/1460-6984.12607](https://doi.org/10.1111/1460-6984.12607)
2. Kim Y, Kent RD, Weismer G. *An Acoustic Study of the Relationships Among
   Neurologic Disease, Dysarthria Type, and Severity of Dysarthria.* Journal of
   Speech, Language, and Hearing Research. 2011;54(2):417–429.
   [DOI 10.1044/1092-4388(2010/10-0020)](https://doi.org/10.1044/1092-4388%282010/10-0020%29)
3. Mou Z, et al. *Acoustic properties of vowel production in
   Mandarin-speaking patients with post-stroke dysarthria.* Scientific Reports.
   2018;8:14188.
   [DOI 10.1038/s41598-018-32429-8](https://doi.org/10.1038/s41598-018-32429-8)
4. Ge S, et al. *Quantitative acoustic metrics of vowel production in
   Mandarin-speakers with post-stroke spastic dysarthria.* Clinical Linguistics
   & Phonetics. 2021;35(8):779–792.
   [DOI 10.1080/02699206.2020.1827295](https://doi.org/10.1080/02699206.2020.1827295)
5. Wong MN, et al. *Transcranial direct current stimulation over the primary
   motor cortex improves speech production in post-stroke dysarthric speakers:
   A randomized pilot study.* PLOS ONE. 2022;17(10):e0275779.
   [DOI 10.1371/journal.pone.0275779](https://doi.org/10.1371/journal.pone.0275779)
6. Sanguedolce G, et al. *Physiologically-Informed Feature Analysis of Acquired
   Speech Disorders for Stroke Assessment.* Interspeech 2025:813–817.
   [DOI 10.21437/Interspeech.2025-2313](https://doi.org/10.21437/Interspeech.2025-2313)
7. Jyothi MVS, et al. *Characterizing stroke-affected speech using F0 and
   duration-based features.* Scientific Reports. 2026;16:9146.
   [DOI 10.1038/s41598-026-40155-9](https://doi.org/10.1038/s41598-026-40155-9)

可复用的 BibTeX 元数据见 [`references.bib`](references.bib)。仓库不复制论文 PDF；
以上链接指向 DOI、期刊或开放全文页面，以避免版权和版本问题。
