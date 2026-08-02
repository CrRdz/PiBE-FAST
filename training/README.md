# PiBE-FAST 离线训练

当前管线实现 A（Arm weakness）模型的完整流程：

```text
带标签的关键点 JSONL
  -> 每次试验的共享时序特征
  -> 按 subject_id 分组交叉验证
  -> 类别平衡逻辑回归
  -> 在折外预测上选择敏感度约束阈值
  -> models/arm_screen_v2.json
  -> 树莓派运行端自动加载；缺失或无效时回退规则
```

这是一套研究训练基础设施，不会把内部交叉验证结果描述为临床性能。

## 1. 环境

训练只依赖项目已有的 NumPy：

```bash
python3 -m venv .venv-train
source .venv-train/bin/activate
python -m pip install -r training/requirements.txt
```

训练在电脑上进行；树莓派只加载导出的 JSON 模型。

## 2. 输入 JSONL

每行至少包含时间和 MoveNet 17 点：

```json
{"ts":0.1,"keypoints":[{"name":"left_shoulder","x":0.4,"y":0.3,"score":0.95}]}
```

项目的 `data/keypoints/*.jsonl` 已符合该结构。若一个文件包含整场会话，清单中的
`start_ts` 和 `end_ts` 必须只包围正式的 A 采集窗口，不能包含准备抬臂阶段。

外部 Kinect/光学动捕数据应先映射到 MoveNet 关节名称，并投影/归一化到当前的
`x/y/score` 格式。外部数据的补偿动作标签不能冒充 NIHSS 手臂下垂标签；需要在
`source` 中保留来源，最终以匹配本项目协议的数据验证。

## 3. 标签清单

复制示例后填写真实信息：

```bash
cp training/manifests/arms.example.csv training/manifests/arms.local.csv
```

字段：

- `subject_id`：受试者标识；同一人的所有试验必须相同。
- `trial_id`：本次双臂平举的唯一标识。
- `label`：0 正常，1 异常。
- `affected_side`：`left`、`right` 或 `none`。
- `source`：`local`、`toronto_pose`、`clinical` 等。
- `path`：JSONL 路径；相对路径以清单所在目录为基准。
- `start_ts/end_ts`：可选的文件内时间范围。

不要将真实姓名、病历号、生日等直接标识写入仓库。

## 4. 提取共享特征

```bash
python training/prepare_arms.py \
  --manifest training/manifests/arms.local.csv \
  --output training/processed/arms.csv
```

低于 20 个有效帧的试验会被拒绝。训练端与运行端共同调用
`app.arm_model.summarize_arm_samples`，避免训练—推理特征漂移。

## 5. 训练与导出

```bash
python training/train_arms.py \
  --features training/processed/arms.csv \
  --output models/arm_screen_v2.json \
  --report training/reports/arm_screen_v2.json \
  --folds 5 \
  --target-sensitivity 0.90
```

训练脚本会：

1. 按 `subject_id` 分折，绝不按帧或单次试验随机泄漏；
2. 对每折只用训练受试者估计标准化参数和模型；
3. 汇总折外概率后选择达到目标敏感度且特异度最高的阈值；
4. 用全部开发数据拟合最终系数；
5. 导出均值、尺度、系数、截距、阈值和内部验证摘要。

每个类别至少需要与折数相同数量的独立受试者。正式实验应另外保留从未参与
交叉验证的外部测试受试者；当前脚本报告的是内部折外验证，不是临床验证。

## 6. 运行端行为

默认模型位置为 `models/arm_screen_v2.json`。V2 保留原有 12 个手腕高度/漂移特征，
并加入 63 个肩、肘、腕、髋的人体坐标时序特征，包括归一化位置、肘角、躯干比例、
中位数、波动和首尾变化。训练端和运行端共用 `app.arm_model` 与
`app.compensation_model`，避免特征定义漂移。质量门控仍由现有 A 流程执行：

- 双肩和双腕可见；
- 有效帧数量和比例达标；
- 双臂确实抬起。

质量合格且模型存在时使用学习概率；模型缺失、格式无效或特征异常时回退到原有
工程规则。可在 `BefastConfig` 中设置 `arm_model_enabled=False` 强制禁用模型。

## 7. 数据最低要求

软件冒烟测试可以使用合成数据，但合成动作不能报告为模型性能。第一轮真实训练建议：

- 每类至少 20 名独立受试者，最好更多；
- 每人至少 3 次完整协议试验；
- 健康老年人、既往偏瘫和不同严重度分别覆盖；
- 阳性样本由临床人员给出 NIHSS Motor Arm 和受累侧；
- 相机、距离、光照变化不能只集中在某个标签组。

F（Face）将在获得 Toronto NeuroFace 的可用视频及许可后复用相同的分组、校准和
导出框架；当前版本不生成伪造的 F 模型。

## 8. 已下载的公开运动学数据与研究模型

仓库本地已下载三套数据（均位于被 `.gitignore` 排除的 `data/training/`）：

- Toronto Rehab Stroke Pose：10 名健康参与者、9 名卒中幸存者、Kinect 25 点三维
  骨骼与逐帧代偿标签；只适合训练康复动作中的前倾、耸肩、躯干旋转等代偿识别。
- Lucchetti–Bailo–Lencioni：10 名健康参与者、10 名卒中参与者、6 项功能任务、
  19 个运动学角度与 12 通道 EMG，许可为 CC BY 4.0，DOI
  `10.6084/m9.figshare.c.7720187.v1`。
- IntelliRehabDS 2.0.1：29 名参与者、Kinect 25 点骨架、9 种康复动作及专家
  正确/错误标签，CC BY 4.0，DOI `10.5281/zenodo.4610859`。本项目只下载约
  199 MB 的 `SkeletonData.zip`，没有下载约 41 GB 的深度图。

第二套数据已经接入可复现的研究训练流程：

```bash
python training/prepare_kinematic_stroke.py
python training/train_kinematic_stroke.py
```

预处理以官方 `Events.Start/End` 切分每次重复，提取每个角度通道的 5–95% 幅度、
标准差和 90% 分位角速度，并加入任务编号。缺失超过一半的角度通道会导致该次重复
被剔除。训练目标是“功能任务中的卒中患侧 vs 健康优势侧”，阈值在每位受试者的
折外平均概率上选择，输出：

- `models/research/kinematic_stroke_v1.json`
- `training/reports/kinematic_stroke_v1.json`

这个模型不会被应用运行时加载。它的光学动捕协议、健康/卒中年龄分布和目标标签
都不等于 BE-FAST 双臂平举筛查，因此只能作为数据管线验证和后续迁移学习参考。
真正替代 A 工程阈值仍需要使用本项目相机、完整双臂平举协议和临床标注采集数据，
再运行第 2–6 节的 `prepare_arms.py` / `train_arms.py` 流程。

## 9. IntelliRehabDS 动作质量影子模型

当前训练只选择与 Web 协议最接近的左/右肩外展（gesture 4/5），把 Kinect 三维点
投影到与 MoveNet 共有的肩、肘、腕、髋二维人体坐标，生成每侧的幅度、肘伸展、
轨迹和躯干稳定性特征：

```bash
python training/prepare_intellirehab_action_quality.py
python training/train_action_quality.py
```

当前得到 534 次试验、29 名受试者。五折验证严格按受试者分组；“错误/不完整动作”
的折外敏感度为 0.682、特异度为 0.786、平衡准确率为 0.734。模型输出到
`models/research/intellirehab_action_quality_v1.json`。由于训练是 Kinect 单侧动作，
运行是普通摄像头双侧动作，该模型只显示 `shadow_action_invalid_*`，不参与完成门控。

## 10. Toronto 二维代偿模型（影子模式）

在无法采集临床患者的当前阶段，使用 Toronto 的专家逐帧代偿标签训练：

```bash
python training/prepare_toronto_compensation.py
python training/train_toronto_compensation.py
```

这条管线只提取 Kinect 与 MoveNet 共有的肩、肘、腕、髋二维特征，排除 Kinect
深度，按 30 帧窗口生成三个一对其余模型。运行端会自动从 `models/research/` 加载，
并记录双臂完整动作期间的前倾、耸肩和躯干旋转概率。这些 `shadow_*` 字段不会改变
双臂高度/下落结果，也不会直接形成用户告警。

完整的数据分布、折外结果和限制见
[`docs/toronto-compensation-model.md`](../docs/toronto-compensation-model.md)。
