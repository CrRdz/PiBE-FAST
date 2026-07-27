# PiBE-FAST

## 基于树莓派多模态感知的急性脑卒中早期筛查系统

## A Raspberry Pi–Based Multimodal System for Early Acute Stroke Screening

[中文说明](#中文说明) · [English](#english)

> **紧急提示 / Emergency notice**
>
> 如果突然出现面部歪斜、单侧无力、言语不清、视力或平衡异常，请立即拨打 120，
> 不要等待本系统完成检测。
>
> If facial drooping, one-sided weakness, speech difficulty, visual disturbance,
> or loss of balance occurs suddenly, call local emergency services immediately.
> Do not wait for this system to finish.

---

# 中文说明

## 1. 项目定位

PiBE-FAST 是面向树莓派边缘部署的 BE-FAST 急性脑卒中早期筛查研究原型。
系统融合以下信息：

- 摄像头视频流；
- MediaPipe 面部、虹膜和表情关键点；
- MoveNet 人体姿态关键点；
- 多帧时序运动特征；
- 本人或照护者提供的言语、平衡和发病时间信息。

树莓派负责模型推理、规则计算、日志和 Web 服务；默认使用与服务主机相连的
CSI/USB 摄像头，也可由打开页面的手机或电脑浏览器提供前置摄像头帧。
推理仍在运行 Python 服务的设备上完成，不上传到第三方云服务。

本项目不是医疗器械，不能诊断或排除脑卒中。当前阈值是没有可靠临床数据时的
工程初值，不是临床决策阈值，也不是卒中概率。

## 2. 树莓派的运行模式与作用

PiBE-FAST 不是要求使用者全天重复固定动作的连续检查系统。它采用“**长期待机、
事件触发、短时主动筛查**”的工作方式：

```text
低负载待机
  ├─ 用户感觉不适，点击“开始筛查”
  ├─ 照护者发现突然变化，启动筛查
  ├─ 被动层确认疑似跌倒，只触发筛查
  └─ 到达可选的定时筛查时间
             │
             ▼
1～2 分钟标准化 BE-FAST 主动筛查
             │
             ▼
结果提示 / 紧急提醒 → 用户确认后返回待机
```

固定动作只在主动筛查阶段出现。E 需要明确的左右视觉目标，F 需要中性脸与微笑基线，
A 需要双臂平举，B 需要在安全条件下站立。标准化动作让不同时间的测量具有可比性；
待机阶段不要求使用者面对摄像头或配合动作。

树莓派是**始终可用的本地边缘主机**，而不是电脑的摄像头配件。它负责：

- 连接 CSI/USB 摄像头，并持续提供本地预览；
- 待机时以默认约 2 FPS 低频运行 MoveNet，被动观察疑似跌倒；
- 根据当前步骤在 MoveNet 与 MediaPipe 之间切换，避免两套模型同时满负荷运行；
- 执行 BE-FAST 时序特征计算、质量门控和保守决策；
- 提供 Flask Web 页面，手机或电脑只是局域网内的显示与控制终端；
- 断网时继续本地筛查；按配置记录 JSONL 和紧急事件短片段；
- 后续可连接实体求助按钮、蜂鸣器、麦克风或可穿戴设备。

当前被动层实现了两类只用于触发后续确认的信号：

- 低频人体姿态中的“快速转变后持续躺倒”跌倒序列，以及相对个人基线的持续平衡变化；
- 本地麦克风自然语音窗口与个人声学基线的持续变化。

被动语音默认连续采集 8 秒窗口、间隔 1 秒，先用 6 个有效窗口建立个人基线。v2 根据
卒中构音障碍声学研究把特征分为时序、发声、构音/共振三域；同一窗口至少两个域明显
偏离个人基线才投异常票，最近 5 个有效窗口至少 3 票时，页面才建议执行固定句 S 确认。
它不运行全天 Whisper，不保存日常原始录音，也尚未验证说话人身份。被动输出
**不能作为构音障碍或脑卒中阳性/阴性结果**。论文到代码的逐项映射、工程假设及验证要求
见 [`docs/speech-evidence.md`](docs/speech-evidence.md)。

实际推理调度如下：

| 运行状态 | 摄像头预览 | MoveNet | MediaPipe Face | 医学含义 |
|---|---:|---:|---:|---|
| 待机 | 默认 15 FPS | 默认约 2 FPS | 暂停 | 姿态 + 自然语音变化触发信号 |
| 等待 E/F 动作 | 继续 | 暂停 | 低频约 5 FPS | 判断脸部/眼睛是否进入正确位置 |
| 等待 A/B 动作 | 继续 | 低频约 5 FPS | 暂停 | 判断肢体是否完整入镜及姿势是否安全 |
| E/F 阶段 | 继续 | 暂停 | 默认 5 FPS | 眼动或面部标准化测量 |
| A/B 阶段 | 继续 | 摄像头帧率 | 暂停 | 手臂或平衡标准化测量 |
| 项目选择 / B 输入 / 单项报告 | 继续 | 暂停 | 暂停 | 选择项目、填写 B、查看历史报告 |
| S 录音 / 分析 | 继续 | 暂停 | 暂停 | 短时 ALSA 采集与本地 whisper.cpp 推理 |

待机时 JSONL 只在实际执行低频推理时写入，不按每个摄像头帧写盘。平衡基线默认保存在
`data/balance-baseline.json`，自然语音基线保存在 `data/speech/passive-baseline.json`，
每个临时 WAV 在分析后立即删除。可以通过
`--standby-pose-fps` 调整负载，通过 `--disable-passive-monitor` 完全关闭被动推理，
通过 `--disable-passive-speech` 单独关闭长期语音监测，通过
`--scheduled-screen-interval-hours` 启用周期提醒。

> BE-FAST 识别的是已经突然出现的警示体征，不预测“即将中风”。如果已经出现任何
> 突发症状，应立即拨打 120，不要为了完成主动筛查而等待。

### Pi BE-FAST 筛查 Demo Web 界面

打开 Web 页面并进入筛查后，会先展示 B、E、F、A、S 五个独立项目。使用者可以任意
选择其中一项，不需要按固定顺序完成整套筛查。每个项目完成后立即显示本次单项报告，
并可直接“重新检测本项”或返回选择其他项目；同一轮中的每次重复结果都会按“第 N 次”
保留在报告历史中，不会被后一次覆盖。

E/F/A/B 自动项目的动作说明直接叠加在预览画面底部：红色表示人物位置、关键点可见性
或动作尚未满足采集条件，绿色表示当前动作与画面质量已经适合开始测量。系统只有在动作
质量连续通过后才自动进入该项采集；若采集期间质量不足，会立即生成“数据质量不足”的
单项报告，使用者可不限次数重试，不会把数据不足当作正常结果。每个自动检测项目也提供
“跳过本项”；确认后立即生成标记为“已跳过”的单项报告，不会视为阴性。

页面右上角可在两种输入之间选择：

- **自动切换的主机摄像头**：当前 MacBook 配置在 E/F 使用索引 `1` 的电脑前置
  摄像头，在待机及 A/B 使用索引 `0` 的手机连续互通相机；树莓派 Picamera2
  后端仍使用同一台 CSI 摄像头完成所有步骤。
- **手机 / 当前设备**：调用当前打开页面的设备前置摄像头。要用手机镜头，
  必须在手机上打开该页面并选择此项。浏览器将画面缩放到最长边 640 px，
  以约 6 FPS 上传 JPEG 帧，服务端将其送入同一套 MediaPipe / MoveNet 流程。

“主机自动切换”和“手机 / 当前设备”两种输入可在待机页或项目选择页手动切换；正式采集动作期间
会锁定输入，避免中途更换视角。主机模式内部的
E/F 与 A/B 摄像头路由会在步骤交界处自动完成，同一项测量期间不会改变视角。除 `localhost` 外，
手机和电脑浏览器都要求可信任的 HTTPS 上下文才会允许页面调用摄像头。浏览器麦克风仍被禁用；
S 直接使用连接到树莓派的 ALSA 麦克风，因此不依赖手机授权、网络质量或人工勾选。
预览帧内的大型调试文字默认关闭，需排障时可用 `--debug-overlay` 显式开启。

E、F、A 分别引导眼动、微笑和双臂平举。B 会先询问是否已经出现失衡：若已有异常，
直接生成报告而不要求冒险站立；若未报告异常，再要求使用者确认周围有支撑且站立安全。
S（言语）会采集约 7 秒固定提示句，在树莓派上完成音频质量、本地转写、复述差异、语速和
停顿分析；T（是否突然发生、发病时间）随 B/S 记录。页面
右上角的 `中文 / EN` 可即时切换界面语言。这里的绿色仅代表**动作及采集质量合格**，
不代表医学筛查结果为阴性。每份单项报告会显示异常类型、可能涉及的侧别和数据不足原因，
例如“左眼水平移动范围明显小于右眼”，而不是只显示一个“异常”标签。

## 3. 系统架构

```text
CSI / USB 摄像头 ────┐
浏览器前置摄像头 ─────┴─▶ 选定的视频输入 ─┐
USB / I²S 麦克风 ────────────────────────┴─▶
Raspberry Pi
  ├─ Picamera2 / OpenCV：采集视频
  ├─ ALSA arecord：采集 S 的 16 kHz 单声道音频
  ├─ whisper.cpp：本地离线转写
  ├─ MediaPipe Face Landmarker：E、F
  ├─ MoveNet Lightning：A、B
  ├─ BE-FAST 时序特征与保守决策
  ├─ JSONL 研究日志 / 可选事件片段
  └─ Flask Web 服务
        │ 局域网 / SSH 端口转发
        ▼
手机或电脑浏览器：预览、引导、操作和结果
```

为控制树莓派 CPU 占用和温度，系统采用阶段调度：

- 待机阶段仅以默认约 2 FPS 运行 MoveNet，Face Landmarker 暂停；
- E/F 阶段仅以默认 5 FPS 运行 Face Landmarker，暂停 MoveNet；
- A/B 阶段运行 MoveNet，暂停人脸推理；
- 等待 E/F 动作时低频运行 Face Landmarker，等待 A/B 动作时低频运行 MoveNet；
- 项目选择、B 人工观察、S 录音和单项报告阶段暂停两套视觉推理模型；
- S 仅在用户点击“开始录音”后短时运行音频处理和 whisper.cpp，不持续监听；
- 浏览器预览仍按摄像头帧率更新；
- 任一模型不可用时保留原始预览，对应项目返回数据不足，不伪造正常结果。

## 4. BE-FAST 检测技术细节

### B — Balance / 平衡

**目标：**在持续监控中发现相对个人基线的躯干方向或横向摆动变化，并允许
本人/照护者直接报告突然失衡。

**输入：**MoveNet 的左右肩、左右髋和左右踝关键点，以及站立姿态质量门槛。

**计算：**

1. 计算肩中心、髋中心、躯干轴和双踝中点。
2. 用肩宽归一化躯干中心相对双踝中点的横向位置；该量只是摄像头运动学代理，
   不是力台压力中心（COP）或真实全身质心。
3. 计算躯干轴相对画面竖直方向的角度：

   ```text
   trunk_roll = atan2(shoulder_center_x - hip_center_x,
                      hip_center_y - shoulder_center_y)
   ```

4. 预热 1.5 秒后采集 30 秒有效安静站立；默认积累 5 个合格窗口形成个人基线。
5. 窗口级计算躯干方向中位数、横向摆动 RMS、第 5–95 百分位范围、路径长度及
   平均速度。
6. 后续窗口用 median/MAD（MAD 退化时同时参考 IQR）与个人基线比较。

**判定规则：**

- 至少 30 个有效样本且有效帧比例不低于 75%；
- 校准完成前返回 `insufficient`，不伪造自动正常结论；
- 移除了 `0.40 × 肩宽` 和 `0.50 × 肩宽` 固定工程阈值；
- 躯干方向变化或横向摆动平均速度增加达到 modified Z-score `3.5` 时，仅触发
  后续主动筛查；`3.5` 是稳健统计过程界值，不是卒中临床 cutoff；
- 若个人基线离散度为零，变化分数不可估计时返回 `insufficient`，不加入任意噪声下限；
- 全身、脚踝或稳定站姿持续不可见：`insufficient`；
- 本人/照护者报告突然失衡时，无需冒险站立，B 可直接记为阳性。

参数与论文的逐项映射见
[持续平衡监测：证据、实现与限制](docs/balance-evidence.md)。

**限制：**单目姿态不能测量 COP、眩晕、共济失调或深度方向摆动；地面、镜头角度、
辅助器具、骨科疾病和既往残疾都会影响结果。“与个人基线一致”不能排除卒中。

### E — Eyes / 眼睛

**摄像头来源：**可使用服务主机的本地摄像头，也可使用当前浏览器设备的
前置摄像头。两者都会进入相同的 E/F 特征识别流程。

**目标：**检测对移动视觉目标的可见眼动响应、左右眼活动范围差和双眼共轭运动异常。

**输入：**MediaPipe Face Landmarker 的 478 点面部网格，其中包括：

- 右虹膜中心 `468`，左虹膜中心 `473`；
- 右眼角 `33/133`，左眼角 `362/263`；
- 鼻尖 `1`；
- 双眼外眼角 `33/263` 作为尺度和头部滚转参考。

**引导与计算：**

1. 页面目标点依次停留在中间、左侧、右侧，每处约 3 秒，界面同步显示方向和倒计时。
2. 使用双眼外眼角连线校正画面内头部滚转。
3. 用外眼角距离归一化脸部尺度；小于画面宽度约 7.5% 时认为脸太小。
4. 每只眼的虹膜位置转换为眼裂内相对位置：

   ```text
   gaze_x = (iris_x - eye_corner_min_x) / eye_width
   ```

5. 分别计算左右眼从左目标到右目标的活动范围。
6. 比较两眼活动范围差、相对中心目标的共轭误差，以及鼻尖相对双眼中心的头部代偿。

**当前工程规则：**

- 每个目标至少 4 个有效样本，总有效帧比例不低于 50%；
- 两眼活动范围都低于 `0.12`：可见目标跟随减弱；
- 左右眼活动范围差达到 `0.10`：眼球活动不对称；
- 最大共轭误差达到 `0.14`：双眼共轭运动异常；
- 头部代偿范围达到 `0.35 × 双眼外眼角距离`：数据不足并提示重试；
- 虹膜、面部持续不可见：`insufficient`。

**关键医学边界：**E 只检测摄像头可见的眼动，不能测量视力，也不能排除视物模糊、
复视、黑蒙或视野缺损。本人报告突然视力异常时，必须直接按急症处理，不能被模型
阴性结果覆盖。

### F — Face / 面部

**目标：**检测从中性表情到微笑时的单侧下脸部运动减弱。

**输入：**

- 右嘴角 `61`、左嘴角 `291`；
- 双眼外眼角 `33/263`；
- MediaPipe blendshape：`mouthSmileLeft`、`mouthSmileRight`。

**引导与计算：**

1. 先采集约 2 秒中性表情作为个体基线。
2. 页面提示后采集约 3 秒自然微笑。
3. 使用双眼连线做滚转校正，并用双眼外眼角距离归一化嘴角高度差。
4. 计算微笑阶段相对中性阶段的嘴角差变化。
5. 同时比较左右 `mouthSmile` 激活，减少仅依赖单个几何点造成的误报。

**当前工程规则：**

- 中性和微笑阶段各至少 5 个有效样本，总有效帧比例不低于 50%；
- 最大微笑激活低于 `0.22`：未检测到足够微笑，返回数据不足；
- 基线校正后的嘴角差变化达到 `0.075 × 双眼距离`：阳性；
- 左右微笑激活差达到 `0.24`：阳性；
- 面部太小、遮挡或阶段样本不足：`insufficient`。

**限制：**天然面部不对称、既往面瘫、牙科/颌面疾病、光照、胡须、口罩和大幅转头
都可能影响结果。

### A — Arms / 手臂

**目标：**检测双臂平举时的持续高度差和保持过程中的单侧下沉。

**输入：**MoveNet 的左右肩、左右肘和左右腕关键点。

**计算：**

1. 受试者坐稳并将双臂向前平举。
2. 预热 1.5 秒后采集约 6 秒。
3. 每侧手腕相对同侧肩膀的纵向距离按肩宽归一化：

   ```text
   wrist_relative_y = (wrist_y - shoulder_y) / shoulder_width
   ```

4. 使用整个阶段的中位数计算持续双臂高度差。
5. 比较采样前 1/3 与后 1/3 的手腕位置，计算左右下沉量之差。

**当前工程规则：**

- 至少 20 个有效样本且有效帧比例不低于 55%；
- 持续高度差达到 `0.30 × 肩宽`：阳性；
- 左右下沉量之差达到 `0.22 × 肩宽`：阳性；
- 开始时双臂没有抬起，或肩/腕持续不可见：`insufficient`。

**限制：**肩周疾病、疼痛、旧有偏瘫、活动受限、宽松衣物和透视角度都会影响结果。

### S — Speech / 言语

Speech 现在分成两层。第一层是在待机页持续运行的自然语音变化监测：它不依赖固定文本，
也不执行 ASR，而是与本机保存的个人声学基线比较。只有多个窗口持续变化时才提示进入
第二层固定句确认。

长期层的 v2 特征按论文中的言语子系统分组：

1. 时序：停顿比例、平均停顿时长和基于能量峰的音节核速率代理；
2. 发声：强度范围、F0 中位数与变化、局部 jitter、局部 shimmer 和 HNR；
3. 构音/共振：自由语音 LPC 帧中的 F1/F2 分布四分位距。

这些是适合树莓派本地运行的轻量近似量，不等同于论文里的强制对齐构音速率、标准元音
VSA/VAI/FCR、Praat、YAGA 或 openSMILE/eGeMAPS。单一域变化不会产生异常票；
多域、多窗口规则用于降低误触发，但仍是未经临床验证的工程规则。详见
[`长期语音监测证据说明`](docs/speech-evidence.md)。

固定句 S 不依赖人工勾选。用户点击“开始录音”后，服务端通过 `arecord` 从树莓派连接的
ALSA 麦克风采集约 7 秒、16 kHz、16-bit、单声道 WAV，并用本地 `whisper.cpp`
转写固定提示句“今天天气很好，我们一起去公园散步”。

当前工程特征包括：

1. 录音时长、RMS 音量、削波比例和有效说话时长，用于拒绝太短、太小声、失真或无人声的样本；
2. 识别文本与提示句的字符错误率（CER），默认达到 `0.35` 时提示复述内容差异；
3. 有效说话区间内的停顿比例，默认达到 `0.55` 时提示表达不连续；
4. 每秒识别字符数，默认低于 `1.0` 或高于 `8.0` 时提示语速异常。

麦克风、识别程序或模型不可用时，本项返回 `insufficient`，绝不会把缺少音频或识别能力
当作阴性。阳性 S 报告会把 WAV、识别文本和量化指标保存在异常历史中；阴性和数据不足
录音在报告生成后删除。

这些特征能检查“是否正确、连续地复述固定句子”，但还不是经过临床验证的构音障碍模型。
方言、环境噪声、听力问题、原有言语障碍以及 ASR 本身的错误都可能影响结果，不能把
ASR 差异直接解释为脑卒中。

### T — Time / 时间

记录异常是否为新出现或突然发生，以及首次发现异常的时间。决策逻辑为：

- 任一 B/E/F/A/S 项阳性，且确认为新发/突然发生：`emergency`；
- 存在阳性，但尚未确认新发：`warning`；
- 任一项目质量不足或未完成：`incomplete`；
- 所有项目均为阴性：`clear`，但仍不能排除脑卒中。

## 5. 引导流程

1. 树莓派平时处于低负载待机，不需要持续完成固定动作。
2. 用户、照护者、被动异常或定时提醒触发一次筛查。
3. 确认实时预览、光线和取景。
4. E：头保持不动，只用眼睛跟随中/左/右目标。
5. F：先保持中性表情，提示后自然微笑。
6. A：坐稳并保持双臂向前平举。
7. B：只有在有人看护且安全时站立；不安全就跳过并报告异常。
8. S/T：点击录音并复述提示句，填写是否突然出现及首次发现时间；本地分析完成后自动出报告。
9. 查看结果后点击“结束并返回待机”。

## 6. 安装与模型

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

桌面开发固定使用 `mediapipe==0.10.31`。Linux ARM64/Raspberry Pi 目前没有对应的
官方 PyPI wheel；`requirements.txt` 会在该平台跳过 MediaPipe，需要按 Google 的
[Python wheel 构建说明](https://developers.google.com/edge/mediapipe/solutions/build_python)
构建并安装相同版本。

安装 MoveNet 所需的解释器：

```bash
# 桌面开发
python -m pip install tensorflow

# Raspberry Pi：使用与系统/Python 匹配的轻量解释器
python -m pip install tflite-runtime
```

树莓派 S 项使用 ALSA 录音工具和本地 `whisper.cpp`。先确认麦克风设备：

```bash
./scripts/install_speech_pi.sh

# 或手动安装/检查：
sudo apt update
sudo apt install -y alsa-utils cmake build-essential git
arecord -L
arecord -D default -f S16_LE -r 16000 -c 1 -d 3 /tmp/pibefast-mic-test.wav
```

构建 `whisper.cpp` 并下载多语言 `base` 模型：

```bash
git clone --depth 1 https://github.com/ggml-org/whisper.cpp.git ~/whisper.cpp
cmake -S ~/whisper.cpp -B ~/whisper.cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build ~/whisper.cpp/build --config Release -j2
~/whisper.cpp/models/download-ggml-model.sh base
cp ~/whisper.cpp/models/ggml-base.bin models/ggml-base.bin
```

项目通过命令行调用 `whisper-cli`，不引入 TensorFlow/PyTorch 语音运行时。如果模型尚未
安装、命令不可执行或麦克风采集失败，S 会明确返回“数据质量不足”。

macOS 不提供 ALSA。项目会自动改用 FFmpeg 的 AVFoundation 后端采集系统默认麦克风，
Linux/树莓派仍使用 `arecord`：

```bash
brew install ffmpeg whisper-cpp
ffmpeg -f avfoundation -list_devices true -i ""
```

macOS 上默认 `--speech-device default` 使用 AVFoundation 音频设备索引 `0`
（通常是内置麦克风）；也可以传入 AVFoundation 列出的音频设备名称或索引。Linux/
树莓派上的 `default` 仍表示 ALSA 默认设备。

模型文件：

```text
models/movenet_lightning.tflite
models/face_landmarker.task
models/ggml-base.bin
```

下载官方 Face Landmarker 模型：

```bash
curl -L \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task \
  -o models/face_landmarker.task
```

## 7. 运行

MacBook 双摄像头自动切换（推荐）：

```bash
.venv/bin/python -m app.main
```

本项目针对当前 MacBook + 连续互通相机配置提供阶段感知切换：待机、A（手臂）和
B（平衡）使用索引 `0` 的手机摄像头，以便拍摄上半身或全身；E（眼动）和 F（面部）
使用索引 `1` 的 MacBook 前置摄像头，以便稳定捕捉面部细节。进入新阶段时程序会先
释放旧摄像头再打开目标摄像头，因此启动时不再需要传入 `--camera-index`。如果索引
`1` 不可用，系统会自动回退到索引 `0`，不会中断筛查。

macOS 的设备索引可能因连接顺序而变化。如需覆盖自动配置，可使用
`--camera-index N` 指定待机/A/B 摄像头，并使用 `--face-camera-index M` 指定 E/F
摄像头。在 Web 页面选择“自动切换（E/F 电脑 · A/B 手机）”即可使用主机端自动路由；
“手机 / 当前设备”仍表示浏览器通过 `getUserMedia` 上传画面。

树莓派官方摄像头：

```bash
python -m app.main \
  --source camera \
  --camera-backend picamera2 \
  --speech-device default \
  --whisper-cli ~/whisper.cpp/build/bin/whisper-cli \
  --speech-model models/ggml-base.bin \
  --web-host 0.0.0.0 \
  --web-port 8080
```

树莓派长期待机示例：

```bash
python -m app.main \
  --source camera \
  --camera-backend picamera2 \
  --standby-pose-fps 2 \
  --balance-baseline-path data/balance-baseline.json \
  --passive-speech-window-seconds 8 \
  --passive-speech-baseline-windows 6 \
  --scheduled-screen-interval-hours 0
```

- `--standby-pose-fps 2`：待机 MoveNet 频率，越低越省 CPU，但快速事件采样更稀疏；
- `--balance-baseline-path`：同一人的 5 个合格平衡窗口持久化位置；
- `--disable-passive-monitor`：只保留手动或照护者触发；
- `--passive-speech-window-seconds 8`：每个自然语音分析窗口的长度；
- `--passive-speech-interval-seconds 1`：窗口之间的间隔；
- `--passive-speech-baseline-windows 6`：个人基线需要的有效窗口数；
- `--disable-passive-speech`：关闭长期自然语音监测，但保留固定句 S；
- `--scheduled-screen-interval-hours 12`：每 12 小时打开一次主动筛查提醒；默认 `0` 关闭。
- `--speech-device plughw:CARD,DEV`：覆盖 ALSA 录音设备；用 `arecord -L` 查询；
- `--speech-capture-seconds 7`：S 的单次固定录音时长；
- `--disable-speech`：硬件尚未接入时关闭 S 后端；页面调用会返回不可用。

只测试本地长期语音监测、不启动摄像头和姿态模型：

```bash
.venv/bin/python -m scripts.run_passive_speech_local --reset-baseline
```

启动后由同一人在较安静环境中自然说话，直到 `assessment` 变为 `stable`。之后可改变
说话节律或播放另一种频谱的语音观察多窗口投票；按 `Ctrl-C` 停止。这个脚本不会保留
日常 WAV。macOS 首次运行时需要允许终端/FFmpeg 使用麦克风。

完整 Web 服务的 `http://localhost:8080` 中，点击 S 后可选择“长期语音监测”或
“朗读固定句”。长期监测页实时显示个人基线进度、当前响度、有效语音时长、三类
声学域偏离分数以及最近 5 个有效窗口的异常投票。长期监听只负责提示变化；进入固定句
检测时长期监听会暂时暂停，完成后恢复。

然后在同一局域网的手机或电脑访问：

```text
http://<树莓派IP>:8080
```

上述 HTTP 地址可使用主机摄像头，但移动浏览器通常会拒绝在 HTTP 局域网页面中打开
手机摄像头。要使用“手机 / 当前设备”，请配置手机信任的 TLS 证书，并运行：

```bash
python -m app.main \
  --camera-backend picamera2 \
  --web-host 0.0.0.0 \
  --web-port 8443 \
  --web-cert cert.pem \
  --web-key key.pem
```

手机访问 `https://<树莓派IP或证书域名>:8443`。证书必须包含所访问的主机名/IP 且被手机信任；
仅忽略不可信证书警告不能保证 `getUserMedia` 可用。MacBook 本机使用
`http://localhost:8080` 时，localhost 被浏览器视为安全上下文，可直接选择“当前设备”。

SSH 端口转发：

```bash
ssh -L 8080:localhost:8080 pi@raspberrypi.local
```

## 8. API、日志与隐私

```text
GET  /api/status
GET  /api/history?component=E,F&reason=asymmetric_eye_excursion
GET  /api/history/<记录ID>
GET  /api/history/<记录ID>/frame
GET  /api/history/<记录ID>/audio
GET  /api/speech/status
GET  /api/speech/passive/status
POST /api/camera/source      {"source":"host" | "client"}
POST /api/camera/frame       Content-Type: image/jpeg
POST /api/monitoring/trigger {"source":"user","reason":"felt_unwell"}
POST /api/monitoring/standby
POST /api/befast/stage   {"stage":"eyes"}
POST /api/befast/stage   {"stage":"face"}
POST /api/befast/stage   {"stage":"arms"}
POST /api/befast/stage   {"stage":"balance"}
POST /api/befast/skip
POST /api/befast/manual
POST /api/befast/reset
POST /api/speech/start    {"language":"zh","new_or_sudden":false}
POST /api/speech/complete
POST /api/speech/cancel
POST /api/speech/passive/pause
POST /api/speech/passive/resume
POST /api/speech/passive/reset-baseline
```

`/api/status` 中的 `befast.mode` 为 `standby` 或 `screening`；`monitoring` 字段显示
当前推理调度、待机 FPS、最近被动状态和
`medical_role=trigger_only_not_stroke_diagnosis`。`passive_speech` 字段显示基线进度、
最近窗口、多窗口异常票数以及是否建议执行固定句确认。

历史接口只保存状态为 `positive` 的单项报告。元数据持久化在
`data/history/history.sqlite3`，对应 JPEG 帧保存在 `data/history/frames/`；S 的
异常 WAV 保存在 `data/history/audio/`。
`GET /api/history` 默认按最新时间返回，可使用以下查询参数：

- `component=B,E,F,A,S`：单个项目或逗号分隔的多个项目；
- `reason`：精确筛选异常原因；
- `affected_side=left|right`：筛选可能受影响侧；
- `new_or_sudden=true|false`：筛选是否新发/突发；
- `limit=1..200` 和 `offset`：分页。

可以用 `--history-dir <目录>` 修改保存位置。自动 E/F/A/B/S 阳性和人工 B 阳性使用
同一历史存储；重复轮询同一报告不会产生重复记录。

B 人工输入示例：

```json
{
  "balance_problem": false,
  "new_or_sudden": false,
  "onset_time": "2026-07-19T10:30"
}
```

默认 JSONL 日志只保存姿态关键点、质量、结果、原因和量化指标，不持续保存原始视频。
只有显式启用 `--save-event-clips` 才会保存事件片段。S 阳性会有意保存单次异常 WAV，
阴性或数据不足 WAV 会被删除。真实患者视频、人脸、语音和身份
信息不得提交到 GitHub；采集前应取得知情同意并制定访问、加密、留存和删除策略。

## 9. 测试与性能基准

```bash
python -m unittest discover -s tests -v
python -m scripts.benchmark_face --frames 100
```

测试使用合成关键点，只验证软件和规则逻辑，不代表临床敏感度或特异度。取得合规临床
数据后，应采用患者级数据划分、卒中类似疾病阴性组、神经科医师结合 CT/MRI 的最终诊断，
并报告敏感度、特异度、PPV、NPV、校准度和外部验证结果。

## 10. 项目结构

```text
app/
  main.py                 # 分阶段实时推理与 Web 服务入口
  befast/                 # 按职责拆分的 BE-FAST 筛查包
    balance.py            # B：平衡检测与人工结果合并
    eyes.py               # E：眼球运动检测
    face.py               # F：面部微笑对称性检测
    arms.py               # A：手臂下垂检测
    speech.py             # S：麦克风分析结果转为统一报告
    urgency.py            # T：发病时间与紧急程度判定
    session.py            # 筛查阶段编排与线程安全状态
    guidance.py           # 实时取景和动作引导
    config.py             # BE-FAST 阈值配置
    result.py             # 统一检测结果模型
  face_landmarker.py      # MediaPipe 478 点/52 blendshape 适配器
  monitoring.py           # 低频被动触发层与树莓派待机节流
  passive_speech.py       # 长期自然语音基线与多窗口变化触发
  movenet.py              # TFLite MoveNet 推理
  camera.py               # OpenCV / Picamera2 视频输入
  pose_classifier.py      # standing / sitting / lying 质量门槛
  history.py              # 异常报告 SQLite 索引与 JPEG 帧存储
  speech_audio.py         # ALSA 录音、whisper.cpp 适配和 S 音频特征
  web.py                  # 双语项目名、中文操作引导与 API
  drawing.py              # 姿态/面部点和状态叠加
  keypoint_logger.py      # JSONL 研究日志
  event_recorder.py       # 可选紧急事件片段
scripts/
  benchmark_face.py       # 不保存画面的 Face Landmarker 基准
  run_passive_speech_local.py # 不启动摄像头的本地长期语音测试
tests/
models/
data/
```

---

# English

## 1. Project scope

PiBE-FAST is a research prototype for Raspberry Pi edge deployment and early
BE-FAST screening of acute stroke. It fuses:

- a live camera stream;
- MediaPipe facial, iris, and expression landmarks;
- MoveNet body-pose landmarks;
- temporal motion features across multiple frames; and
- structured observations about speech, balance, and symptom onset.

The Raspberry Pi performs model inference, feature computation, logging, and Web
serving. It uses an attached CSI/USB camera by default, or it can receive front-
camera frames from the phone or computer that opened the page. Inference still
runs on the Python-service host and no video is sent to a third-party cloud.

This project is not a medical device and cannot diagnose or rule out stroke.
All current thresholds are unvalidated engineering defaults, not clinical
decision thresholds or stroke probabilities.

## 2. Raspberry Pi operating model and responsibilities

PiBE-FAST does not ask a person to repeat fixed actions all day. It uses an
**always-available standby, event-triggered, short active-screening** model:

```text
Low-load standby
  ├─ the user feels unwell and presses Start
  ├─ a caregiver notices a sudden change
  ├─ the passive layer confirms a possible fall and requests a screen
  └─ an optional scheduled reminder becomes due
                         │
                         ▼
1–2 minute standardized guided BE-FAST screen
                         │
                         ▼
Result / urgent warning → return to standby after acknowledgement
```

The fixed actions appear only during an active screen. E needs explicit visual
targets, F needs neutral and smile phases, A needs a bilateral arm hold, and B
needs supervised standing when safe. Standardized actions make measurements at
different times comparable; standby does not require the person to face the
camera or perform any action.

The Raspberry Pi is the **always-on local edge host**, not a camera accessory for
a desktop computer. It:

- connects to a CSI/USB camera and serves a local live preview;
- runs MoveNet at about 2 FPS by default in standby to observe possible falls;
- switches between MoveNet and MediaPipe by stage instead of saturating the CPU
  with both models;
- computes temporal features, quality gates, and conservative BE-FAST decisions;
- hosts the Flask interface while a phone or computer acts only as a LAN client;
- continues local screening without Internet access and optionally records JSONL
  data or short emergency event clips; and
- captures fixed-duration S audio from an attached ALSA microphone and runs local
  offline transcription; a physical help button, buzzer, or wearable can be added later.

The passive layer now has two trigger-only signals:

- low-rate pose observation with a fall sequence requiring a rapid transition
  followed by sustained lying; and
- long-running local natural-speech windows compared with a personal acoustic
  baseline.

Passive speech captures 8-second windows with a 1-second gap by default and
builds a baseline from 6 valid windows. v2 groups paper-informed features into
timing, phonation, and articulation/resonance domains. A window votes as changed
only when at least two domains deviate from the personal baseline; the guided S
check is recommended after at least 3 changed votes among the latest 5 valid
windows. It does not run Whisper continuously, retain routine raw audio, or
currently verify the speaker. Passive output is **never a positive or negative
dysarthria or stroke result**. See
[`docs/speech-evidence.md`](docs/speech-evidence.md) for the paper-to-code map,
engineering assumptions, and required clinical validation.

Actual inference scheduling:

| State | Camera preview | MoveNet | MediaPipe Face | Medical role |
|---|---:|---:|---:|---|
| Standby | 15 FPS default | about 2 FPS default | paused | pose + natural-speech change triggers |
| Waiting for E/F setup | continues | paused | about 5 FPS | verify face/eye framing |
| Waiting for A/B setup | continues | about 5 FPS | paused | verify body framing and safe posture |
| E/F | continues | paused | 5 FPS default | standardized gaze/face measurement |
| A/B | continues | camera cadence | paused | standardized arm/balance measurement |
| Check picker / B input / individual report | continues | paused | paused | choose a check, collect B, show report history |
| S recording / analysis | continues | paused | paused | short ALSA capture and local whisper.cpp inference |

In standby, JSONL is written only when low-rate inference actually runs, not for
every camera frame. The balance baseline defaults to `data/balance-baseline.json`,
and the natural-speech baseline is stored in `data/speech/passive-baseline.json`;
each temporary WAV is deleted immediately after analysis. Tune load with
`--standby-pose-fps`, disable pose triggers with
`--disable-passive-monitor`, disable long-running speech with
`--disable-passive-speech`, and enable periodic prompts with
`--scheduled-screen-interval-hours`.

> BE-FAST recognizes warning signs that have already appeared suddenly; it does
> not predict an impending stroke. If any sudden sign is already present, call
> emergency services immediately instead of waiting for the active screen.

### Pi BE-FAST Screening Demo Web interface

After entering screening, the page presents five independent B, E, F, A, and S
checks. The user may choose any check without completing a fixed sequence. Each
finished check immediately displays its own report and offers **Repeat this
check** or **Choose another check**. Every repeated result is retained as report
number N in the current session instead of being overwritten.

For automated E/F/A/B checks, the current instruction appears over the bottom of
the live video. Red means framing, landmark visibility, or the requested action is
not yet suitable for capture. Green means the action and capture quality are ready.
A check starts automatically only after readiness is observed consistently. If
capture quality is insufficient, an individual insufficient-data report appears
immediately and the check can be retried without limit. Each automated check also
provides **Skip this check**; after confirmation, an individual report marks the
attempt as `skipped`, never as negative.

The top-right selector provides two inputs:

- **Automatically routed host cameras:** on the current MacBook setup, E/F uses
  the built-in front camera at index `1`, while standby and A/B use the phone
  Continuity Camera at index `0`. The Raspberry Pi Picamera2 backend continues
  to use its single attached CSI camera for every stage.
- **Phone / this device:** opens the front camera of the device displaying the
  page. To use a phone camera, open the page on that phone and select this option.
  The browser scales frames to a 640 px maximum edge and uploads JPEG at about
  6 FPS; the service feeds them into the same MediaPipe / MoveNet pipeline.

The user can switch between host routing and **Phone / this device** from either
standby or the component menu. The input is locked during active capture to avoid
changing viewpoint mid-measurement. Host E/F-to-A/B routing happens automatically
between stages, never during one temporal measurement. Browsers require a trusted HTTPS context for camera
access except on `localhost`; browser microphone access remains disabled because S uses the
Pi-attached ALSA microphone directly. The large
diagnostic text drawn into the video is off by default and can be restored with
`--debug-overlay` when troubleshooting.

E, F, and A independently guide eye motion, smile, and bilateral arm hold. B first
asks whether balance loss is already present: a reported problem produces a report
without risking a standing task; otherwise supported standing still requires an
explicit safety confirmation. S records a fixed prompt through the Raspberry Pi
microphone and locally assesses audio quality, transcript difference, rate, and
pauses. T (sudden onset and first-known time) is recorded with B and S.
The top-right `中文 / EN` control switches the whole interface immediately. Green
indicates **action and capture readiness only**; it is not a negative medical
screening result. Each individual report identifies the specific sign, affected
side when available, and the reason for insufficient data—for example, “the left
eye had a smaller horizontal range” instead of showing only “abnormal.”

## 3. Architecture

```text
CSI / USB camera ──────┐
browser front camera ───┴─▶ selected video input ─┐
USB / I²S microphone ─────────────────────────────┴─▶
Raspberry Pi
  ├─ Picamera2 / OpenCV video capture
  ├─ ALSA arecord speech capture
  ├─ local whisper.cpp transcription
  ├─ MediaPipe Face Landmarker for E and F
  ├─ MoveNet Lightning for A and B
  ├─ temporal BE-FAST features and conservative decisions
  ├─ JSONL research logs / optional event clips
  └─ Flask Web service
        │ LAN or SSH tunnel
        ▼
Phone or computer browser: preview, guidance, controls, and results
```

To limit Raspberry Pi CPU load and thermal pressure, inference is stage-aware:

- standby runs MoveNet at about 2 FPS by default and pauses Face Landmarker;
- E/F runs Face Landmarker at 5 FPS by default and pauses MoveNet;
- A/B runs MoveNet and pauses facial inference;
- Face Landmarker runs at low rate while waiting for E/F setup, and MoveNet runs
  at low rate while waiting for A/B setup;
- both models pause during component selection, B input, S audio processing, and result review;
- the browser preview continues at camera cadence; and
- if either model is unavailable, raw preview remains available and the affected
  item becomes insufficient instead of being reported as normal.

## 4. BE-FAST detection details

### B — Balance

**Purpose:** detect sustained changes in trunk orientation or mediolateral sway
relative to a personal baseline during long-running monitoring, while allowing the
person or caregiver to directly report sudden loss of balance.

**Inputs:** MoveNet left/right shoulders, hips, and ankles, gated by standing-pose
quality.

**Method:**

1. Compute shoulder and hip centers, the trunk axis, and the ankle midpoint.
2. Normalize the trunk-center position relative to the ankle midpoint by shoulder
   width. This is a camera kinematic proxy, not force-plate COP or true whole-body
   center of mass.
3. Compute trunk-axis orientation relative to image vertical:

   ```text
   trunk_roll = atan2(shoulder_center_x - hip_center_x,
                      hip_center_y - shoulder_center_y)
   ```

4. After a 1.5-second warm-up, collect 30 seconds of valid quiet standing. Five
   qualified windows form the default personal baseline.
5. Summarize median trunk orientation, mediolateral RMS, 5th–95th percentile
   range, path length, and mean velocity.
6. Compare later windows with the personal median/MAD profile, using IQR when MAD
   degenerates.

**Decision rules:** at least 30 valid samples and 75% valid frames. Calibration
returns `insufficient` rather than a false normal result. The former
`0.40 × shoulder width` and `0.50 × shoulder width` engineering cutoffs have been
removed. A trunk-orientation change or increased mediolateral mean velocity at
modified Z-score `3.5` triggers active follow-up only; `3.5` is a robust process
monitoring boundary, not a clinical stroke cutoff. If personal-baseline dispersion
is zero, an unscorable change returns `insufficient` instead of introducing an
arbitrary noise floor. Missing ankles/body or an unstable pose produces
`insufficient`. A reported sudden balance problem can mark B positive without
requiring an unsafe standing attempt.

See [Long-running balance monitoring: evidence, implementation, and
limitations](docs/balance-evidence.md) for the paper-to-code mapping.

**Limitations:** monocular pose cannot measure vertigo, ataxia, or depth-axis sway.
Camera angle, walking aids, orthopedic disease, and pre-existing disability can
affect the result. Agreement with the personal baseline cannot rule out stroke.

### E — Eyes

**Camera source:** either the local camera attached to the service host or the
front camera of the current browser device. Both enter the same E/F feature
extraction path.

**Purpose:** detect visible gaze response to moving targets, inter-eye excursion
asymmetry, and abnormal conjugate eye movement.

**Inputs:** MediaPipe's 478-point mesh, including right/left iris centers `468/473`,
right eye corners `33/133`, left eye corners `362/263`, nose tip `1`, and outer
eye corners `33/263` for scale and roll correction.

**Method:**

1. A target remains at center, left, and right for about 3 seconds each; the UI
   shows the current direction and a countdown.
2. The outer-eye line corrects in-plane head roll.
3. Interocular distance normalizes scale; a distance below about 7.5% of image
   width is treated as a face that is too small.
4. Iris position is represented within each eye opening:

   ```text
   gaze_x = (iris_x - eye_corner_min_x) / eye_width
   ```

5. Compute left-to-right excursion for both eyes.
6. Compare excursion asymmetry, conjugacy error relative to center, and head
   compensation derived from the nose position.

**Current engineering rules:** at least 4 valid samples per target and 50% valid
frames. Both excursions below `0.12` indicate reduced visible target response;
excursion difference of `0.10` indicates asymmetry; conjugacy error of `0.14`
indicates abnormal conjugate motion; head compensation of `0.35 × interocular
distance` makes the result insufficient. Missing irises/face also produces
`insufficient`.

**Critical medical boundary:** E only evaluates camera-visible eye motion. It does
not measure visual acuity and cannot rule out blurred vision, diplopia, transient
vision loss, or visual-field defects. A sudden subjective visual symptom must
override a negative model result and be treated as an emergency symptom.

### F — Face

**Purpose:** detect unilateral lower-face movement weakness from neutral expression
to smile.

**Inputs:** right/left mouth corners `61/291`, outer eye corners `33/263`, and the
`mouthSmileLeft` / `mouthSmileRight` MediaPipe blendshapes.

**Method:** collect a 2-second neutral baseline followed by a 3-second natural
smile. Correct roll using the eye line, normalize mouth-corner displacement by
interocular distance, compare smile-versus-neutral corner asymmetry, and combine
it with the left/right smile activation difference.

**Current engineering rules:** at least 5 valid samples in each phase and 50% valid
frames. Maximum smile activation below `0.22` means no adequate smile was detected.
Baseline-corrected mouth-corner change of `0.075 × interocular distance`, or a
left/right smile activation difference of `0.24`, is positive. A small, occluded,
or insufficiently sampled face produces `insufficient`.

**Limitations:** natural asymmetry, previous facial palsy, dental/maxillofacial
conditions, lighting, facial hair, masks, and head rotation can affect the result.

### A — Arms

**Purpose:** detect persistent arm-height asymmetry or unilateral downward drift
during a forward arm hold.

**Inputs:** MoveNet left/right shoulders, elbows, and wrists.

**Method:** after a 1.5-second warm-up, collect approximately 6 seconds. Normalize
each wrist's vertical position relative to its shoulder by shoulder width:

```text
wrist_relative_y = (wrist_y - shoulder_y) / shoulder_width
```

Use the stage median for persistent height difference. Compare the first and
last thirds of the samples to measure differential arm drift.

**Current engineering rules:** at least 20 valid samples and 55% valid frames;
`0.30 × shoulder width` persistent height difference or `0.22 × shoulder width`
differential drift is positive. Arms not initially raised or shoulders/wrists not
visible long enough produces `insufficient`.

**Limitations:** shoulder disease, pain, prior hemiparesis, restricted movement,
loose clothing, and perspective distortion can affect the result.

### S — Speech

Speech now has two layers. The standby page continuously compares natural speech
with a local personal acoustic baseline without fixed text or ASR. Only a
sustained multi-window change recommends the second-layer guided phrase check.

The v2 long-running layer groups its features by speech subsystem:

1. timing: pause fraction, mean pause duration, and an energy-peak syllable-nuclei
   rate proxy;
2. phonation: intensity range, F0 median/variation, local jitter, local shimmer,
   and HNR;
3. articulation/resonance: F1/F2 distribution IQR from connected-speech LPC frames.

These lightweight Raspberry Pi estimators are not interchangeable with
forced-aligned articulation rate, standardized-vowel VSA/VAI/FCR, Praat, YAGA,
or openSMILE/eGeMAPS. A single changed domain does not cast an anomaly vote. The
multi-domain and multi-window rules reduce nuisance triggers but remain
unvalidated engineering rules. See the
[`speech evidence note`](docs/speech-evidence.md).

The guided S check does not use a manual abnormality checkbox. After **Start recording**, the
server captures about 7 seconds of 16 kHz, 16-bit mono WAV through `arecord` and
the Pi-attached ALSA microphone, then transcribes a fixed prompt locally with
`whisper.cpp`.

Current engineering features are recording duration, RMS level, clipping
fraction, voiced duration, fixed-prompt character error rate (CER), pause
fraction, and recognized characters per voiced second. Defaults flag CER at
`0.35`, pause fraction at `0.55`, and rate outside `1.0..8.0` characters/second.
An unavailable microphone, CLI, or model returns `insufficient`, never a false
negative. Positive S attempts retain the WAV, transcript, and metrics in abnormal
history; negative and insufficient temporary recordings are deleted.

This checks whether a fixed sentence was repeated accurately and continuously;
it is not a clinically validated dysarthria model. Dialects, noise, hearing
problems, pre-existing speech disorders, and ASR errors can affect the result and
must not be interpreted directly as stroke.

### T — Time

The interface records whether a sign is new/sudden and when it was first noticed:

- any positive B/E/F/A/S item plus new/sudden onset: `emergency`;
- a positive item without confirmed sudden onset: `warning`;
- an unfinished or low-quality item: `incomplete`;
- all items negative: `clear`, which still does not rule out stroke.

## 5. Guided workflow

1. The Raspberry Pi remains in low-load standby; no repeated actions are required.
2. A user, caregiver, passive anomaly, or scheduled reminder triggers one screen.
3. Confirm live preview, lighting, and framing.
4. E: keep the head still and follow center/left/right targets using only the eyes.
5. F: remain neutral, then smile when prompted.
6. A: sit safely and hold both arms forward.
7. B: stand only with supervision and only when safe; otherwise skip and report it.
8. S/T: start recording, repeat the prompt, and record sudden onset and first-known time;
   the individual report appears after local processing.
9. Review the result and select **Return to standby**.

## 6. Installation and models

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Desktop development is pinned to `mediapipe==0.10.31`. There is currently no
official Linux ARM64/Raspberry Pi PyPI wheel. The requirements file skips
MediaPipe on that platform; build and install the same version using Google's
[Python wheel build guide](https://developers.google.com/edge/mediapipe/solutions/build_python).

Install a MoveNet-compatible TFLite interpreter:

```bash
# Desktop development
python -m pip install tensorflow

# Raspberry Pi: select a wheel matching the OS and Python version
python -m pip install tflite-runtime
```

S also needs ALSA capture tools and a local `whisper.cpp` executable:

```bash
./scripts/install_speech_pi.sh

# Or install and inspect the dependencies manually:
sudo apt update
sudo apt install -y alsa-utils cmake build-essential git
arecord -L
arecord -D default -f S16_LE -r 16000 -c 1 -d 3 /tmp/pibefast-mic-test.wav

git clone --depth 1 https://github.com/ggml-org/whisper.cpp.git ~/whisper.cpp
cmake -S ~/whisper.cpp -B ~/whisper.cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build ~/whisper.cpp/build --config Release -j2
~/whisper.cpp/models/download-ggml-model.sh base
cp ~/whisper.cpp/models/ggml-base.bin models/ggml-base.bin
```

The project invokes `whisper-cli` as a process, avoiding a PyTorch speech
runtime. A missing capture device, executable, or model yields an insufficient
S result.

macOS does not provide ALSA. The project automatically selects FFmpeg's
AVFoundation capture backend on macOS while keeping `arecord` on Linux and
Raspberry Pi:

```bash
brew install ffmpeg whisper-cpp
ffmpeg -f avfoundation -list_devices true -i ""
```

On macOS, `--speech-device default` selects AVFoundation audio device index `0`
(normally the built-in microphone); an AVFoundation audio device name or index
can be passed instead. On Linux/Raspberry Pi, `default` retains its ALSA meaning.

Required models:

```text
models/movenet_lightning.tflite
models/face_landmarker.task
models/ggml-base.bin
```

Download the official Face Landmarker model:

```bash
curl -L \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task \
  -o models/face_landmarker.task
```

## 7. Running the system

Automatic dual-camera routing on a MacBook (recommended):

```bash
.venv/bin/python -m app.main
```

For the current MacBook + Continuity Camera setup, the service uses stage-aware
routing. Camera index `0` (the phone) is used for standby, A (arms), and B
(balance), where a wider upper-body or full-body view is needed. Camera index `1`
(the MacBook front camera) is used for E (eyes) and F (face), where close facial
detail is needed. The old device is released before the target device is opened,
so no camera argument is required at startup. If index `1` is unavailable, the
screen falls back to index `0` instead of stopping.

macOS camera indices can change with device connection order. Override the
automatic mapping with `--camera-index N` for standby/A/B and
`--face-camera-index M` for E/F. In the Web UI, **Auto (E/F Mac · A/B phone)**
uses this host-side routing; **Phone / this device** still means browser-uploaded
frames from `getUserMedia`.

Raspberry Pi camera:

```bash
python -m app.main \
  --source camera \
  --camera-backend picamera2 \
  --speech-device default \
  --whisper-cli ~/whisper.cpp/build/bin/whisper-cli \
  --speech-model models/ggml-base.bin \
  --web-host 0.0.0.0 \
  --web-port 8080
```

Long-running Raspberry Pi standby example:

```bash
python -m app.main \
  --source camera \
  --camera-backend picamera2 \
  --standby-pose-fps 2 \
  --balance-baseline-path data/balance-baseline.json \
  --passive-speech-window-seconds 8 \
  --passive-speech-baseline-windows 6 \
  --scheduled-screen-interval-hours 0
```

- `--standby-pose-fps 2`: lower values save CPU but sample rapid events less often;
- `--balance-baseline-path`: persistent location for one person's five qualified balance windows;
- `--disable-passive-monitor`: retain only user/caregiver-triggered screening;
- `--passive-speech-window-seconds 8`: duration of each natural-speech window;
- `--passive-speech-interval-seconds 1`: gap between windows;
- `--passive-speech-baseline-windows 6`: valid windows needed for the baseline;
- `--disable-passive-speech`: disable long-running speech while retaining guided S;
- `--scheduled-screen-interval-hours 12`: open a screen every 12 hours; `0` disables it.
- `--speech-device plughw:CARD,DEV`: override the ALSA capture device listed by `arecord -L`;
- `--speech-capture-seconds 7`: fixed S recording duration;
- `--disable-speech`: disable the S backend until microphone hardware is installed.

Test only long-running local speech, without starting camera or pose models:

```bash
.venv/bin/python -m scripts.run_passive_speech_local --reset-baseline
```

Have the same speaker talk naturally in a quiet setting until `assessment` becomes
`stable`. The status output exposes per-feature and per-domain scores for local
engineering tests. Do not treat a failure to trigger as a medical negative or try
to imitate stroke symptoms. Press `Ctrl-C` to stop. Routine WAV files are not
retained. On first use, macOS must allow the terminal/FFmpeg to access the
microphone.

At `http://localhost:8080`, select S and then choose either long-running
monitoring or fixed-phrase reading. The monitoring view visualizes baseline
progress, input level, voiced duration, three acoustic-domain deviation scores,
and the five most recent valid-window votes. Passive monitoring remains a change
trigger; the fixed-phrase reading check temporarily pauses passive capture while
it runs.

Open `http://<raspberry-pi-ip>:8080` from a phone or computer on the same network,
or use an SSH tunnel. This HTTP URL supports the host camera, but mobile browsers
normally reject camera capture from an HTTP LAN page.

To use **Phone / this device**, configure a TLS certificate trusted by the phone:

```bash
python -m app.main \
  --camera-backend picamera2 \
  --web-host 0.0.0.0 \
  --web-port 8443 \
  --web-cert cert.pem \
  --web-key key.pem
```

Open `https://<raspberry-pi-ip-or-certificate-hostname>:8443`. The certificate
must cover that hostname/IP and be trusted by the phone; merely bypassing an
untrusted-certificate warning may not enable `getUserMedia`. On the MacBook that
runs the service, `http://localhost:8080` is treated as a secure context and can
use **this device** without TLS.

SSH tunnel:

```bash
ssh -L 8080:localhost:8080 pi@raspberrypi.local
```

## 8. API, logging, and privacy

```text
GET  /api/status
GET  /api/history?component=E,F&reason=asymmetric_eye_excursion
GET  /api/history/<record-id>
GET  /api/history/<record-id>/frame
GET  /api/history/<record-id>/audio
GET  /api/speech/status
GET  /api/speech/passive/status
POST /api/camera/source      {"source":"host" | "client"}
POST /api/camera/frame       Content-Type: image/jpeg
POST /api/monitoring/trigger {"source":"user","reason":"felt_unwell"}
POST /api/monitoring/standby
POST /api/befast/stage   {"stage":"eyes"}
POST /api/befast/stage   {"stage":"face"}
POST /api/befast/stage   {"stage":"arms"}
POST /api/befast/stage   {"stage":"balance"}
POST /api/befast/skip
POST /api/befast/manual
POST /api/befast/reset
POST /api/speech/start    {"language":"en","new_or_sudden":false}
POST /api/speech/complete
POST /api/speech/cancel
POST /api/speech/passive/pause
POST /api/speech/passive/resume
POST /api/speech/passive/reset-baseline
```

In `/api/status`, `befast.mode` is `standby` or `screening`. The `monitoring`
object reports inference scheduling, standby FPS, the last passive state, and
`medical_role=trigger_only_not_stroke_diagnosis`. `passive_speech` reports
baseline progress, the latest window, recent anomaly votes, and whether the
guided phrase check is recommended.

The history API persists positive single-check reports only. Metadata is stored
in `data/history/history.sqlite3`, with JPEG frames in `data/history/frames/`
and abnormal S WAV files in `data/history/audio/`.
`GET /api/history` accepts a single or comma-separated `component`, exact
`reason`, `affected_side=left|right`, `new_or_sudden=true|false`, `limit=1..200`,
and `offset`. Use `--history-dir` to change the storage location. Polling the
same report does not create duplicate records.

B manual-input example:

```json
{
  "balance_problem": false,
  "new_or_sudden": false,
  "onset_time": "2026-07-19T10:30"
}
```

By default, JSONL logs contain landmarks, quality, decisions, reasons, and numeric
metrics, but not continuous raw video. Event clips are saved only when
`--save-event-clips` is explicitly enabled. A positive S attempt intentionally
retains its one-shot WAV; negative and insufficient WAV files are deleted. Do not commit real patient video,
faces, speech, or identifiers to GitHub. Obtain informed consent and define
access, encryption, retention, and deletion policies before data collection.

## 9. Tests and benchmarks

```bash
python -m unittest discover -s tests -v
python -m scripts.benchmark_face --frames 100
```

Tests use synthetic landmarks and validate software logic only. They do not
establish clinical sensitivity or specificity. Clinical evaluation should use
patient-level splits, stroke-mimic controls, neurologist-adjudicated CT/MRI-based
outcomes, calibration analysis, and external validation.

## 10. Repository layout

```text
app/
  main.py                 # stage-aware inference and Web entry point
  befast/                 # BE-FAST package split by responsibility
    balance.py            # B: balance check and manual-result merge
    eyes.py               # E: eye-movement check
    face.py               # F: smile-symmetry check
    arms.py               # A: arm-drift check
    speech.py             # S: convert microphone analysis to a report item
    urgency.py            # T: onset and urgency decision
    session.py            # thread-safe screening orchestration
    guidance.py           # live placement and action guidance
    config.py             # BE-FAST thresholds
    result.py             # shared result model
  face_landmarker.py      # MediaPipe 478-landmark/52-blendshape adapter
  monitoring.py           # throttled passive trigger layer for Pi standby
  passive_speech.py       # natural-speech baseline and multi-window change trigger
  movenet.py              # TFLite MoveNet inference
  camera.py               # OpenCV / Picamera2 input
  pose_classifier.py      # standing / sitting / lying quality gate
  history.py              # abnormal-report SQLite index and JPEG frame storage
  speech_audio.py         # ALSA capture, whisper.cpp adapter, and S audio features
  web.py                  # guided interface and API
  drawing.py              # pose/face/status overlays
  keypoint_logger.py      # JSONL research logs
  event_recorder.py       # optional emergency event clips
scripts/
  run_passive_speech_local.py # local long-running speech test without camera models
  benchmark_face.py       # no-save Face Landmarker benchmark
tests/
models/
data/
```

## References / 参考资料

- [持续平衡监测：证据、实现与限制](docs/balance-evidence.md)
- [长期语音监测：证据、实现与限制](docs/speech-evidence.md)
- [Balance and speech BibTeX metadata](docs/references.bib)
- [Aroor et al. 2017 — BE-FAST and strokes missed by FAST](https://doi.org/10.1161/STROKEAHA.116.015169)
- [Dai et al. 2022 — lateropulsion prevalence after stroke](https://doi.org/10.1212/WNL.0000000000200010)
- [Aryan et al. 2023 — standing-balance force-plate reliability after stroke](https://doi.org/10.1016/j.heliyon.2023.e21046)
- [Ruhe et al. 2010 — COP test–retest reliability review](https://doi.org/10.1016/j.gaitpost.2010.09.012)
- [Yeung et al. 2014 — Kinect body-sway assessment](https://doi.org/10.1016/j.gaitpost.2014.06.012)
- [De Cock et al. 2021 — acute ischemic stroke dysarthria](https://doi.org/10.1111/1460-6984.12607)
- [Mou et al. 2018 — Mandarin post-stroke vowel acoustics](https://doi.org/10.1038/s41598-018-32429-8)
- [Sanguedolce et al. 2025 — acoustic and glottal stroke-speech features](https://doi.org/10.21437/Interspeech.2025-2313)
- [Jyothi et al. 2026 — F0 and duration features in stroke speech](https://doi.org/10.1038/s41598-026-40155-9)
- [American Stroke Association — Stroke symptoms and BE-FAST](https://www.stroke.org/en/about-stroke/stroke-symptoms)
- [MediaPipe Face Landmarker for Python](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python)
- [MediaPipe Raspberry Pi Face Landmarker example](https://github.com/google-ai-edge/mediapipe-samples/tree/main/examples/face_landmarker/raspberry_pi)
- [MoveNet models on TensorFlow Hub](https://www.tensorflow.org/hub/tutorials/movenet)
- [MediaPipe Python wheel build guide](https://developers.google.com/edge/mediapipe/solutions/build_python)
- [whisper.cpp local inference and model setup](https://github.com/ggml-org/whisper.cpp)
- [Raspberry Pi audio documentation](https://www.raspberrypi.com/documentation/accessories/audio.html)
