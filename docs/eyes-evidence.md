# E 眼动辅助检查：证据、实现与限制

实现版本：`stroke-eye-phenotype-v3`

## 定位

E 首先询问本人或照护者是否观察到突然视力下降、黑蒙、视野缺损、复视或持续凝视
偏向。只要报告了这类新发症状，就直接形成 E 阳性提示，不要求摄像头再次“证明”。
这符合 BE-FAST 中 Eyes 主要用于补充视觉症状的原始定位 [1]。

摄像头部分只是**低帧率水平注视终点辅助检查**。它不测量视力、视野、眼底，也不能
测量可靠的扫视潜伏期或峰值速度；摄像头阴性不能排除卒中。MediaPipe Iris 本身也不
直接输出视线方向 [6]。

## 论文到代码的映射

| 文献结论 | 当前实现 | 不能推导出的结论 |
|---|---|---|
| BE-FAST 加入视觉和步态症状可减少 FAST 漏检 [1] | 视觉症状表单优先于摄像头检查；报告异常时直接结束 E | 摄像头眼动阴性可以排除视觉型卒中 |
| 急性单侧幕上卒中可出现对侧扫视幅度/速度降低，以及方向性眼动减少 [2] | 以双眼共同向左或向右的终点减弱作为主要动态表型 | 任一工程不对称比例就是卒中 cutoff |
| 共轭凝视偏斜与较大幕上病灶和严重程度有关，但并非所有卒中都有 [3] | 在目标出现前增加自然直视试次，检测双眼共同静息偏向 | 缺少偏斜可以排除卒中 |
| 影像研究表明共轭凝视偏向与急性卒中有关，但测量定义和设备与普通摄像头不同 [10,11] | 记录连续静息偏向代理和浏览器实际刺激角，等待普通摄像头数据标定 | 影像角度阈值可以直接迁移为本系统阈值 |
| 核间性眼肌麻痹可表现为一眼内收障碍和对侧外展眼震 [12] | 仅把稳定终点的单眼方向缺失/显著不共轭作为辅助表型 | 约 5 FPS 能识别眼震、扫视动态或确诊 INO |
| 正常扫视是快速运动，参数随幅度而变化 [4] | 目标切换后排除预先规定的稳定段，只分析随后稳定终点 | 低帧率终点采集可以测扫视速度、潜伏期或 main sequence |
| 眼动采样率会显著影响扫视检测和持续时间估计，120 Hz 优于 60 Hz [5] | API 和文档明确标注 endpoint，不输出速度和潜伏期 | 5 FPS 与实验室视频眼震仪或高速眼动仪等效 |
| MediaPipe 虹膜模型是面部动画/虹膜定位模型，不是医疗视线仪 [6,7] | 使用个人中心终点作同轮基线；增加眼部像素、MAD 和重复性门控 | 虹膜在眼裂内的比值可直接换算成临床视角 |
| 正常水平扫视也会出现一定瞬时双眼不共轭 [8] | 只比较稳定终点的相对响应，并使用三轮稳健汇总 | 单帧或切换瞬间的最大双眼差就是病理性不共轭 |
| 头部或相机位移会造成很大的视线估计误差 [9] | 启用 MediaPipe canonical-face transformation matrix，按 pitch/yaw/roll 一轮内变化拒绝转头代偿 | 头姿质量门控已经把单目几何误差完全消除 |

## 刺激和采集结构

前端先记录屏幕物理宽度和眼睛到屏幕的估计距离，再按

```text
target_offset_cm = tan(requested_target_angle) × viewing_distance_cm
```

换算左右目标的 CSS 位置。视口无法容纳时会限幅，因此前端反算并向后端报告
`achieved_target_visual_angle_degrees`；报告同时保留请求角度、实际角度、
`viewing_distance_cm` 和 `screen_width_cm`。
这仍不是专业视野计的几何校准；浏览器缩放、显示器物理尺寸填写误差和观看距离变化
都会产生刺激角误差。

试次顺序为：

```text
自然直视 → 中心 → 左 → 中心 → 右 → 中心 → 右 →
中心 → 左 → 中心 → 左 → 中心 → 右
```

左右各重复三次并交替方向顺序。三次是可计算中位数并容许一次非典型响应的最小重复
结构。目标保持时间、切换后排除时间和请求刺激角均作为采集设计变量写入报告；在获得
目标设备数据前，它们不表述为生理或临床界值。每个侧方试次只与紧邻在前的中心试次
比较，减少慢性漂移。

## 测量质量量

当前默认版本不使用固定的样本数、有效率、眼部像素宽度、MAD、重复误差或头姿角度
界限来输出摄像头阳性、阴性或质量合格。程序逐试次记录有效/采集帧数、眼部像素宽度、
虹膜位置 MAD、三次响应离散度、头姿可用率和一轮内头姿角变化。缺少计算终点所必需的
关键点或没有任何可计算终点时，结果记为数据不足；其余情况保留连续指标，供健康受试者
技术实验估计设备与条件相关的测量分布。

左右各三次响应取中位数；选择三次的理由是中位数需要至少三个观测才能在保留一次
非典型响应时仍由另外两个观测确定。旧版固定质量门槛仅保留在显式开启的离线复现模式，
默认关闭，也不作为论文结论依据。

## 终点特征

每只眼、每个方向的可见响应为：

```text
response = configured_camera_sign × expected_direction
           × (lateral_endpoint - preceding_center_endpoint)
```

该响应是眼裂宽度归一化的无量纲终点差。MAD、眼部像素宽度和逐次离散度分别作为连续
质量量报告，不组合成未经目标设备验证的固定 SNR 判定。随后计算：

```text
directional_asymmetry =
    |binocular_leftward_response - binocular_rightward_response|
    / max(binocular_leftward_response, binocular_rightward_response)

directional_conjugacy_error =
    |left_eye_direction_response - right_eye_direction_response|
    / max(left_eye_direction_response, right_eye_direction_response)

rest_bias_degrees =
    (rest_endpoint - median_center_endpoint) / lateral_excursion
    × 2 × achieved_target_angle
```

当前发布版不为静息偏向、方向响应不对称或双眼终点差设置自动阳性/阴性界限，摄像头
完成后返回“已记录待验证指标”。突然视觉症状仍直接形成 E 警示。影像研究只支持凝视
偏向这一表型值得观察，不能为本项目普通摄像头代理提供可迁移数值。健康受试者实验可
估计重复性、失败率和参考分布；只有包含急性卒中、类似疾病与对照者的独立临床数据，
才能估计并验证诊断阈值、敏感度和特异度。

实现要求在采集配置中明确摄像头原始坐标是否镜像；不能从受试者响应反推，否则始终
看向目标反方向的任务错误也可能被解释为镜像。方向异常使用双眼平均响应；双眼终点
不协调按方向直接比较两眼各自在本眼裂内归一化后的终点响应，从而保留异常方向。
本实现明确不输出眼震、扫视速度、扫视潜伏期或平滑追踪增益；这些动态指标需要更高
采样率和相应刺激范式。

## 验证要求

在把自动阳性作为临床研究终点前，至少需要：

1. 用同步视频眼震仪或研究级眼动仪验证终点位移、方向和双眼相对响应；
2. 在目标树莓派和所有摄像头上测量不同距离、光照、眼镜、眼型和年龄的重复性；
3. 报告逐试次数据丢失率、MAD、重测一致性、ICC 和 Bland–Altman 一致性界限；
4. 纳入急性卒中、卒中类似疾病、既往斜视/眼肌麻痹和健康对照；
5. 以患者为单位划分开发集与独立验证集，用 ROC/PR 曲线确定研究 cutoff；
6. 预先规定偏向高灵敏度的阈值和不确定灰区，并报告 95% 置信区间；
7. 无论自动结果如何，突然视觉症状都不能被摄像头结果覆盖。

## 参考文献

1. Aroor S, Singh R, Goldstein LB. *BE-FAST (Balance, Eyes, Face, Arm,
   Speech, Time): Reducing the Proportion of Strokes Missed Using the FAST
   Mnemonic.* Stroke. 2017;48:479–481.
   <https://doi.org/10.1161/STROKEAHA.116.015169>
2. Kudo Y, Takahashi K, Sugawara E, et al. *Bedside video-oculographic
   evaluation of eye movements in acute supratentorial stroke patients: A
   potential biomarker for hemispatial neglect.* Journal of the Neurological
   Sciences. 2021;425:117442.
   <https://doi.org/10.1016/j.jns.2021.117442>
3. Singer OC, et al. *Conjugate eye deviation in acute stroke: incidence,
   hemispheric asymmetry, and lesion pattern.* Stroke. 2006.
   <https://pubmed.ncbi.nlm.nih.gov/17008621/>
4. Bahill AT, Clark MR, Stark L. *The main sequence, a tool for studying human
   eye movements.* Neurology. 1975;25:1065–1070.
   <https://doi.org/10.1212/WNL.25.11.1065>
5. *Sampling rate influences saccade detection in mobile eye tracking of a
   reading task.* 60 Hz and 120 Hz mobile trackers compared with a 1000 Hz
   stationary tracker:
   <https://pmc.ncbi.nlm.nih.gov/articles/PMC7141092/>
6. Bazarevsky V, Kartynnik Y, Vakunov A, Raveendran K, Grundmann M.
   *MediaPipe Iris: Real-time Iris Tracking & Depth Estimation.*
   <https://arxiv.org/abs/2006.11341>
7. Google. *MediaPipe Face Mesh V2 Model Card.*
   <https://storage.googleapis.com/mediapipe-assets/Model%20Card%20MediaPipe%20Face%20Mesh%20V2.pdf>
8. *Metrics of horizontal saccadic eye movements in normal humans.*
   Vision Research. <https://www.sciencedirect.com/science/article/pii/0042698971902124>
9. Karmali F, Shelhamer M. *Compensating for camera translation in video
   eye-movement recordings by tracking a representative landmark selected
   automatically by a genetic algorithm.* Journal of Neuroscience Methods.
   2009;176:157–165. <https://doi.org/10.1016/j.jneumeth.2008.09.010>
10. McKean D, et al. *Looking for a stroke: is the direction of gaze
    sensitive for acute ischaemic stroke?* European Radiology. 2014.
    <https://pubmed.ncbi.nlm.nih.gov/25172206/>
11. *CT-based assessment of spontaneous conjugate eye deviation in acute
    ischaemic stroke.* <https://pubmed.ncbi.nlm.nih.gov/26903074/>
12. Kim JS. *Internuclear ophthalmoplegia as an isolated or predominant
    symptom of brainstem infarction.* Neurology. 2004.
    <https://pubmed.ncbi.nlm.nih.gov/15136670/>
