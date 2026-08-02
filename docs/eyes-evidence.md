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
| 影像中的眼球凝视偏向约 11.95°～14° 时特异度较高，但灵敏度有限 [10,11] | 将 12° 作为研究型高幅度候选门槛，并记录浏览器实际刺激角 | 影像测角 cutoff 已经验证适用于普通摄像头 |
| 核间性眼肌麻痹可表现为一眼内收障碍和对侧外展眼震 [12] | 仅把稳定终点的单眼方向缺失/显著不共轭作为辅助表型 | 约 5 FPS 能识别眼震、扫视动态或确诊 INO |
| 正常扫视是几十毫秒级快速运动，参数随幅度而变化 [4] | 目标切换后先丢弃 0.5 秒，只分析随后稳定终点 | 默认约 5 FPS 可以测扫视速度、潜伏期或 main sequence |
| 眼动采样率会显著影响扫视检测和持续时间估计，120 Hz 优于 60 Hz [5] | API 和文档明确标注 endpoint，不输出速度和潜伏期 | 5 FPS 与实验室视频眼震仪或高速眼动仪等效 |
| MediaPipe 虹膜模型是面部动画/虹膜定位模型，不是医疗视线仪 [6,7] | 使用个人中心终点作同轮基线；增加眼部像素、MAD 和重复性门控 | 虹膜在眼裂内的比值可直接换算成临床视角 |
| 正常水平扫视也会出现一定瞬时双眼不共轭 [8] | 只比较稳定终点的相对响应，并使用三轮稳健汇总 | 单帧或切换瞬间的最大双眼差就是病理性不共轭 |
| 头部或相机位移会造成很大的视线估计误差 [9] | 启用 MediaPipe canonical-face transformation matrix，按 pitch/yaw/roll 一轮内变化拒绝转头代偿 | 头姿质量门控已经把单目几何误差完全消除 |

## 刺激和采集结构

前端先记录屏幕物理宽度和眼睛到屏幕的估计距离，再按

```text
target_offset_cm = tan(15°) × viewing_distance_cm
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

左右各重复三次并交替方向顺序。每个目标默认 2 秒，前 0.5 秒为切换稳定段，
只分析其后的保持注视终点。每个侧方试次只与紧邻在前的中心试次比较，减少慢性漂移。

## 质量门控

每个试次独立检查：

- 至少 5 个稳定段有效样本；
- 稳定段有效帧比例至少 60%；
- 有原始帧尺寸时，每只眼的眼裂宽度至少 24 像素；
- 眼裂内虹膜位置 MAD 不高于 0.08；
- 每个方向三次响应取中位数，允许一个离群试次；
- 至少 2/3 试次方向正确且达到响应 SNR；
- 中位数与最接近的另一轮相对差不高于 1.00；
- 至少 80% 有效帧带有三维头姿；
- 一轮内 pitch、yaw 或 roll 范围不超过 8°。

这些都是**设备和任务质量参数**，不是医学 cutoff。它们的作用是拒绝遮挡、模糊、
关键点抖动、动作不一致和转头代偿。正式研究应按目标树莓派、摄像头、分辨率、光照
和佩戴眼镜条件重新估计，并报告各拒绝原因的发生率。

重复误差采用三次响应的中位数，并允许一个离群试次，以降低约 5 FPS 下少量终点样本
变化对有效试次的影响。失败报告同时输出六组逐次响应、MAD 噪声、SNR、最近配对误差
和完整离散范围，便于后续按实测分布标定。

## 终点特征

每只眼、每个方向的可见响应为：

```text
response = expected_direction × (lateral_endpoint - preceding_center_endpoint)
response_snr = response / (1.4826 × (center_MAD + lateral_MAD))
```

三次响应中至少两次需方向正确且 `response_snr >= 3.5`；只有一次成功时返回
`insufficient`，三次全部失败才产生“可见目标响应减弱”提示。`3.5` 在这里是同轮
信号相对稳健噪声的工程显著性边界，不是眼动幅度的临床正常下限。

通过可见响应门控后，计算：

```text
directional_asymmetry =
    |binocular_leftward_response - binocular_rightward_response|
    / max(binocular_leftward_response, binocular_rightward_response)

conjugacy_relative_error =
    |left_eye_direction_fraction - right_eye_direction_fraction|

rest_bias_degrees =
    (rest_endpoint - median_center_endpoint) / lateral_excursion
    × 2 × achieved_target_angle
```

当前判定顺序为：

1. 两眼、两个方向全部无响应：视为没有证明完成任务，返回数据不足，而不是异常；
2. 双眼共同在同一方向持续无响应：提示方向性共轭凝视受限；
3. 一只眼在特定方向持续无响应：提示可能不共轭凝视受限；
4. 自然直视时两眼同向偏离达到 12°：提示静息共轭凝视偏向；
5. 双眼平均向一个方向的响应比相反方向弱 45%：提示方向性眼动减少；
6. 两眼归一化方向份额差达到 35%：提示可能核间性不共轭；
7. 其余可重复且通过质量门控的检查返回未见上述可见表型。

其中 12° 借鉴影像研究中高特异度区间 [10,11]，但影像眼球角度不能直接等同于
MediaPipe 终点估计，所以仍是待验证候选值；45% 和 35% 没有论文给出可直接迁移到
本设备的 cutoff，仍属于研究参数。左右眼总范围差仅保留在报告中用于质量分析，不单独
触发阳性：论文中的幕上卒中方向性眼动减少主要是双眼共同的方向异常，而健康人也可能
因单目尺度、眼裂形态和关键点误差出现小幅两眼范围差。

实现会根据本轮左右终点的总体分离自动识别摄像头是否镜像；方向异常使用双眼平均
响应，共轭误差先按每只眼自身总活动范围归一化，避免单眼尺度误差被除法放大。
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
