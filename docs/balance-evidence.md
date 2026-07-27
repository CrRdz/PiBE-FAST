# 持续平衡监测：证据、实现与限制

实现版本：`stroke-balance-evidence-v1`

## 定位

Balance 自动模块用于长期待机中的**个人变化监测**，不是力台检查、临床
lateropulsion 量表或卒中诊断器。本人或照护者报告突然失衡、协调困难或步态异常，
仍是 BE-FAST 的主要 B 信号 [1]。自动模块只在同一人的有效站立窗口已经形成基线后，
提示“躯干方向或横向摆动相对个人基线发生持续变化”。

本版没有加入 PASS 量表或把量表条目改写成摄像头分数；它们属于有人执行的时点式
临床评估，不适合作为本项目长期待机的自动标签。

## 论文到代码的映射

| 文献结论 | 当前实现 | 不能推导出的结论 |
|---|---|---|
| BE-FAST 加入步态失衡/腿无力可减少 FAST 漏检 [1] | 人工报告突然失衡时直接触发 B | 摄像头静站异常等于急性卒中 |
| Lateropulsion 是相对重力的额状面身体方向异常；身体倾斜是核心表现之一 [2,3] | `median_trunk_roll_degrees`：肩中心—髋中心躯干轴相对画面竖直方向的中位角度 | 单一角度可以完整识别 lateropulsion；临床定义还涉及主动推挤和抗拒纠正 |
| 卒中后负重不对称与更高 COP 速度/摆动有关，但关联和因果证据有限 [4] | 保留躯干相对双踝中点的位置序列作为横向运动学代理 | 双踝中点是 COP，或偏移方向就是患侧 |
| 亚急性卒中两次 30 秒静站中，COP 平均速度是可靠性较好的力台指标之一 [5] | 每个有效窗口采集 30 秒；`ml_sway_mean_velocity` 为主要摆动变化特征 | 摄像头速度与力台 COP 速度数值可互换 |
| COP 可靠性综述建议延长总采集时间，并平均 3～5 次重复试次 [6] | 默认积累 5 个有效 30 秒窗口形成个人基线；持续待机可逐步完成，不要求一次做完 | 五个窗口能建立临床参考范围或人群 cutoff |
| Kinect 可估计 TBCM 摆动，但存在分辨率、系统偏差和任务依赖性 [7] | API 明确标注 `camera_trunk_kinematic_proxy_not_cop` | MoveNet 单目 2D 已经得到 Kinect/Vicon 等效验证 |

## 特征定义

单帧要求左右肩、髋和踝均达到关键点质量门槛。

```text
trunk_center_x = mean(shoulder_center_x, hip_center_x)
ankle_midpoint_x = mean(left_ankle_x, right_ankle_x)
trunk_support_offset =
    (trunk_center_x - ankle_midpoint_x) / shoulder_width

trunk_roll_degrees =
    atan2(shoulder_center_x - hip_center_x,
          hip_center_y - shoulder_center_y)
```

窗口级输出包括：

- `median_trunk_roll_degrees`：持续侧倾方向代理；
- `ml_sway_rms`：横向位置相对窗口中位数的 RMS；
- `ml_sway_p95_range`：横向位置第 5～95 百分位范围；
- `ml_sway_path_length`：相邻样本横向位移绝对值之和；
- `ml_sway_mean_velocity`：路径长度除以有效窗口时长；
- `trunk_roll_rms_degrees`：躯干侧倾角围绕其中位数的 RMS。

这些名称有意不使用 `COP` 或 `center_of_mass`。

`balance_min_valid_samples=30` 与 `balance_min_valid_fraction=0.75` 只负责拒绝
关键点缺失严重的窗口，不参与异常判定；目前没有文献能为本项目的 MoveNet 帧率、
镜头和遮挡条件提供可直接移植的质量门槛，因此它们仍明确标记为待外部验证的实现参数。
Ruhe 等对力台 COP 建议约 100 Hz，而本项目树莓派待机默认只有约 2 FPS；两者不是
同一种测量条件。因此 `ml_sway_mean_velocity` 只能解释为同一设备、同一采样配置下的
低频相对变化代理，不能声称达到力台 COP 的可靠性。

## 个人基线判定

当前没有论文给出可直接迁移到 MoveNet、当前镜头和家庭人群的卒中 cutoff。因此代码
不再使用 `0.40 × 肩宽` 或 `0.50 × 肩宽`。前五个合格窗口只用于基线校准；完成后，
以 baseline median 和 MAD（同时用 IQR 防止 MAD 退化）计算稳健变化分数：

```text
robust_score = |current - baseline_median| / robust_scale
```

躯干方向采用双侧变化；摆动速度只把“增加”作为变化方向。`3.5` 是稳健异常检测中
常用的 modified Z-score 界值 [8]，只具有统计过程监测含义，不是医学 cutoff。
一个完整 30 秒窗口本身就是持续性门槛；瞬时单帧不会触发。

代码不为 MAD/IQR 人为设置“最小噪声”工程值。如果个人基线的离散度恰好为零、
后续值却发生改变，则 modified Z-score 数学上不可估计，系统返回
`personal_balance_baseline_variability_not_estimable`，而不是用任意小常数制造阳性
或阴性结论。

基线尚未完成时输出 `personal_balance_baseline_calibrating`，不会输出自动阴性。
基线完成后：

- 躯干方向变化超过个人稳健范围：
  `sustained_trunk_orientation_change`；
- 横向摆动平均速度增加并超过个人稳健范围：
  `increased_mediolateral_sway_velocity`；
- 两者均未变化：
  `no_sustained_change_from_personal_balance_baseline`。

长期待机在受试者自然进入完整站立画面时机会性采集这些窗口；异常窗口只负责打开
主动筛查，不直接形成卒中诊断。画面中的左/右倾斜方向也不映射为卒中患侧。

## 验证要求

目前的论文依据支持的是**特征和采集结构**，不支持本项目已经具有临床准确度。至少还需：

1. 用同步 Vicon/深度相机验证 MoveNet 的躯干角度与横向运动学误差；
2. 用同步力台研究运动学代理与 COP 平均速度、RMS 和路径长度的关联；
3. 在家庭镜头位置、衣着、遮挡和不同辅助器具条件下重复测量；
4. 纳入急性卒中、卒中类似疾病、既往平衡障碍及健康对照；
5. 采用患者级训练/验证划分，并报告灵敏度、特异度、置信区间和误报率；
6. 不能把“与个人基线一致”解释成排除卒中。

## 参考文献

1. Aroor S, Singh R, Goldstein LB. *BE-FAST (Balance, Eyes, Face, Arm,
   Speech, Time): Reducing the Proportion of Strokes Missed Using the FAST
   Mnemonic.* Stroke. 2017;48:479–481.
   <https://doi.org/10.1161/STROKEAHA.116.015169>
2. Dai S, Piscicelli C, Clarac E, et al. *Lateropulsion After Hemispheric
   Stroke: A Form of Spatial Neglect Involving Graviception.* Neurology.
   2021;96:e2160–e2171. <https://doi.org/10.1212/WNL.0000000000011826>
3. Dai S, Lemaire C, Piscicelli C, Pérennou D. *Lateropulsion Prevalence
   After Stroke: A Systematic Review and Meta-analysis.* Neurology.
   2022;98:e1574–e1584.
   <https://doi.org/10.1212/WNL.0000000000200010>
4. Kamphuis JF, de Kam D, Geurts ACH, Weerdesteyn V. *Is Weight-Bearing
   Asymmetry Associated with Postural Instability after Stroke? A Systematic
   Review.* Stroke Research and Treatment. 2013;2013:692137.
   <https://doi.org/10.1155/2013/692137>
5. Aryan R, Inness E, Patterson KK, Mochizuki G, Mansfield A. *Reliability
   of force plate-based measures of standing balance in the sub-acute stage
   of post-stroke recovery.* Heliyon. 2023;9:e21046.
   <https://doi.org/10.1016/j.heliyon.2023.e21046>
6. Ruhe A, Fejer R, Walker B. *The test-retest reliability of centre of
   pressure measures in bipedal static task conditions—a systematic review
   of the literature.* Gait & Posture. 2010;32:436–445.
   <https://doi.org/10.1016/j.gaitpost.2010.09.012>
7. Yeung LF, Cheng KC, Fong CH, Lee WCC, Tong KY. *Evaluation of the
   Microsoft Kinect as a clinical assessment tool of body sway.* Gait &
   Posture. 2014;40:532–538.
   <https://doi.org/10.1016/j.gaitpost.2014.06.012>
8. Iglewicz B, Hoaglin DC. *How to Detect and Handle Outliers.* ASQC Quality
   Press; 1993.
