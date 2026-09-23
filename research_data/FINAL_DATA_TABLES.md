# 当前论文使用的数据

仅保留当前论文使用的源工作簿、去标识化分析表和可复现统计结果；
空白模板、审稿轮次副本、表格生成器和预览文件不纳入版本管理。

| 数据组 | 保留工作簿 | 分析用数据 |
|---|---|---|
| 验证实验 | [pibefast_validation_data.xlsx](../experiments/validation_study/pibefast_validation_data.xlsx) | 同目录的 `sessions.csv`、`events_and_references.csv`、`system_outputs.csv`、`face_eye_reference_measurements.csv` |
| Pi–PC pilot | [pipc_pilot_data.xlsx](pipc_pilot/pipc_pilot_data.xlsx) | `runs.csv`、`events.csv`、`resources.csv`、`derived_latency.csv` |
| 健康参与者重复测量 | [human_repeatability_data.xlsx](human_repeatability/human_repeatability_data.xlsx) | `participants.csv`、`sessions.csv`、`measurements.csv` |

分析脚本只生成结构化数据和统计结果，不再向论文源码目录写入未被正文引用的 LaTeX 表格。
