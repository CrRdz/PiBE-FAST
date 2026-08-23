# PiBE-FAST 离线训练

当前仓库仅保留普通话构音障碍语音表征的离线训练流程。A（Arms）使用 MoveNet
关键点、动作完整性门控、双臂高度差和单侧下落差进行规则判定，不加载或训练额外的
手臂分类模型。

语音数据准备、说话人分组、特征提取、模型训练、运行时 S 判定接入和目标麦克风
验证见 [`docs/mdsc-speech-training.md`](../docs/mdsc-speech-training.md)。

训练公共逻辑位于 `training/modeling.py`，语音相关脚本位于 `training/speech/`。
