"""集中定义引导式筛查的阶段名、跳过流转和界面提示语。"""

# 真正运行模型采样的四个自动检查阶段。
ACTIVE_STAGES = ("eyes", "face", "arms", "balance")

# 把等待、重试和采样状态统一映射回所属检查，供“跳过当前项”使用。
SKIP_STAGE_ALIASES = {
    "idle": "eyes",
    "eyes": "eyes",
    "retry_eyes": "eyes",
    "ready_face": "face",
    "face": "face",
    "retry_face": "face",
    "ready_arms": "arms",
    "arms": "arms",
    "retry_arms": "arms",
    "ready_balance": "balance",
    "balance": "balance",
    "retry_balance": "balance",
}

# 每个检查被跳过后，需要写入的结果属性、重置的检测器和下一阶段。
SKIP_FLOW = {
    "eyes": ("eye_result", "eye_screen", "ready_face"),
    "face": ("face_result", "face_screen", "ready_arms"),
    "arms": ("arm_result", "arm_screen", "ready_balance"),
    "balance": ("balance_result", "balance_screen", "review"),
}

# 阶段提示只负责文案，不参与状态机判断。
PROMPTS = {
    "eyes": "Keep your head still and follow the moving target using only your eyes.",
    "retry_eyes": "Center your face and eyes so the eye check can restart.",
    "ready_face": "Eye movement check finished. Prepare for the smile check.",
    "face": "Keep a neutral face, then smile when prompted.",
    "retry_face": "Show your full face and relax before restarting the smile check.",
    "ready_arms": "Face check finished. Sit down and prepare to raise both arms.",
    "arms": "Keep both arms raised forward and level.",
    "retry_arms": "Show both shoulders and wrists, then raise both arms again.",
    "ready_balance": "Arm check finished. Only start balance check if safe.",
    "balance": "Stand still with support nearby; stop if unsafe.",
    "retry_balance": "Show your full body and stand only with support nearby.",
    "speech_ready": "Prepare to repeat the displayed sentence into the microphone.",
    "speech_recording": "Speak the displayed sentence clearly while recording continues.",
    "review": "Review speech, sudden onset, and all automated results.",
}


def stage_prompt(stage: str, mode: str) -> str:
    """根据当前阶段和待机/筛查模式返回面向用户的操作提示。"""

    if stage == "idle":
        if mode == "standby":
            return "The Raspberry Pi is in low-load standby. Start a screen when needed."
        return "Choose any B, E, F, A, or S check to begin."
    return PROMPTS.get(stage, "Review the screening result.")
