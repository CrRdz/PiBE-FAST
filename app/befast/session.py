"""线程安全地编排一轮引导式 BE-FAST 筛查及其阶段状态。"""

from __future__ import annotations

from copy import deepcopy
import threading
import time
from uuid import uuid4
from typing import Any, Mapping, Sequence

from app.face_landmarker import FaceObservation

from .arms import ArmDriftScreen
from .balance import BalanceScreen, PersonalBalanceBaseline
from .config import BefastConfig
from .eyes import EyeMovementScreen
from .face import FaceSmileScreen
from .fusion import build_feature_fusion
from .guidance import face_guidance, guidance_value, pose_guidance
from .report import build_report_items
from .result import MotionResult
from .stages import ACTIVE_STAGES, SKIP_FLOW, SKIP_STAGE_ALIASES, stage_prompt
from .urgency import screening_assessment


class BefastSession:
    """供推理工作线程和 Web 请求线程共享的待机/筛查会话。"""

    # B 的安全/症状问题仍需明确提交；S 改由本地麦克风自动分析。
    MANUAL_KEYS = ("balance_problem",)
    COMPONENT_STAGES = {
        "E": "eyes",
        "F": "face",
        "A": "arms",
        "B": "balance",
    }
    STAGE_COMPONENTS = {
        stage: code for code, stage in COMPONENT_STAGES.items()
    }
    SKIP_STAGE_ALIASES = SKIP_STAGE_ALIASES
    SKIP_FLOW = SKIP_FLOW

    def __init__(
        self,
        config: BefastConfig | None = None,
        balance_baseline: PersonalBalanceBaseline | None = None,
    ) -> None:
        """创建所有独立检测器，并将设备初始化到待机状态。"""

        self.config = config or BefastConfig()
        # 同一线程内的方法会嵌套进入锁，因此使用可重入锁 RLock。
        self.lock = threading.RLock()
        self.eye_screen = EyeMovementScreen(self.config)
        self.face_screen = FaceSmileScreen(self.config)
        self.arm_screen = ArmDriftScreen(self.config)
        self.balance_screen = BalanceScreen(self.config, balance_baseline)
        self.reset()

    def reset(self) -> None:
        """回到低负载待机模式，并清空上一轮筛查数据。"""

        with self.lock:
            self.mode = "standby"
            self.screening_started_at: float | None = None
            self.trigger: dict[str, Any] | None = None
            self._clear_screen_locked()

    def _clear_screen_locked(self) -> None:
        """在已经持锁时清空筛查状态，但保留调用方设置的 mode。"""

        # 四个自动项目重新置为未运行，人工项重新置为未提交。
        self.session_id = uuid4().hex
        self.stage = "idle"
        self.stage_started_at: float | None = None
        self.eye_result = MotionResult()
        self.face_result = MotionResult()
        self.arm_result = MotionResult()
        self.balance_result = MotionResult()
        self.speech_result = MotionResult()
        self.manual_complete = False
        self.reported_functional_problems: dict[str, bool] = {}
        self.symptom_evidence: dict[str, list[dict[str, Any]]] = {}
        self.speech_attempt_id: str | None = None
        self.required_components = ("B", "E", "F", "A", "S")
        self.manual_completed = {"B": False, "E": False, "S": False}
        self.manual: dict[str, bool | None] = {
            key: None for key in self.MANUAL_KEYS
        }
        self.manual["eye_problem"] = None
        self.new_or_sudden: bool | None = None
        self.onset_time: str | None = None
        self.component_onsets: dict[str, bool | None] = {
            code: None for code in ("B", "E", "F", "A", "S")
        }
        self.component_onset_times: dict[str, str | None] = {
            code: None for code in ("B", "E", "F", "A", "S")
        }
        self.guidance: dict[str, Any] = {
            "ready": False,
            "status": "waiting",
            "reason": "waiting_for_screening_trigger",
            "stage": "idle",
            "updated_at": None,
            "metrics": {},
        }
        self.retry_counts = {stage: 0 for stage in ACTIVE_STAGES}
        self.attempt_counts = {code: 0 for code in ("B", "E", "F", "A", "S")}
        self.reports: list[dict[str, Any]] = []
        self.current_report: dict[str, Any] | None = None
        self.active_component: str | None = None
        # 检测器内部还保存样本序列，必须与会话结果一起重置。
        for screen in self._screens().values():
            screen.reset()

    def start_screening(
        self,
        source: str = "user",
        reason: str = "manual_request",
        now: float | None = None,
        trigger_id: str | None = None,
    ) -> None:
        """响应用户、定时任务或被动触发，打开全新的独立项目选择会话。"""

        ts = time.time() if now is None else float(now)
        with self.lock:
            self._clear_screen_locked()
            self.mode = "screening"
            self.screening_started_at = ts
            # 保存触发来源，便于日志和前端解释为什么进入筛查模式。
            self.trigger = {
                "source": str(source),
                "reason": str(reason),
                "triggered_at": round(ts, 4),
            }
            if trigger_id:
                self.trigger["id"] = str(trigger_id)
            self._set_guidance(False, "choose_component", ts)

    def _has_active_symptom_locked(self, code: str) -> bool:
        """Ordinary questionnaire answers do not retract earlier symptom events."""
        return any(event.get("active", True) for event in self.symptom_evidence.get(code, []))

    def prepare_component(self, code: str, now: float | None = None) -> None:
        """选择一个独立 BE-FAST 项目，并只清空该项目本次采集的数据。"""

        normalized = str(code).strip().upper()
        if normalized not in {"B", "E", "F", "A", "S"}:
            raise ValueError("component must be 'B', 'E', 'F', 'A', or 'S'")
        ts = time.time() if now is None else float(now)
        with self.lock:
            self._ensure_screening_locked(ts, "component_api", "component_selected")
            self.current_report = None
            self.active_component = normalized
            self.speech_attempt_id = None
            self.stage_started_at = None
            if normalized == "S":
                self.speech_result = MotionResult()
                self.manual_completed["S"] = False
                self.stage = "speech_ready"
                self._set_guidance(False, "prepare_speech", ts)
                return
            if normalized == "B":
                self.balance_result = MotionResult()
                self.balance_screen.reset()
                self.manual["balance_problem"] = None
                self.manual_completed["B"] = False
                self.manual_complete = False
                self.stage = "manual_balance"
                self._set_guidance(False, "manual_observation_required", ts)
                return
            if normalized == "E":
                self.eye_result = MotionResult()
                self.eye_screen.reset()
                self.eye_screen.configure_setup()
                self.manual["eye_problem"] = None
                self.manual_completed["E"] = False
                self.stage = "manual_eyes"
                self._set_guidance(False, "manual_visual_observation_required", ts)
                return

            stage = self.COMPONENT_STAGES[normalized]
            result_attr, screen = self._stage_binding(stage)
            setattr(self, result_attr, MotionResult())
            screen.reset()
            self.stage = {
                "eyes": "retry_eyes",
                "face": "ready_face",
                "arms": "ready_arms",
            }[stage]
            self._set_guidance(False, f"prepare_{stage}", ts)

    def open_component_menu(self, now: float | None = None) -> None:
        """返回独立项目选择页，同时保留本轮所有历史报告。"""

        ts = time.time() if now is None else float(now)
        with self.lock:
            self._ensure_screening_locked(ts, "component_api", "component_menu_opened")
            self.stage = "idle"
            self.stage_started_at = None
            self.active_component = None
            self.current_report = None
            self._set_guidance(False, "choose_component", ts)

    def start_stage(self, stage: str, now: float | None = None) -> None:
        """启动指定自动检查；必要时自动从待机切换到筛查模式。"""

        if stage not in ACTIVE_STAGES:
            raise ValueError("stage must be 'eyes', 'face', 'arms', or 'balance'")
        ts = time.time() if now is None else float(now)
        with self.lock:
            if self.mode != "screening":
                # 允许测试或 API 直接启动某一阶段，同时仍建立完整会话上下文。
                self._clear_screen_locked()
                self.mode = "screening"
                self.screening_started_at = ts
                self.trigger = {
                    "source": "stage_api",
                    "reason": "direct_stage_start",
                    "triggered_at": round(ts, 4),
                }
            if stage == "balance" and self._has_active_symptom_locked("B"):
                raise ValueError("correct active balance symptom reports before starting a standing task")
            self.current_report = None
            self.active_component = self.STAGE_COMPONENTS[stage]
            self.stage = stage
            self.stage_started_at = ts
            self._set_guidance(False, "collecting_action", ts)
            result_attr, screen = self._stage_binding(stage)
            # checking 状态会立即反映到 API，随后检测器从空样本开始采集。
            setattr(self, result_attr, MotionResult(status="checking", reason="checking"))
            screen.start(ts)

    def ready_arm_phase(self, now: float | None = None) -> None:
        """确认完整双臂平举动作已看懂，并启动 3 秒倒数。"""

        ts = time.time() if now is None else float(now)
        with self.lock:
            if self.stage != "arms" or self.active_component != "A":
                raise ValueError("the arm check must be active before confirming readiness")
            # Hands-free start may win the race with a still-visible button.
            # Treat a duplicate confirmation during countdown as idempotent.
            if self.arm_screen.phase_state == "countdown":
                return
            self.arm_screen.ready_phase(ts)
            self.stage_started_at = ts
            self._set_guidance(False, "arm_phase_countdown", ts)

    def skip_current_stage(self, now: float | None = None) -> str:
        """跳过当前自动检查，但用 skipped 明确区分于正常阴性。"""

        ts = time.time() if now is None else float(now)
        with self.lock:
            if self.mode != "screening":
                raise ValueError("a screening must be active before a check can be skipped")
            check = self.SKIP_STAGE_ALIASES.get(self.stage)
            if check is None:
                raise ValueError("the current stage cannot be skipped")
            result_attr, screen_attr, next_stage = self.SKIP_FLOW[check]
            # 跳过后丢弃可能已收集的半轮样本，避免下次复用旧数据。
            setattr(self, result_attr, MotionResult(status="skipped", reason="user_skipped"))
            getattr(self, screen_attr).reset()
            self.stage = next_stage
            self.stage_started_at = None
            self._set_guidance(False, f"prepare_{next_stage}", ts)
            self._record_component_report_locked(
                self.STAGE_COMPONENTS[check], ts
            )
            return check

    def submit_component_observation(
        self,
        code: str,
        problem: bool,
        new_or_sudden: bool | None,
        onset_time: str | None = None,
        viewing_distance_cm: float | None = None,
        screen_width_cm: float | None = None,
        achieved_target_visual_angle_degrees: float | None = None,
        now: float | None = None,
    ) -> None:
        """提交 B/E 人工观察；E 无主观症状时再进入摄像头辅助检查。"""

        normalized = str(code).strip().upper()
        if normalized not in {"B", "E"}:
            raise ValueError("manual component must be 'B' or 'E'")
        if not isinstance(problem, bool) or (new_or_sudden is not None and not isinstance(new_or_sudden, bool)):
            raise ValueError("problem must be boolean; onset must be boolean or null")
        ts = time.time() if now is None else float(now)
        with self.lock:
            self._ensure_screening_locked(
                ts, "manual_form", "manual_component_submitted"
            )
            self.active_component = normalized
            manual_key = (
                "balance_problem" if normalized == "B" else "eye_problem"
            )
            if problem:
                self._append_symptom_locked(normalized, new_or_sudden, ts)
            self.manual[manual_key] = problem
            self.manual_completed[normalized] = True
            self.manual_complete = self.manual_completed["B"]
            self.new_or_sudden = new_or_sudden
            self.onset_time = str(onset_time).strip() if onset_time else None
            # A symptom answer supplies its own onset; it must not rewrite
            # the onset already attached to an acquired positive measurement.
            preserve_measurement_onset = (
                normalized == "B" and self.balance_result.status == "positive"
            )
            if not preserve_measurement_onset:
                self.component_onsets[normalized] = new_or_sudden
                self.component_onset_times[normalized] = self.onset_time

            if normalized == "B" and not self._has_active_symptom_locked("B") and not problem and self.balance_result.status in {
                "not_run",
                "checking",
            }:
                # 没有主观失衡时继续完成姿态采集，避免仅凭勾选框给出阴性。
                self.stage = "ready_balance"
                self.stage_started_at = None
                self.current_report = None
                self._set_guidance(False, "prepare_balance", ts)
                return
            if normalized == "E" and not problem:
                self.eye_screen.configure_setup(
                    viewing_distance_cm=viewing_distance_cm,
                    screen_width_cm=screen_width_cm,
                    achieved_target_visual_angle_degrees=(
                        achieved_target_visual_angle_degrees
                    ),
                )
                self.stage = "retry_eyes"
                self.stage_started_at = None
                self.current_report = None
                self._set_guidance(False, "prepare_eyes", ts)
                return

            self.stage = "report"
            self.stage_started_at = None
            self._record_component_report_locked(normalized, ts)

    def start_speech_recording(self, now: float | None = None) -> str:
        """Mark S as actively recording while the audio service runs."""

        ts = time.time() if now is None else float(now)
        with self.lock:
            if self.stage != "speech_ready" or self.active_component != "S":
                raise ValueError("select the S component before recording speech")
            self.speech_result = MotionResult(
                status="checking", reason="recording_speech"
            )
            self.speech_attempt_id = uuid4().hex
            self.stage = "speech_recording"
            self.stage_started_at = ts
            self._set_guidance(False, "recording_speech", ts)
            return self.speech_attempt_id

    def submit_speech_result(
        self,
        result: MotionResult,
        *,
        new_or_sudden: bool | None,
        onset_time: str | None = None,
        now: float | None = None,
    ) -> None:
        """Finish S; recorded attempt IDs reject stale asynchronous results."""

        if result.status not in {"positive", "negative", "insufficient"}:
            raise ValueError("speech result must be positive, negative, or insufficient")
        if new_or_sudden is not None and not isinstance(new_or_sudden, bool):
            raise ValueError("new_or_sudden must be boolean or null")
        ts = time.time() if now is None else float(now)
        with self.lock:
            if self.active_component != "S" or self.stage not in {
                "speech_ready",
                "speech_recording",
            }:
                raise ValueError("no speech component is awaiting a result")
            attempt = result.details.get("speech_attempt_id")
            if (self.stage == "speech_recording" or attempt is not None) and attempt != self.speech_attempt_id:
                raise ValueError("stale speech result")
            self.speech_attempt_id = None
            self.speech_result = result
            self.manual_completed["S"] = True
            self.new_or_sudden = new_or_sudden
            self.onset_time = str(onset_time).strip() if onset_time else None
            self.component_onsets["S"] = new_or_sudden
            self.component_onset_times["S"] = self.onset_time
            self.stage = "report"
            self.stage_started_at = None
            self._set_guidance(False, result.reason, ts)
            self._record_component_report_locked("S", ts)

    def submit_manual(
        self,
        observations: Mapping[str, bool],
        new_or_sudden: bool,
        onset_time: str | None = None,
    ) -> None:
        """提交兼容旧调用方的 B 人工观察和起病时间。"""

        missing = [key for key in self.MANUAL_KEYS if key not in observations]
        if missing:
            raise ValueError(f"missing manual observations: {', '.join(missing)}")
        with self.lock:
            if self.mode != "screening":
                # 独立提交人工表单时也生成一轮可追踪的筛查会话。
                self._clear_screen_locked()
                self.mode = "screening"
                self.screening_started_at = time.time()
                self.trigger = {
                    "source": "manual_form",
                    "reason": "manual_observation_submitted",
                    "triggered_at": round(self.screening_started_at, 4),
                }
            for key in self.MANUAL_KEYS:
                value = observations[key]
                if not isinstance(value, bool):
                    raise ValueError(f"{key} must be a boolean")
                self.manual[key] = value
            if self.manual["balance_problem"] is True:
                self._append_symptom_locked("B", new_or_sudden, time.time())
            self.manual_completed["B"] = True
            self.new_or_sudden = bool(new_or_sudden)
            self.onset_time = str(onset_time).strip() if onset_time else None
            if self.balance_result.status != "positive":
                self.component_onsets["B"] = self.new_or_sudden
                self.component_onset_times["B"] = self.onset_time
            self.manual_complete = True
            self._refresh_evidence_report_locked("B", "symptom_update")
            # 人工表单完成后进入汇总页，但不强行打断正在采样的自动阶段。
            if self.stage in {"idle", "ready_balance", "review"}:
                self.stage = "review"

    def observe_face(self, ts: float, observation: FaceObservation | None) -> None:
        """等待启动 E/F 时只更新取景引导，不累计正式检测样本。"""

        with self.lock:
            self._update_face_guidance(ts, observation)

    def update_face(self, ts: float, observation: FaceObservation | None) -> None:
        """按面部模型推理频率推进当前 E 或 F 自动检查。"""

        with self.lock:
            # 先发布本帧引导，再把同一帧交给当前正式检测器。
            self._update_face_guidance(ts, observation)
            if self.stage == "eyes" and self.eye_screen.update(ts, observation):
                self.eye_result = self.eye_screen.finish()
                self._finish_or_retry(
                    "eyes", self.eye_result, "ready_face", self.eye_screen, ts
                )
            elif self.stage == "face" and self.face_screen.update(ts, observation):
                self.face_result = self.face_screen.finish()
                self._finish_or_retry(
                    "face", self.face_result, "ready_arms", self.face_screen, ts
                )

    def observe_pose(
        self,
        ts: float,
        keypoints: Sequence[Mapping[str, float]],
        pose: str,
    ) -> None:
        """等待启动 A/B 时只更新动作引导，不累计正式检测样本。"""

        with self.lock:
            self._update_pose_guidance(ts, keypoints, pose)

    def update(
        self,
        ts: float,
        keypoints: Sequence[Mapping[str, float]],
        pose: str,
    ) -> None:
        """处理一帧姿态关键点，并推进当前 A 或 B 自动检查。"""

        with self.lock:
            self._update_pose_guidance(ts, keypoints, pose)
            if self.stage == "arms" and self.arm_screen.update(ts, keypoints):
                self.arm_result = self.arm_screen.finish()
                self._finish_or_retry(
                    "arms", self.arm_result, "ready_balance", self.arm_screen, ts
                )
            elif self.stage == "balance" and self.balance_screen.update(
                ts, keypoints, pose
            ):
                self.balance_result = self.balance_screen.finish()
                self._finish_or_retry(
                    "balance", self.balance_result, "review", self.balance_screen, ts
                )

    def snapshot(self, now: float | None = None) -> dict[str, Any]:
        """在线程锁内生成当前会话的完整、可序列化快照。"""

        ts = time.time() if now is None else float(now)
        with self.lock:
            items = build_report_items(
                self.manual,
                self.manual_complete,
                self.eye_result,
                self.face_result,
                self.arm_result,
                self.balance_result,
                self.speech_result,
                manual_completed=self.manual_completed,
                reported_functional_problems=self.reported_functional_problems,
                symptom_evidence=self.symptom_evidence,
            )
            if self.mode == "standby":
                # 待机状态不运行最终判定，避免把未检查项目显示为 incomplete。
                assessment = {
                    "decision": "standby",
                    "urgency": "none",
                    "completeness": "not_started",
                    "positive_components": [],
                    "urgent_components": [],
                    "reasons": ["waiting_for_screening_trigger"],
                }
            else:
                assessment = screening_assessment(items, self.component_onsets, self.required_components)
            decision = str(assessment["decision"])
            reasons = list(assessment["reasons"])
            elapsed = (
                max(0.0, ts - self.stage_started_at)
                if self.stage_started_at is not None
                else 0.0
            )
            duration = self._active_duration()
            progress = min(1.0, elapsed / duration) if duration > 0 else 0.0
            if self.stage == "arms":
                progress = self.arm_screen.progress(ts)
            snapshot = self._snapshot_payload(
                items, decision, reasons, progress
            )
            snapshot["required_components"] = list(self.required_components)
            snapshot["urgency"] = assessment["urgency"]
            snapshot["completeness"] = assessment["completeness"]
            snapshot["positive_components"] = list(
                assessment["positive_components"]
            )
            snapshot["urgent_components"] = list(
                assessment["urgent_components"]
            )
            if self.stage == "eyes":
                # E 阶段额外告诉前端当前目标以及目标切换倒计时。
                snapshot["eye_target"] = self.eye_screen.target(ts)
                target_elapsed = elapsed % self.config.eye_target_seconds
                snapshot["eye_target_remaining"] = round(
                    max(0.0, self.config.eye_target_seconds - target_elapsed), 1
                )
                snapshot["eye_trial"] = self.eye_screen.trial_index(ts) + 1
                snapshot["eye_trial_total"] = len(self.eye_screen.TRIALS)
                snapshot["eye_settling"] = self.eye_screen.is_settling(ts)
            if self.stage == "face":
                # F 阶段额外告诉前端当前应保持中性还是微笑。
                snapshot["face_phase"] = self.face_screen.phase(ts)
            if self.stage == "arms":
                snapshot["arm_phase"] = self.arm_screen.phase(ts)
                arm_phase_index, arm_phase_remaining = self.arm_screen.phase_status(ts)
                snapshot["arm_phase_index"] = arm_phase_index + 1
                snapshot["arm_phase_total"] = len(self.arm_screen.PHASES)
                snapshot["arm_phase_remaining"] = round(arm_phase_remaining, 1)
                snapshot["arm_phase_state"] = self.arm_screen.phase_state
                snapshot["arm_phase_failure_reason"] = (
                    self.arm_screen.phase_failure_reason
                )
                snapshot["arm_phase_metrics"] = dict(
                    self.arm_screen.phase_completion_metrics.get(
                        self.arm_screen.current_phase, {}
                    )
                )
                auto_ready_active, auto_ready_remaining = (
                    self.arm_screen.auto_ready_status(ts)
                )
                snapshot["arm_auto_ready_enabled"] = bool(
                    self.config.arm_auto_ready_enabled
                )
                snapshot["arm_auto_ready_active"] = auto_ready_active
                snapshot["arm_auto_ready_remaining"] = round(
                    auto_ready_remaining, 1
                )
            return snapshot

    def _snapshot_payload(
        self,
        items: dict[str, dict[str, Any]],
        decision: str,
        reasons: list[str],
        progress: float,
    ) -> dict[str, Any]:
        """组装不依赖当前时间计算的快照公共字段。"""

        return {
            "session_id": self.session_id,
            "mode": self.mode,
            "stage": self.stage,
            "prompt": stage_prompt(self.stage, self.mode),
            "progress": round(progress, 4),
            "decision": decision,
            "emergency": decision == "emergency",
            "reasons": reasons,
            "items": items,
            "new_or_sudden": self.new_or_sudden,
            "onset_time": self.onset_time,
            "component_onsets": dict(self.component_onsets),
            "component_onset_times": dict(self.component_onset_times),
            "manual_complete": self.manual_complete,
            "manual_completed": dict(self.manual_completed),
            "screening_started_at": self.screening_started_at,
            "trigger": dict(self.trigger) if self.trigger is not None else None,
            "active_component": self.active_component,
            "current_report": (
                self._copy_report(self.current_report)
                if self.current_report is not None
                else None
            ),
            "reports": [self._copy_report(report) for report in self.reports],
            "attempt_counts": dict(self.attempt_counts),
            "guidance": {
                **self.guidance,
                # 复制嵌套字典，防止调用方修改会话内部实时指标。
                "metrics": dict(self.guidance.get("metrics", {})),
            },
            "retry_counts": dict(self.retry_counts),
            "eye_setup": dict(self.eye_screen.setup),
            "live_collection": self._live_collection_snapshot(),
            "factor_thresholds": self._factor_thresholds_snapshot(),
            # This is an auditable feature vector for research/model training.
            # It intentionally has no authority over ``decision`` or emergency.
            "fusion": build_feature_fusion(items, self.config),
            "disclaimer": (
                "Screening prototype only; it cannot diagnose or exclude stroke. "
                "Any sudden BE-FAST sign requires emergency medical help."
            ),
        }

    def _live_collection_snapshot(self) -> dict[str, Any]:
        """Expose the current check's collected samples without leaking mutable state."""

        screen = self._screens().get(self.stage.replace("retry_", "").replace("ready_", ""))
        if screen is None:
            return {"metrics": {}, "valid_samples": 0, "captured_samples": 0}
        payload: dict[str, Any] = {
            "metrics": {
                str(key): round(float(value), 5)
                for key, value in getattr(screen, "live_metrics", {}).items()
            }
        }
        if screen is self.eye_screen:
            valid = sum(self.eye_screen.valid_frames_by_trial)
            captured = sum(self.eye_screen.capture_frames_by_trial)
            payload.update(
                {
                    "valid_samples": valid,
                    "captured_samples": captured,
                    "trial_valid_samples": list(self.eye_screen.valid_frames_by_trial),
                    "trial_captured_samples": list(self.eye_screen.capture_frames_by_trial),
                }
            )
        else:
            valid = int(getattr(screen, "valid_frames", 0))
            captured = int(getattr(screen, "capture_frames", 0))
            payload.update(
                {
                    "valid_samples": valid,
                    "captured_samples": captured,
                }
            )
            if screen is self.face_screen:
                payload["neutral_samples"] = len(self.face_screen.neutral_samples)
                payload["smile_samples"] = len(self.face_screen.smile_samples)
            if screen is self.balance_screen:
                baseline = self.balance_screen.baseline.snapshot()
                payload["baseline_windows"] = baseline["windows"]
                payload["baseline_target"] = baseline["target_windows"]
        payload["valid_fraction"] = round(valid / max(captured, 1), 4)
        return payload

    def _factor_thresholds_snapshot(self) -> dict[str, float]:
        """Return the engineering thresholds used by the UI factor indicators."""

        names = (
            "eye_max_gaze_mad",
            "eye_max_repeat_relative_error",
            "eye_max_head_rotation_degrees",
            "eye_landmark_noise_floor_pixels",
            "eye_response_snr_threshold",
            "eye_directional_asymmetry_threshold",
            "eye_conjugacy_relative_error_threshold",
            "eye_rest_gaze_deviation_degrees_threshold",
            "eye_min_valid_fraction_per_trial",
            "face_min_valid_fraction",
            "face_min_smile_score",
            "face_corner_delta_threshold",
            "face_smile_score_difference_threshold",
            "arm_min_valid_fraction",
            "arm_raise_wrist_height_tolerance",
            "arm_min_elbow_angle_degrees",
            "arm_min_lateral_reach",
            "arm_level_difference_threshold",
            "arm_drift_difference_threshold",
            "balance_min_valid_fraction",
            "balance_robust_z_threshold",
        )
        thresholds = {name: float(getattr(self.config, name)) for name in names}
        if not self.config.eye_enable_unvalidated_quality_gates:
            for name in (
                "eye_max_gaze_mad",
                "eye_max_repeat_relative_error",
                "eye_max_head_rotation_degrees",
                "eye_landmark_noise_floor_pixels",
                "eye_min_valid_fraction_per_trial",
            ):
                thresholds.pop(name, None)
        if not self.config.eye_enable_unvalidated_warning_thresholds:
            for name in (
                "eye_max_repeat_relative_error",
                "eye_response_snr_threshold",
                "eye_directional_asymmetry_threshold",
                "eye_conjugacy_relative_error_threshold",
                "eye_rest_gaze_deviation_degrees_threshold",
            ):
                thresholds.pop(name, None)
        return thresholds

    def _finish_or_retry(
        self,
        stage: str,
        result: MotionResult,
        next_stage: str,
        screen: Any,
        ts: float,
    ) -> None:
        """根据检查质量进入下一阶段，或转入对应的可重试状态。"""

        self.stage_started_at = None
        if (
            stage == "eyes"
            and result.reason == "eye_metrics_recorded_for_validation"
        ):
            # 已完成且可测量的 E 记录属于 research-only：没有临床阈值时不
            # 生成自动阴性/阳性，也不应把整轮引导流程锁在无限重试中。
            self.stage = next_stage
            self._set_guidance(False, f"prepare_{next_stage}", ts)
            self._record_component_report_locked(self.STAGE_COMPONENTS[stage], ts)
            return
        if result.status == "insufficient":
            # 质量不足不会前进；保留结果用于报告，同时清空检测器准备重试。
            self.retry_counts[stage] += 1
            self.stage = f"retry_{stage}"
            screen.reset()
            self._set_guidance(False, result.reason, ts)
            self._record_component_report_locked(
                self.STAGE_COMPONENTS[stage], ts
            )
            return
        self.stage = next_stage
        self._set_guidance(False, f"prepare_{next_stage}", ts)
        self._record_component_report_locked(self.STAGE_COMPONENTS[stage], ts)

    def _update_face_guidance(
        self, ts: float, observation: FaceObservation | None
    ) -> None:
        """计算 E/F 引导，并仅在适用阶段覆盖当前引导。"""

        phase = self.face_screen.phase(ts) if self.stage == "face" else "neutral"
        value = face_guidance(self.stage, ts, observation, self.config, phase)
        if value is not None:
            self.guidance = value

    def _update_pose_guidance(
        self,
        ts: float,
        keypoints: Sequence[Mapping[str, float]],
        pose: str,
    ) -> None:
        """计算 A/B 引导，并仅在适用阶段覆盖当前引导。"""

        value = pose_guidance(self.stage, ts, keypoints, pose, self.config)
        if value is not None:
            self.guidance = value

    def _set_guidance(self, ready: bool, reason: str, ts: float) -> None:
        """用当前阶段写入一条不带测量指标的引导状态。"""

        self.guidance = guidance_value(self.stage, ready, reason, ts)

    def _screens(self) -> dict[str, Any]:
        """返回阶段名到检测器实例的映射。"""

        return {
            "eyes": self.eye_screen,
            "face": self.face_screen,
            "arms": self.arm_screen,
            "balance": self.balance_screen,
        }

    def _stage_binding(self, stage: str) -> tuple[str, Any]:
        """返回某阶段对应的结果属性名和检测器实例。"""

        result_attrs = {
            "eyes": "eye_result",
            "face": "face_result",
            "arms": "arm_result",
            "balance": "balance_result",
        }
        return result_attrs[stage], self._screens()[stage]

    def _active_duration(self) -> float:
        """返回当前自动阶段总时长；等待/汇总阶段返回零。"""

        if self.stage not in ACTIVE_STAGES:
            return 0.0
        return float(self._screens()[self.stage].duration_seconds)

    def _ensure_screening_locked(self, ts: float, source: str, reason: str) -> None:
        """必要时建立筛查上下文；调用方必须已经持有会话锁。"""

        if self.mode == "screening":
            return
        self._clear_screen_locked()
        self.mode = "screening"
        self.screening_started_at = ts
        self.trigger = {
            "source": source,
            "reason": reason,
            "triggered_at": round(ts, 4),
        }

    def _record_component_report_locked(self, code: str, ts: float, kind: str = "acquisition") -> None:
        """把一次已经结束的单项结果加入历史，并设为当前报告。"""

        items = build_report_items(
            self.manual,
            self.manual_complete,
            self.eye_result,
            self.face_result,
            self.arm_result,
            self.balance_result,
            self.speech_result,
            manual_completed=self.manual_completed,
            reported_functional_problems=self.reported_functional_problems,
            symptom_evidence=self.symptom_evidence,
        )
        item = items[code]
        if kind == "acquisition" and item["status"] in {"pending", "checking"}:
            self.current_report = None
            return
        if kind == "acquisition":
            self.attempt_counts[code] += 1
        status = str(item["status"])
        if item.get("decision_eligible", True) is False:
            decision = "research_only"
        else:
            decision = screening_assessment({code: item}, self.component_onsets)["decision"]
        previous = next((r for r in reversed(self.reports) if r["component"] == code), None)
        assessment = screening_assessment({code: item}, self.component_onsets)
        onsets = [e["new_or_sudden"] for e in item.get("symptom_evidence", []) if e.get("active", True)]
        if item.get("measurement_status", item["status"]) == "positive":
            onsets.append(self.component_onsets.get(code))
        onset_summary = (True if True in onsets else None if None in onsets else False) if onsets else self.component_onsets.get(code)
        report = {
            "session_id": self.session_id,
            "kind": kind,
            "revision": 1 if previous is None else previous["revision"] + 1,
            "supersedes": None if previous is None else previous["id"],
            "urgency": assessment["urgency"],
            "id": len(self.reports) + 1,
            "component": code,
            "attempt": self.attempt_counts[code],
            "completed_at": round(float(ts), 4),
            "decision": decision,
            "completeness": screening_assessment({code: item}, self.component_onsets)["completeness"],
            "item": {
                **item,
                "metrics": dict(item.get("metrics", {})),
                "details": dict(item.get("details", {})),
            },
            "new_or_sudden": onset_summary,
            "onset_time": self.component_onset_times.get(code),
        }
        report = deepcopy(report)
        self.reports.append(report)
        self.current_report = report

    def report_functional_problem(self, code: str, new_or_sudden: bool | None) -> None:
        """Record an explicit symptom report without relabeling sensor failure.

        Technical task failure alone never implies a reported symptom. Reports
        persist across retries until the screening session is reset.
        """
        code = str(code).strip().upper()
        if code not in {"F", "A", "S"}:
            raise ValueError("functional report component must be F, A, or S")
        if new_or_sudden is not None and not isinstance(new_or_sudden, bool):
            raise ValueError("new_or_sudden must be boolean or null")
        with self.lock:
            self._ensure_screening_locked(time.time(), "symptom_report", "reported_functional_problem")
            self._append_symptom_locked(code, new_or_sudden, time.time())
            self._refresh_evidence_report_locked(code, "symptom_update")

    def _append_symptom_locked(self, code: str, onset: bool | None, ts: float) -> None:
        self.reported_functional_problems[code] = True
        self.symptom_evidence.setdefault(code, []).append({
            "event_id": uuid4().hex, "source": "user_or_caregiver",
            "reported_at": ts, "new_or_sudden": onset, "active": True,
        })

    def _refresh_evidence_report_locked(self, code: str, kind: str) -> None:
        current = self.current_report
        self._record_component_report_locked(code, time.time(), kind)
        # Record every update without covering an in-progress acquisition UI.
        if current is None or current["component"] != code:
            self.current_report = current

    def retract_symptom(self, event_id: str, reason: str) -> None:
        """Explicitly correct one report, retaining its original evidence and audit."""
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("a correction reason is required")
        with self.lock:
            for code, events in self.symptom_evidence.items():
                event = next((e for e in events if e["event_id"] == event_id), None)
                if event is None:
                    continue
                if not event["active"]:
                    raise ValueError("symptom report already retracted")
                event.update(active=False, retracted_at=time.time(), correction_reason=reason.strip())
                self.reported_functional_problems[code] = any(e["active"] for e in events)
                if code in {"B", "E"} and not self.reported_functional_problems[code]:
                    self.manual["balance_problem" if code == "B" else "eye_problem"] = None
                    self.manual_completed[code] = False
                    self.manual_complete = self.manual_completed["B"]
                self._refresh_evidence_report_locked(code, "symptom_correction")
                return
            raise ValueError("symptom event not found in current session")

    def set_component_onset(
        self,
        code: str,
        new_or_sudden: bool,
        onset_time: str | None = None,
    ) -> None:
        """Attach onset to one component so mixed chronic/acute signs are representable."""

        normalized = str(code).strip().upper()
        if normalized not in {"B", "E", "F", "A", "S"}:
            raise ValueError("component must be B, E, F, A, or S")
        if not isinstance(new_or_sudden, bool):
            raise ValueError("new_or_sudden must be a boolean")
        with self.lock:
            self.component_onsets[normalized] = new_or_sudden
            self.component_onset_times[normalized] = (
                str(onset_time).strip() if onset_time else None
            )
            # Preserve the legacy fields for existing UI/history consumers.
            self.new_or_sudden = new_or_sudden
            self.onset_time = self.component_onset_times[normalized]
            if self.mode == "screening":
                self._refresh_evidence_report_locked(normalized, "onset_update")

    @staticmethod
    def _copy_report(report: Mapping[str, Any]) -> dict[str, Any]:
        """复制报告中的嵌套结构，防止 API 调用方修改会话数据。"""

        return deepcopy(dict(report))
