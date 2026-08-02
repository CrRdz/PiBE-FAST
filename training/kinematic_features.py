"""Features for the public post-stroke functional-movement dataset.

The source files contain 19 joint-angle channels sampled at 125 Hz.  Feature
names deliberately use channel indices because MATLAB string objects in the
published v5 files are not decoded consistently outside MATLAB.  The channel
order is nevertheless fixed by the dataset schema.
"""

from __future__ import annotations

import numpy as np


ANGLE_CHANNELS = 19
TASKS = 6

KINEMATIC_FEATURES = (
    "duration_seconds",
    *(
        name
        for channel in range(1, ANGLE_CHANNELS + 1)
        for name in (
            f"angle_{channel:02d}_range",
            f"angle_{channel:02d}_std",
            f"angle_{channel:02d}_speed_p90",
        )
    ),
    *(f"task_{task:02d}" for task in range(1, TASKS + 1)),
)


def summarize_angle_segment(
    angles: np.ndarray,
    *,
    sample_rate: float,
    task_index: int,
    minimum_finite_fraction: float = 0.5,
) -> dict[str, float]:
    """Summarize one event-bounded repetition using side-invariant features."""

    values = np.asarray(angles, dtype=float)
    if values.ndim != 2 or values.shape[0] != ANGLE_CHANNELS:
        raise ValueError(f"angles must have shape ({ANGLE_CHANNELS}, frames)")
    if values.shape[1] < 10:
        raise ValueError("angle segment must contain at least 10 frames")
    if sample_rate <= 0.0:
        raise ValueError("sample rate must be positive")
    if not 1 <= int(task_index) <= TASKS:
        raise ValueError(f"task index must be in [1, {TASKS}]")

    output = {"duration_seconds": values.shape[1] / float(sample_rate)}
    for index, channel in enumerate(values, 1):
        finite = np.isfinite(channel)
        if finite.mean() < minimum_finite_fraction:
            raise ValueError(f"angle channel {index} has too many missing values")
        valid = channel[finite]
        adjacent = finite[1:] & finite[:-1]
        speeds = np.abs(np.diff(channel)[adjacent]) * float(sample_rate)
        if not len(speeds):
            raise ValueError(f"angle channel {index} has no finite velocity samples")
        output[f"angle_{index:02d}_range"] = float(
            np.percentile(valid, 95) - np.percentile(valid, 5)
        )
        output[f"angle_{index:02d}_std"] = float(np.std(valid))
        output[f"angle_{index:02d}_speed_p90"] = float(np.percentile(speeds, 90))
    output.update(
        {
            f"task_{task:02d}": float(task == int(task_index))
            for task in range(1, TASKS + 1)
        }
    )
    if set(output) != set(KINEMATIC_FEATURES):
        raise RuntimeError("kinematic feature schema drift")
    return output
