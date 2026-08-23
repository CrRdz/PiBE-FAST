#!/usr/bin/env python3
"""Generate the English PiBE-FAST fusion architecture figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


HERE = Path(__file__).resolve().parent


PALETTE = {
    "ink": "#17212B",
    "muted": "#52606D",
    "line": "#8293A5",
    "sensor": "#E8F1FA",
    "sensor_edge": "#3776A8",
    "gate": "#E8F5EE",
    "gate_edge": "#33805C",
    "vector": "#FFF2D8",
    "vector_edge": "#B97816",
    "research": "#F2EEFA",
    "research_edge": "#7556A5",
    "safety": "#FDECEC",
    "safety_edge": "#B44343",
    "future": "#F7F8FA",
}


TEXT = {
    "en": {
        "modalities": [
            ("B  Balance (2)", "trunk-orientation change\nML-sway change"),
            (
                "E  Eyes (6)",
                "left gaze range\nright gaze range\ninter-eye range asym.\n"
                "directional asym.\nmax conjugacy error\nrest-gaze deviation",
            ),
            ("F  Face (3)", "mouth-corner $\\Delta$\nsmile-score $\\Delta$\nsmile strength"),
            ("A  Arms (4)", "arm-level $\\Delta$\nleft drop\nright drop\ndrift $\\Delta$"),
            ("S  Speech (4)", "character error rate\ncharacters/s\npause fraction\nMDSC probability"),
        ],
        "completed": "Completed automatic measurements",
        "raw": "Raw feature subsets\n$x_B,\u2026,x_S$  (19 total)",
        "gate": "Quality gate per modality\n$\\tilde{x}_m=q_m x_m$",
        "mask": "retain $q_B,\u2026,q_S$ and\nmissing masks $r_B,\u2026,r_S$",
        "vector": "Research representation\n$z$",
        "research": "Research summaries",
        "future": "Future calibrated model\n$p(\\mathrm{stroke}\mid z)$",
        "future_req": "NOT IMPLEMENTED\nlabeled subjects \u00b7 calibration \u00b7 external evaluation",
        "status": "Modality-specific\nB/E/F/A/S statuses\n+ new/sudden onset",
        "urgency": "Rule-based urgency\nformal user-facing decision",
        "boundary": "Safety boundary: the research vector has no decision authority",
        "lane_research": "RESEARCH FUSION PATH",
        "lane_safety": "SAFETY DECISION PATH",
    },
}


def _font(language: str, size: float, weight: str = "normal") -> FontProperties:
    return FontProperties(family="DejaVu Sans", size=size, weight=weight)


def _box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    language: str,
    *,
    face: str,
    edge: str,
    size: float = 9.0,
    weight: str = "normal",
    linestyle: str = "solid",
) -> None:
    x, y = xy
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.015,rounding_size=0.025",
            linewidth=1.15,
            linestyle=linestyle,
            edgecolor=edge,
            facecolor=face,
        )
    )
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        color=PALETTE["ink"],
        fontproperties=_font(language, size, weight),
        linespacing=1.25,
    )


def _arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = "line",
    linestyle: str = "solid",
    mutation_scale: float = 11,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=mutation_scale,
            linewidth=1.15,
            linestyle=linestyle,
            color=PALETTE[color],
            shrinkA=0,
            shrinkB=0,
        )
    )


def _feature_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    title: str,
    details: str,
    language: str,
) -> None:
    x, y = xy
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.025",
            linewidth=0.95,
            edgecolor=PALETTE["sensor_edge"],
            facecolor=PALETTE["sensor"],
        )
    )
    ax.text(
        x + width / 2,
        y + height * 0.76,
        title,
        ha="center",
        va="center",
        color=PALETTE["ink"],
        fontproperties=_font(language, 6.4, "bold"),
    )
    ax.text(
        x + width / 2,
        y + height * 0.38,
        details,
        ha="center",
        va="center",
        color=PALETTE["ink"],
        fontproperties=_font(language, 5.9),
        linespacing=1.12,
    )


def draw(language: str) -> None:
    text = TEXT[language]
    fig, ax = plt.subplots(figsize=(13.4, 6.45))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 13.4)
    ax.set_ylim(0, 6.45)
    ax.axis("off")

    # Lane labels.
    ax.text(
        0.15,
        6.25,
        text["lane_research"],
        color=PALETTE["muted"],
        fontproperties=_font(language, 9.5, "bold"),
        va="top",
    )
    ax.text(
        0.15,
        1.43,
        text["lane_safety"],
        color=PALETTE["safety_edge"],
        fontproperties=_font(language, 9.5, "bold"),
        va="top",
    )

    # Five modality inputs.
    modality_y = 4.68
    modality_w = 2.28
    modality_h = 1.34
    modality_xs = [0.28, 2.92, 5.56, 8.20, 10.84]
    for x, (title, count) in zip(modality_xs, text["modalities"]):
        _box(
            ax,
            (x, modality_y),
            modality_w,
            modality_h,
            f"{title}\n{count}",
            language,
            face=PALETTE["sensor"],
            edge=PALETTE["sensor_edge"],
            size=7.2,
            weight="bold",
        )

    # Join modality boxes into the raw feature stage.
    join_y = 4.43
    for x in modality_xs:
        cx = x + modality_w / 2
        ax.plot([cx, cx], [modality_y, join_y], color=PALETTE["line"], lw=0.9)
    ax.plot(
        [modality_xs[0] + modality_w / 2, modality_xs[-1] + modality_w / 2],
        [join_y, join_y],
        color=PALETTE["line"],
        lw=0.9,
    )

    ax.text(
        6.70,
        4.36,
        text["completed"],
        ha="center",
        va="top",
        color=PALETTE["muted"],
        fontproperties=_font(language, 8.5),
    )
    _arrow(ax, (4.01, join_y), (4.01, 4.08))

    _box(
        ax,
        (2.86, 3.35),
        2.30,
        0.73,
        text["raw"],
        language,
        face=PALETTE["sensor"],
        edge=PALETTE["sensor_edge"],
        size=9.0,
    )
    _arrow(ax, (5.16, 3.715), (5.62, 3.715))
    _box(
        ax,
        (5.62, 3.35),
        2.25,
        0.73,
        text["gate"],
        language,
        face=PALETTE["gate"],
        edge=PALETTE["gate_edge"],
        size=9.0,
        weight="bold",
    )
    _box(
        ax,
        (5.70, 2.42),
        2.09,
        0.55,
        text["mask"],
        language,
        face="white",
        edge=PALETTE["gate_edge"],
        size=8.2,
    )
    _arrow(ax, (6.745, 2.97), (6.745, 3.35), color="gate_edge")

    _arrow(ax, (7.87, 3.715), (8.30, 3.715))
    _box(
        ax,
        (8.30, 3.20),
        2.48,
        0.98,
        text["vector"],
        language,
        face=PALETTE["vector"],
        edge=PALETTE["vector_edge"],
        size=8.3,
        weight="bold",
    )

    _arrow(ax, (10.78, 3.715), (11.15, 3.715))
    _box(
        ax,
        (11.15, 3.20),
        2.05,
        0.98,
        text["research"],
        language,
        face=PALETTE["research"],
        edge=PALETTE["research_edge"],
        size=7.4,
    )

    # Dashed future-only probability path.
    _arrow(
        ax,
        (9.54, 3.20),
        (9.54, 2.66),
        color="vector_edge",
        linestyle="dashed",
    )
    _arrow(
        ax,
        (9.54, 2.66),
        (10.97, 2.66),
        color="vector_edge",
        linestyle="dashed",
    )
    _box(
        ax,
        (10.97, 2.34),
        2.23,
        0.64,
        text["future"],
        language,
        face=PALETTE["future"],
        edge=PALETTE["vector_edge"],
        size=8.5,
        linestyle="dashed",
    )
    ax.text(
        12.08,
        2.22,
        text["future_req"],
        ha="center",
        va="top",
        color=PALETTE["muted"],
        fontproperties=_font(language, 7.2),
        linespacing=1.2,
    )

    # Explicitly separate the formal rule path.
    ax.plot([0.15, 13.2], [1.63, 1.63], color=PALETTE["safety_edge"], lw=1.0)
    ax.text(
        6.7,
        1.68,
        text["boundary"],
        ha="center",
        va="bottom",
        color=PALETTE["safety_edge"],
        fontproperties=_font(language, 8.5, "bold"),
    )
    _box(
        ax,
        (3.06, 0.40),
        2.45,
        0.72,
        text["status"],
        language,
        face=PALETTE["safety"],
        edge=PALETTE["safety_edge"],
        size=9.0,
    )
    _arrow(ax, (5.51, 0.76), (7.80, 0.76), color="safety_edge")
    _box(
        ax,
        (7.80, 0.40),
        2.55,
        0.72,
        text["urgency"],
        language,
        face=PALETTE["safety"],
        edge=PALETTE["safety_edge"],
        size=9.0,
        weight="bold",
    )

    output_stem = "fusion-architecture-en"
    for suffix, kwargs in (("pdf", {}), ("png", {"dpi": 220})):
        fig.savefig(
            HERE / f"{output_stem}.{suffix}",
            bbox_inches="tight",
            pad_inches=0.08,
            facecolor="white",
            **kwargs,
        )
    plt.close(fig)


def main() -> None:
    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.fontset": "dejavusans",
            "axes.unicode_minus": False,
        }
    )
    draw("en")


if __name__ == "__main__":
    main()
