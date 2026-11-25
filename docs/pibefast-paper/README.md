# Paper

**Multimodal Embodied AI for real-time healthcare monitoring**

- [Final paper](pibefast-conference-paper.pdf)
- [Editorial changes in red](pibefast-conference-paper-highlight.pdf)
- [LaTeX source](main.tex)
- [Data index](../../research_data/FINAL_DATA_TABLES.md)

This version uses the supplied id255 final manuscript, with direct wording for the system design, evaluation results and future work.

## Build

```bash
cd docs/pibefast-paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex main-highlight.tex
```

The final and marked PDFs are generated as `build/main.pdf` and `build/main-highlight.pdf`. Figures are in `picture/`.
