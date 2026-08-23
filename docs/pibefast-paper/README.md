# PiBE-FAST conference paper

The latest compiled English manuscript is available as
[`pibefast-conference-paper.pdf`](pibefast-conference-paper.pdf).

The paper is organized as one LaTeX file per top-level section. `main.tex`
contains only the IEEE preamble and ordered `\input` statements.

```text
pibefast-paper/
├── main.tex
├── IEEEtran.cls
├── references.bib
└── sections/
    ├── frontmatter.tex
    ├── abstract.tex
    ├── introduction.tex
    ├── related-work.tex
    ├── system-overview.tex
    ├── multimodal-methods.tex
    ├── fusion.tex
    ├── experiments.tex
    ├── discussion-conclusion.tex
    └── references.tex
```

Compile from this directory:

```bash
latexmk main.tex
```

The local `.latexmkrc` writes the PDF and all intermediate files to `build/`.
That directory is ignored by Git. The compiled paper is therefore available at
`build/main.pdf` without adding generated files to the source directory.
