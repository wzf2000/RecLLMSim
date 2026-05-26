# LaTeX Chinese Prompt Box Diagnosis

## Context

This note records whether the current ACL paper draft can include Chinese prompt templates inside boxed LaTeX environments.
The checked draft is under `detection/assets/paper-draft/`.

## Existing Draft Setup

The main file `detection/assets/paper-draft/latex/acl_latex.tex` uses the standard ACL `pdflatex`-oriented setup:

- `\usepackage[review]{acl}`
- `\usepackage{times}`
- `\usepackage[T1]{fontenc}`
- `\usepackage[utf8]{inputenc}`
- `\usepackage{microtype}`
- `\usepackage{inconsolata}`
- table and math packages including `booktabs`, `multirow`, `makecell`, `threeparttable`, `amsmath`, and `amssymb`

It did not originally include Chinese support or `tcolorbox`.

## Environment Issues Found

Initial compilation failed before reaching any Chinese text because several TeX Live packages were missing:

- `helvetic`, required by the ACL style through the `phvb` font metric.
- `inconsolata`, required by the current preamble.
- `todonotes`, required by the current preamble.
- `multirow`, `makecell`, and `threeparttable`, required by the current tables.
- `tcolorbox`, required for prompt boxes.

These packages were installed with `tlmgr install`.
The successful install commands were:

```bash
tlmgr install helvetic tcolorbox
tlmgr install inconsolata
tlmgr install todonotes
tlmgr install multirow makecell threeparttable
```

## Compilation Command

Because the main `.tex` file is under `latex/` but imports `Headings/`, `Tex/`, and `Appendix/` relative to `detection/assets/paper-draft/`, compilation should be launched from `detection/assets/paper-draft/`.
The `latex/` folder must be added to TeX, BibTeX, and BST search paths:

```bash
TEXMFVAR=/tmp/texmf-var TEXINPUTS=latex//: BIBINPUTS=latex//: BSTINPUTS=latex//: latexmk -pdf -interaction=nonstopmode -halt-on-error latex/acl_latex.tex
```

The `TEXMFVAR=/tmp/texmf-var` setting avoids writing generated TeX font/cache files under the user home directory in sandboxed environments.

## Chinese Prompt Box Result

The lightweight `pdflatex + CJKutf8 + tcolorbox` route was tested successfully.
The key point is to load plain `tcolorbox`, not `tcolorbox[most]`, because `[most]` pulls in extra optional TikZ libraries such as `tikzfill.image.sty`.

Recommended preamble additions:

```tex
\usepackage{CJKutf8}
\usepackage{tcolorbox}
```

Recommended environment pattern:

```tex
\newenvironment{promptbox}[1]{%
  \begin{CJK*}{UTF8}{gbsn}%
  \begin{tcolorbox}[colback=gray!5,colframe=black!40,title={#1},fonttitle=\bfseries]%
}{%
  \end{tcolorbox}%
  \end{CJK*}%
}
```

Example usage:

```tex
\begin{promptbox}{Memory Construction Prompt}
你是一名用户满意度评估助手。
请根据用户画像、历史对话和当前回复，判断本轮满意度。
\end{promptbox}
```

## XeLaTeX Route

XeLaTeX is not recommended for this draft unless the submission environment is explicitly changed.
The current machine does not expose Chinese system fonts through `fontconfig`, and common font names such as `Times New Roman`, `TeX Gyre Termes`, and `Noto Serif CJK SC` were not directly matched by XeLaTeX.
TeX Live does include Fandol CJK fonts, but using them reliably would require additional font configuration and would move the paper away from the standard ACL `pdflatex` path.

## Recommendation

Use the current ACL `pdflatex` workflow and add Chinese prompt boxes with `CJKutf8` plus lightweight `tcolorbox`.
Avoid `tcolorbox[most]` unless additional optional TikZ packages are installed and there is a clear need for those features.
