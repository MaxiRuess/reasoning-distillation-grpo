#!/bin/sh
set -eu
mkdir -p output/paper
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/output/paper/mpl-cache}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python scripts/21_inference_extension_report.py --proposal results/proposals/inference-extension/f1b156f14754100c186bc396
pdflatex -interaction=nonstopmode -halt-on-error -output-directory output/paper paper/paper.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory output/paper paper/paper.tex
cp output/paper/paper.pdf assets/paper.pdf
