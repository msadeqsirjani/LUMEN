#!/bin/bash
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode -halt-on-error main.tex > /dev/null 2>&1 && \
bibtex main > /dev/null 2>&1 && \
pdflatex -interaction=nonstopmode -halt-on-error main.tex > /dev/null 2>&1 && \
pdflatex -interaction=nonstopmode -halt-on-error main.tex > /dev/null 2>&1
rm -f main.aux main.bbl main.blg main.log main.out main.fls main.fdb_latexmk
