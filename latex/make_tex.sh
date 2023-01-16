pdflatex -interaction nonstopmode final_format.tex
pdflatex -interaction nonstopmode final_format.tex
bibtex final_format.aux
pdflatex -interaction nonstopmode final_format.tex
qpdf final_format.pdf --pages . 1-12 -- main.pdf
qpdf final_format.pdf --pages . 13-z -- supplement.pdf
