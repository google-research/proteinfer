pdflatex -interaction nonstopmode final_format.tex
pdflatex -interaction nonstopmode final_format.tex
bibtex final_format.aux
pdflatex -interaction nonstopmode final_format.tex
qpdf final_format.pdf --pages . 1-10 -- main.pdf
qpdf final_format.pdf --pages . 11-z -- supplement.pdf
