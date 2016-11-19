# make -f Makefile_pdflatex

R = yoshinari_notes_vec_derivative

pdf: $(R).pdf

$(R).pdf: $(R).tex $(R).bbl
	latex $(R)
	bibtex $(R)
	latex $(R)
	latex $(R)
	dvipdfmx $(R).dvi

$(R).bbl: $(R).tex $(R).bib
	latex $(R)
	bibtex $(R)

clean:
	rm -f $(R).dvi $(R).log $(R).aux $(R).bbl $(R).blg $(R).out *.log
