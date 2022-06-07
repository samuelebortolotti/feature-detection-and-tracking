## MAIN PDF

MAIN := main

## DIRECTORIES

BUILD_FOLDER := ./build
CHAPTER_FOLDER := ./
IMG_FOLDER := ./images
BIBLIOGRAPY_FOLDER := ./bibliography

## FILES

TEX_FILES := $(shell find $(CHAPTER_FOLDER) -name '*.tex' -or -name '*.sty' -or -name '*.cls')
BIB_FILES := $(shell find $(BIBLIOGRAPY_FOLDER) -name '*.bib')
IMG_FILES := $(shell find $(IMG_FOLDER) -path '*.jpg' -or -path '*.png'-or \( \! -path '$(BUILD_FOLDER)/*.pdf' -path '*.pdf' \) )

## COMPILER AND FLAGS

PDFLATEX := pdflatex
PDFLATEX_FLAGS := -halt-on-error -output-directory $(BUILD_FOLDER)
BIBTEX := bibtex
BIBTEX_FLAGS := --output_directory=$(BUILD_FOLDER)

## COMMANDS

MKDIR := mkdir -p
RM_FOLDER := rm -r -f
RM := rm -f
ECHO := echo -e
PDFVIEWER = zathura

## TARGETS

.PHONY: all build createBuild help open clean cleanall cleanaux cleanlog cleantoc

#.SILENT:

all: build clean
	@$(ECHO) 'Clean build done'

build: createBuild $(BUILD_FOLDER)/$(MAIN).pdf
	@$(ECHO) 'Build done'

createBuild:
	@$(MKDIR) $(BUILD_FOLDER)

open: build
	(${PDFVIEWER} $(BUILD_FOLDER)/$(MAIN).pdf &)

help:
	@$(ECHO) "Makefile help:\n \
	* build  : compiles the $(MAIN) LaTeX document. The curresponding pdf will be available in the $(BUILD_FOLDER) folder\n \
	* open  : compiles the $(MAIN) LaTeX document using the build target listed before. It also opens it with Zathura document viewer\n \
	* clean  : removes all files created during the document compilation in the $(BUILD_FOLDER) folder, except the resulting pdf (if present)\n \
	* cleanall  : removes the $(BUILD_FOLDER) folder and all the files inside it\n \
	* all  : [default target] compiles the $(MAIN) LaTeX document and and keeps only the pdf among the files generated during the compilation (build + clean)"

cleanaux:
	@$(RM) $(BUILD_FOLDER)/*.aux

cleanlog:
	@$(RM) $(BUILD_FOLDER)/*.log

cleantoc:
	@$(RM) $(BUILD_FOLDER)/*.toc

cleanall:
	@$(RM_FOLDER) $(BUILD_FOLDER)
	@$(ECHO) "$(BUILD_FOLDER) folder removed"

clean: cleanaux cleanlog cleantoc
	@$(ECHO) "$(BUILD_FOLDER) folder cleaned"

$(BUILD_FOLDER)/$(MAIN).aux: $(TEX_FILES) $(IMG_FILES)
	@$(PDFLATEX) $(PDFLATEX_FLAGS) $(MAIN)

$(BUILD_FOLDER)/$(MAIN).bbl: $(BIB_FILES) | $(BUILD_FOLDER)/$(MAIN).aux
	@$(BIBTEX) $(BIBTEX_FLAGS) $(BUILD_FOLDER)/$(MAIN)
	@$(PDFLATEX) $(PDFLATEX_FLAGS) $(MAIN)
	
$(BUILD_FOLDER)/$(MAIN).pdf: $(BUILD_FOLDER)/$(MAIN).aux $(if $(BIB_FILES), $(BUILD_FOLDER)/$(MAIN).bbl)
	@$(PDFLATEX) $(PDFLATEX_FLAGS) $(MAIN)