# Feature Detection and Tracking Presentation

In order to generate the presentation slides, be sure you have [pandoc](https://pandoc.org/) and [LaTeX](https://www.latex-project.org/) installed.

To check the requirements you can run the following lines (the output will depend on the version you have installed):

```sh
pandoc --version
```

```sh
pdflatex --version
```

## Generate the slides

To generate the beamer presentation run the following command:

```sh
pandoc main.md --include-in-header=preamble.tex \
--citeproc --bibliography=bibliography.bib -t \
beamer -o main.pdf
```

Once the slides have been generated, you can open them with your favourite document viewer (for instance [Zathura](https://pwmt.org/projects/zathura/installation/))

```sh
zathura main.pdf
```
