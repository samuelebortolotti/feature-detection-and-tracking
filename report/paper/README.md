# Report

In order to generate the report pdf be sure you have and [LaTeX](https://www.latex-project.org/) installed.

To check the requirements you can run the following lines (the output will depend on the version you have installed):

```sh
pdflatex --version
```

You can access the report by opening the pdf file named [`main.pdf`](./main.pdf) or you can compile the LaTeX file and obtain the pdf file by typing the following commands

```bash
pdflatex main.tex
```

Or you can employ the `Makefile` by typing:

```bash
make
```
