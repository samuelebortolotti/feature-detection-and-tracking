# ======= FOLDERS ==================
VENV := venv
PROJECT_NAME := fdt

# ======= PROGRAMS AND FLAGS =======
PYTHON := python3
PYFLAGS := -m
PIP := pip
UPGRADE_PIP := --upgrade pip

# ======= MAIN =====================
MAIN := fdt
MAIN_FLAGS :=
PIP := pip

# ======= SIFT =====================
SIFT := sift 
SIFT_IMAGE := material/test/calchera.jpg
SIFT_FLAGS := --n-features 150

# ======= ORB ======================
ORB := orb 
ORB_IMAGE := material/test/calchera.jpg
ORB_FLAGS := --n-features 150

# ======= HARRIS ===================
HARRIS := harris 
HARRIS_IMAGE := material/test/calchera.jpg
HARRIS_FLAGS := --config-file

# ======= BLOB =====================
BLOB := blob 
BLOB_IMAGE := material/test/Lenna.png
BLOB_FLAGS := --config-file

# ======= MATCHER ==================
MATCHER := matcher
MATCHER_METHOD := sift
MATCHER_FLAGS := --n-features 150 --flann --matching-distance 60 --video material/Contesto_industriale1.mp4 --frame-update 30

# ======= KALMAN ==================
KALMAN := kalman
KALMAN_METHOD := orb
KALMAN_FLAGS := --n-features 100 --flann --matching-distance 40 --video material/Contesto_industriale1.mp4 --frame-update 50

# ======= KALMAN ==================
LUCAS_KANADE := lucas-kanade
LUCAS_KANADE_METHOD := sift
LUCAS_KANADE_FLAGS := --n-features 100 --video material/Contesto_industriale1.mp4 --frame-update 50

# ======= FORMAT ===================
FORMAT := black
FORMAT_FLAG := fdt

# ======= DOC ======================
AUTHORS := --author "Samuele Bortolotti" 
VERSION :=-r 0.1 
LANGUAGE := --language en
SPHINX_EXTENSIONS := --extensions sphinx.ext.autodoc --extensions sphinx.ext.napoleon --extensions sphinx.ext.viewcode --extensions myst_parser
DOC_FOLDER := docs

## Quickstart
SPHINX_QUICKSTART := sphinx-quickstart
SPHINX_QUICKSTART_FLAGS := --sep --no-batchfile --project feature-detection-and-tracking $(AUTHORS) $(VERSION) $(LANGUAGE) $(SPHINX_EXTENSIONS)

## Build
BUILDER := html
SPHINX_BUILD := make $(BUILDER)
SPHINX_API_DOC := sphinx-apidoc
SPHINX_API_DOC_FLAGS := -P -o $(DOC_FOLDER)/source .
SPHINX_THEME = sphinx_rtd_theme
DOC_INDEX := index.html

## INDEX.rst preamble
define INDEX

.. ffdnet documentation master file, created by
   sphinx-quickstart on Fri May 10 23:38:46 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. include:: ../../README.md
	 :parser: myst_parser.sphinx_

.. toctree::
   :maxdepth: 5
   :caption: Contents:

   modules


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

endef

export INDEX

# ======= COLORS ===================
RED := \033[31m
GREEN := \033[32m
YELLOW := \033[33m
BLUE := \033[34m
NONE := \033[0m

# ======= COMMANDS =================
ECHO := echo -e
MKDIR := mkdir -p
OPEN := xdg-open
SED := sed
	
# RULES
.PHONY: help env install install-dev sift orb harris blob matcher kalman lucas-kanade doc doc-layout open-doc format-code

help:
	@$(ECHO) '$(YELLOW)Makefile help$(NONE)'
	@$(ECHO) " \
	* env 			: generates the virtual environment using the current python version and venv\n \
	* install		: install the requirements listed in requirements.txt\n \
	* install-dev		: install the development requirements listed in requirements.dev.txt\n \
	* sift			: run the SIFT feature detector on the image passed as parameter\n \
	* orb			: run the ORB feature detector on the image passed as parameter\n \
	* harris			: run the Harris corner detector on the image passed as parameter\n \
	* blob			: run the blob feature detector on the image passed as parameter\n \
	* matcher		: run either the brute force matcher or the FLANN matcher on a video\n \
	* kalman		: run the Kalman filter to track the feature on a video\n \
	* lucas-kanade		: run the Lucas-Kanade optical flow to track the feature on a video\n \
	* doc-layout 		: generates the Sphinx documentation layout\n \
	* doc 			: generates the documentation (requires an existing documentation layout)\n \
	* open-doc 		: opens the documentation\n"

env:
	@$(ECHO) '$(GREEN)Creating the virtual environment..$(NONE)'
	@$(MKDIR) $(VENV)
	@$(eval PYTHON_VERSION=$(shell $(PYTHON) --version | tr -d '[:space:]' | tr '[:upper:]' '[:lower:]' | cut -f1,2 -d'.'))
	@$(PYTHON_VERSION) -m venv $(VENV)/$(PROJECT_NAME)
	@$(ECHO) '$(GREEN)Done$(NONE)'

install:
	@$(ECHO) '$(GREEN)Installing requirements..$(NONE)'
	@$(PYTHON) -m pip install $(UPGRADE_PIP)
	@pip install -r requirements.txt
	@$(ECHO) '$(GREEN)Done$(NONE)'

install-dev:
	@$(ECHO) '$(GREEN)Installing requirements..$(NONE)'
	@$(PYTHON) -m pip install $(UPGRADE_PIP)
	@$(PIP) install -r requirements.dev.txt
	@$(ECHO) '$(GREEN)Done$(NONE)'

sift:
	@$(ECHO) '$(BLUE)Running SIFT on an image ..$(NONE)'
	@$(PYTHON) $(PYFLAGS) $(MAIN) $(MAIN_FLAGS) $(SIFT) $(SIFT_IMAGE) $(SIFT_FLAGS)
	@$(ECHO) '$(BLUE)Done$(NONE)

orb:
	@$(ECHO) '$(BLUE)Running ORB on an image ..$(NONE)'
	@$(PYTHON) $(PYFLAGS) $(MAIN) $(MAIN_FLAGS) $(ORB) $(ORB_IMAGE) $(ORB_FLAGS)
	@$(ECHO) '$(BLUE)Done$(NONE)

harris:
	@$(ECHO) '$(BLUE)Running Harris corner detector on an image ..$(NONE)'
	@$(PYTHON) $(PYFLAGS) $(MAIN) $(MAIN_FLAGS) $(HARRIS) $(HARRIS_IMAGE) $(HARRIS_FLAGS)
	@$(ECHO) '$(BLUE)Done$(NONE)

blob:
	@$(ECHO) '$(BLUE)Running the blob detector on an image ..$(NONE)'
	@$(PYTHON) $(PYFLAGS) $(MAIN) $(MAIN_FLAGS) $(BLOB) $(BLOB_IMAGE) $(BLOB_FLAGS)
	@$(ECHO) '$(BLUE)Done$(NONE)

matcher:
	@$(ECHO) '$(BLUE)Running the keypoint matcher ..$(NONE)'
	@$(PYTHON) $(PYFLAGS) $(MAIN) $(MAIN_FLAGS) $(MATCHER) $(MATCHER_METHOD) $(MATCHER_FLAGS)
	@$(ECHO) '$(BLUE)Done$(NONE)

kalman:
	@$(ECHO) '$(BLUE)Running the feature tracking employing a feature detector and Kalman filters ..$(NONE)'
	@$(PYTHON) $(PYFLAGS) $(MAIN) $(MAIN_FLAGS) $(KALMAN) $(KALMAN_METHOD) $(KALMAN_FLAGS)
	@$(ECHO) '$(BLUE)Done$(NONE)

lucas-kanade:
	@$(ECHO) '$(BLUE)Running the feature tracking employing a feature detector and Lucas-Kanade optical flow ..$(NONE)'
	@$(PYTHON) $(PYFLAGS) $(MAIN) $(MAIN_FLAGS) $(LUCAS_KANADE) $(LUCAS_KANADE_METHOD) $(LUCAS_KANADE_FLAGS)
	@$(ECHO) '$(BLUE)Done$(NONE)

doc-layout:
	@$(ECHO) '$(BLUE)Generating the Sphinx layout..$(NONE)'
	# Sphinx quickstart
	$(SPHINX_QUICKSTART) $(DOC_FOLDER) $(SPHINX_QUICKSTART_FLAGS)
	# Including the path for the current README.md
	@$(ECHO) "\nimport os\nimport sys\nsys.path.insert(0, os.path.abspath('../..'))">> $(DOC_FOLDER)/source/conf.py
	# Inserting custom index.rst header
	@$(ECHO) "$$INDEX" > $(DOC_FOLDER)/source/index.rst
	# Sphinx theme
	@$(SED) -i -e "s/html_theme = 'alabaster'/html_theme = '$(SPHINX_THEME)'/g" $(DOC_FOLDER)/source/conf.py 
	@$(ECHO) '$(BLUE)Done$(NONE)'

doc:
	@$(ECHO) '$(BLUE)Generating the documentation..$(NONE)'
	$(SPHINX_API_DOC) $(SPHINX_API_DOC_FLAGS)
	cd $(DOC_FOLDER); $(SPHINX_BUILD)
	@$(ECHO) '$(BLUE)Done$(NONE)'

open-doc:
	@$(ECHO) '$(BLUE)Open documentation..$(NONE)'
	$(OPEN) $(DOC_FOLDER)/build/$(BUILDER)/$(DOC_INDEX)
	@$(ECHO) '$(BLUE)Done$(NONE)'

format-code:
	@$(ECHO) '$(BLUE)Formatting the code..$(NONE)'
	@$(FORMAT) $(FORMAT_FLAG)
	@$(ECHO) '$(BLUE)Done$(NONE)'
