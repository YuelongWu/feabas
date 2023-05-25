# FEABAS

FEABAS (Finite-Element Algorithmic Brain Assembly System) is a Python library for stitching & alignment of connectome datasets using finite-element analysis.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Installation

We used Python 3.10.4 to develop and test the package, but the codebase should be compatible with Python 3.8+. To install FEABAS, you can clone the repository and [pip](https://pip.pypa.io/en/stable/)-install it into a proper virtual environment:

```bash
git clone https://github.com/YuelongWu/feabas.git
cd feabas
pip install -e .
```

Alternatively, you can directly install FEABAS from [PyPI](https://pypi.org/project/feabas/):

```bash
pip install feabas
```

## Usage

The user needs to first create a dedicated *working directory* for each dataset that will go through FEABAS stitching and alignment pipelines. The *working directory* acts as a self-contained environment for individual datasets, defines project-specific configurations, and saves intermediate checkpoints/results generated through the workflow. At the beginning of the process, the *working directory* is expected to have the following file structure:

```
(working directory)
 ├── configs
 │   ├── stitching_configs.yaml (optional)
 │   ├── thumbnail_configs.yaml (optional)
 │   ├── alignment_configs.yaml (optional)
 │   └── material_table.json (optional)
 ├── stitch
 │   └── stitch_coord
 │       ├── (section_name_0).txt
 │       ├── (section_name_1).txt
 │       ├── (section_name_2).txt
 │       └── ...
 └── section_order.txt (optional)
```

The `configs` folder in the *working directory* contains project-specific configuration files that override the default settings. If any of these files don't exist, FEABAS will use the corresponding default configuration files in the `configs` folder under the repository root directory (NOT the *working directory*) with the same file names but prefixed by `default_`, e.g. `default_stitching_configs.yaml`.

The .txt files in the `stitch\stitch_coord` folder specify the information about the tile arrangement for each section and serve as the inputs to the stitcher code of FEABAS.

The filenames of the stitch coordinate text files define the name of the sections. By default, FEABAS assumes the order of sections in the final aligned stack can be reconstructed by sorting the section name alphabetically. If that's not the case, the user can define the right section order by providing an optional `section_order.txt` file directly under the working directory. In the file, each line is a section name corresponding to the stitch coordinate filenames (without `.txt` extension), and their positions in the file define their position in the aligned stack.