# FEABAS

FEABAS (Finite-Element Analysis Brain Assembly System) is a Python library powered by finite-element analysis for stitching & alignment of serial-sectioning electron microscopy connectomic datasets.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Installation

We used Python 3.10.4 to develop and test the package, but the codebase should be compatible with Python 3.8+. To install FEABAS, you can clone the [GitHub repository](https://github.com/YuelongWu/feabas) and pip-install it into a proper virtual environment:

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

### Preparation

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

#### configuration files
The `configs` folder in the *working directory* contains project-specific configuration files that override the default settings. If any of these files don't exist, FEABAS will use the corresponding default configuration files in the `configs` folder under the repository root directory (NOT the *working directory*) with the same file names but prefixed by `default_`, e.g. `default_stitching_configs.yaml`. The user can copy these default configuration files to their *working directory* `configs` folder, remove the prefix in the filename, and adjust the file contents accordingly based on the specific needs of their dataset.

#### stitch coordinate files
The .txt files in the `stitch\stitch_coord` folder are user-created [TSV](https://en.wikipedia.org/wiki/Tab-separated_values) files specifying the approximate tile arrangement for each section. They are the inputs to the stitcher pipeline of FEABAS and usually can be derived from the metadata from the microscopy. In one coordinate file, it first defines some metadata info like the root directory of the images, the pixel resolution (in nanometers), and the size of each image tile (in pixels). Following the metadata is a table of all the image tiles associated with that section, with the first column giving the relative path of each image file relative to the root directory, and the second and the third column defining the x and y coordinates (in pixels) of the images. An example stitch coordinate text file looks like this:

<div><code><ins>s0001.txt</ins></code></div>

```
{ROOT_DIR}	/home/feabas/my_project/raw_data/s0001
{RESOLUTION}	4.0
{TILE_SIZE}	4096	4096
Tile_0001.tif	0	0
Tile_0002.tif	3686	0
Tile_0003.tif	7373	0
Tile_0004.tif	0	3686
Tile_0005.tif	3686	3686
Tile_0006.tif	3686	3686
```

It describes a section whose raw image tiles from the microscopy are saved under the directory `/home/feabas/my_project/raw_data/s0001`. It contains 6 images of size 4096x4096 pixels, arranged on a 2-rows-by-3-columns grid with 10% overlaps. Note that in general the images do not necessarily need to be arranged in a rectilinear pattern and the image files can have arbitrary names, as long as the coordinates are as accurate as possible. Also, make sure that the fields in the coordinate files are separated by Horizontal Tab `\t`, other delimiters are currently not supported.  

#### section order file (optional)
The filenames of the stitch coordinate text files define the name of the sections. By default, FEABAS assumes the order of sections in the final aligned stack can be reconstructed by sorting the section name alphabetically. If that's not the case, the user can define the right section order by providing an optional `section_order.txt` file directly under the working directory. In the file, each line is a section name corresponding to the stitch coordinate filenames (without `.txt` extension), and their positions in the file define their position in the aligned stack.

#### direct FEABAS to the current project
To enable FEABAS to identify the dataset it needs to process, the user needs to modify the `working_directory` field in the `configs/general_configs.yaml` file under FEABAS code root directory:

<div><code><ins>feabas/configs/general_configs.yaml</ins></code></div>

```yaml
working_directory: FULL_PATH_TO_THE_WORKING_DIRECTORY_OF_THE_CURRENT_PROJECT
cpu_budget: null

full_resolution: 4

# logging configs
...
```

The user can also define the number of CPU cores to use and the logging behaviors in `general_configs.yaml`. By default, FEABAS will try to use all the CPUs available; and log important info to the `logs` folder under the *working directory*, while keeping a more detailed record in the `logs/archive` folder.

### Stitching

The stitching pipeline comprises three distinctive steps: matching, optimization and rendering. The user can follow these steps by executing the main stitching script consecutively in different modes. `configs/stitching_configs.yaml` under the dataset *working directory* is the place to exercise finer control over the pipeline. For example, the user can balance the speed and memory usage by manipulating `num_workers` and `cache_size` fields in the YAML config file.

#### matching

To launch the matching step, first navigate to the FEABAS code root directory, and run:

```bash
python scripts/stitch_main.py --mode matching
```

The script parses the coordinate files in `(working_directory)/stitch/stitch_coord` folder, detects image overlaps, finds the matching points in these detected overlapping areas, and finally outputs the results to `(working_directory)/stitch/match_h5` folder in [HDF5](https://www.hdfgroup.org/solutions/hdf5/) file format.

If it encounters any errors during the matching step, FEABAS will still try to save whatever results it has but with `.h5_err` extension instead of the usual `.h5`, and at the same time register an error entry in the log file. In our experience, the most common failure case is a corrupted raw image file. After the issue is resolved, the user can run the matching command again, and FEABAS will pick up where it left by loading in the `.h5_err` file and only matching the remaining part. However, if tile arrangements in the stitch coordinate files were modified, or the contents of the existing images were changed, the user should delete the `.h5_err` file and start fresh for that section.

#### optimization

To launch the optimization step, navigate to the FEABAS code root directory, and run:

```bash
python scripts/stitch_main.py --mode optimization
```

It reads the `.h5` files in `(working_directory)/stitch/match_h5` folder, elastically deforms each image tile based on the matches found in the previous step, and finds the global lowest energy state for the system. The resulting transformations are then saved to `(working_directory)/stitch/tform` folder, also in HDF5 format.

#### rendering

To launch the rendering step, navigate to the FEABAS code root directory, and run:

```bash
python scripts/stitch_main.py --mode rendering
```

It reads the transformation files in `(working_directory)/stitch/tform` folder, and renders the stitched section in the form of non-overlapping PNG tiles. The user can control the rendering process (like the output tile size, whether to use [CLAHE](https://en.wikipedia.org/wiki/Adaptive_histogram_equalization#Contrast_Limited_AHE) etc.) by manipulating the `rendering` element in the *working directory*'s `configs/stitching_configs.yaml` file. By default, FEABAS will render the images to the *working directory*. If the user would like to keep the *working directory* lightweight and put the stitched images elsewhere, it can be achieved by defining the target path to `rendering: out_dir` field in the stitching configuration file.