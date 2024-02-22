# FEABAS

FEABAS (Finite-Element Assisted Brain Assembly System) is a Python library powered by finite-element analysis for stitching & alignment of serial-sectioning electron microscopy connectomic datasets.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![PyPI](https://img.shields.io/pypi/v/feabas.svg)](https://pypi.org/project/feabas/) [![Python](https://img.shields.io/pypi/pyversions/feabas)](https://www.python.org/downloads/)

## Installation

We used Python 3.10.4 to develop and test the package, but the codebase should be compatible with Python 3.8+. To install FEABAS, you can clone the [GitHub repository](https://github.com/YuelongWu/feabas) and pip-install it into a proper virtual environment:

```bash
git clone https://github.com/YuelongWu/feabas.git
cd feabas
pip install -e .
```
Note: on Apple silicon, you may need to manually install [triangle](https://github.com/drufat/triangle) from git repo `pip install git+https://github.com/drufat/triangle.git` due to missing wheels.


## Usage

### Preparation

The user needs to first create a dedicated *working directory* for each dataset that will go through FEABAS stitching and alignment pipelines. The *working directory* acts as a self-contained environment for individual datasets, defines project-specific configurations, and saves intermediate checkpoints/results generated through the workflow. At the beginning of the process, the *working directory* is expected to have the following file structure:

```
(working directory)
 |-- configs
 |   |-- stitching_configs.yaml (optional)
 |   |-- thumbnail_configs.yaml (optional)
 |   |-- alignment_configs.yaml (optional)
 |   |-- material_table.json (optional)
 |
 |-- stitch
 |   |-- stitch_coord
 |       |-- (section_name_0).txt
 |       |-- (section_name_1).txt
 |       |-- (section_name_2).txt
 |       |-- ...
 |
 |-- section_order.txt (optional)
```

#### configuration files
The `configs` folder in the *working directory* contains project-specific configuration files that override the default settings. If any of these files don't exist, FEABAS will use the corresponding default configuration files in the `configs` folder under the repository root directory (NOT the *working directory*) with the same file names but prefixed by `default_`, e.g. `default_stitching_configs.yaml`. The user can copy these default configuration files to their *working directory* `configs` folder, remove the prefix in the filename, and adjust the file contents accordingly based on the specific needs of their dataset.

#### stitch coordinate files
The .txt files in the `stitch/stitch_coord` folder are user-created [TSV](https://en.wikipedia.org/wiki/Tab-separated_values) files specifying the approximate tile arrangement for each section. They are the inputs to the stitcher pipeline of FEABAS and usually can be derived from the metadata from the microscopy. In one coordinate file, it first defines some metadata info like the root directory of the images, the pixel resolution (in nanometers), and the size of each image tile (height followed by width, in pixels). Following the metadata is a table of all the image tiles associated with that section, with the first column giving the relative path of each image file relative to the root directory, and the second and the third column defining the x and y coordinates (in pixels) of the images. An example stitch coordinate text file looks like this:

<div><code><ins>s0001.txt</ins></code></div>

```
{ROOT_DIR}	/home/feabas/my_project/raw_data/s0001
{RESOLUTION}	4.0
{TILE_SIZE}	3000	4000
Tile_0001.tif	0	0
Tile_0002.tif	3600	0
Tile_0003.tif	7200	0
Tile_0004.tif	0	2700
Tile_0005.tif	3600	2700
Tile_0006.tif	7200	2700
```

It describes a section whose raw image tiles from the microscopy are saved under the directory `/home/feabas/my_project/raw_data/s0001`. It contains 6 images with height of 3000 pixels and width of 4000 pixels, arranged on a 2-rows-by-3-columns grid with 10% overlaps. Note that in general the images do not necessarily need to be arranged in a rectilinear pattern and the image files can have arbitrary names, as long as the coordinates are as accurate as possible. Also, make sure that the fields in the coordinate files are separated by Horizontal Tab `\t`, other delimiters are currently not supported.

#### section order file (optional)
The filenames of the stitch coordinate text files define the name of the sections. By default, FEABAS assumes the order of sections in the final aligned stack can be reconstructed by sorting the section name alphabetically. If that's not the case, the user can define the right section order by providing an optional `section_order.txt` file directly under the working directory. In the file, each line is a section name corresponding to the stitch coordinate filenames (without `.txt` extension), and their positions in the file define their position in the aligned stack.

#### direct FEABAS to the current project
To enable FEABAS to identify the dataset it needs to process, the user needs to modify the `working_directory` field in the `configs/general_configs.yaml` file under FEABAS root directory:

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

To launch the matching step, first navigate to the FEABAS root directory, and run:

```bash
python scripts/stitch_main.py --mode matching
```

The script parses the coordinate files in `(work_dir)/stitch/stitch_coord` folder, detects image overlaps, finds the matching points in these detected overlapping areas, and finally outputs the results to `(work_dir)/stitch/match_h5` folder in [HDF5](https://www.hdfgroup.org/solutions/hdf5/) file format.

If it encounters any errors during the matching step, FEABAS will still try to save whatever results it has but with `.h5_err` extension instead of the usual `.h5`, and at the same time register an error entry in the log file. In our experience, the most common failure case is a corrupted raw image file. After the issue is resolved, the user can run the matching command again, and FEABAS will pick up where it left by loading in the `.h5_err` file and only matching the remaining part. However, if tile arrangements in the stitch coordinate files were modified, or the contents of the existing images were changed, the user should delete the `.h5_err` file and start fresh for that section.

#### optimization

To launch the optimization step, navigate to the FEABAS root directory, and run:

```bash
python scripts/stitch_main.py --mode optimization
```

It reads the `.h5` files in `(work_dir)/stitch/match_h5` folder, elastically deforms each image tile based on the matches found in the previous step, and finds the global lowest energy state for the system. The resulting transformations are then saved to `(work_dir)/stitch/tform` folder, also in HDF5 format.

#### rendering

To launch the rendering step, navigate to the FEABAS root directory, and run:

```bash
python scripts/stitch_main.py --mode rendering
```

It reads the transformation files in `(work_dir)/stitch/tform` folder, and renders the stitched section in the form of non-overlapping PNG tiles or [TensorStore](https://google.github.io/tensorstore/python/api/index.html) datasets (e.g. Neuroglancer precomputed format). The user can control the rendering process (like the output tile size, whether to use [CLAHE](https://en.wikipedia.org/wiki/Adaptive_histogram_equalization#Contrast_Limited_AHE) etc.) by manipulating the `rendering` element in the *working directory*'s `configs/stitching_configs.yaml` file. By default, FEABAS will render the images to the *working directory*. If the user would like to keep the *working directory* lightweight and put the stitched images elsewhere, it can be achieved by defining the target path (currently supporting local storage or [Google Cloud Storage](https://cloud.google.com/storage)) to `rendering: out_dir` field in the stitching configuration file

### Thumbnail Alignment

The FEABAS alignment workflow follows the 'coarse-to-fine' doctrine, with the thumbnail alignment as the most coarse alignment step that happens immediately after the stitching. The goal of the thumbnail alignment step is to find rough correspondences at a lower resolution between neighboring sections, making it easier for the finer matching step later on.

#### thumbnail generation

To generate the thumbnails of the stitched images, navigate to the FEABAS root directory, and run:

```bash
python scripts/thumbnail_main.py --mode downsample
```

The downsampled thumbnails will be written to `(work_dir)/thumbnail_align/thumbnails` folder. There are a few options the user can control via `configs/thumbnail_configs.yaml` in the *working directory*; among which `thumbnail_mip_level` and `downsample: thumbnail_highpass` are probably the most important.

- `thumbnail_mip_level` controls the resolution of the thumbnail by specifying its [mipmap level](https://en.wikipedia.org/wiki/Mipmap). The images are downsampled by a factor of two when the mipmap level increases by 1, with mip0 associated with the full resolution. The thumbnail should be small enough that each section can easily fit into one image file and the computation be efficient; while at the same time large enough so that there are enough image contents to extract features from. In our experience, the thumbnail alignment works best when the downsampled images are roughly 500~4000 pixels on each side. The user may target that thumbnail size based on the dimensions of their sections.

- `downsample: thumbnail_highpass`: we observed that for some datasets, applying a high-pass filter before the downsampling step will enhance the visibility of the somata in the thumbnails, thereby facilitating the downstream alignment steps. While we haven't concluded what datasets benefit most from this small trick, our current rule of thumb is to set this to true when working with secondary electron images, and false for images taken with backscattered mode. And if the `thumbnail_mip_level` is very small, better to turn the high-pass filter off.

#### thumbnail matching

To find the matching points between neighboring thumbnails, navigate to the FEABAS root directory, and run:

```bash
python scripts/thumbnail_main.py --mode match
```

It will save the files containing the rough matching points to `(work_dir)/thumbnail_align/matches` folder.

This is the most error-prone step in the entire pipeline, so we'd like to advise the user to pay special attention to the warning log files in `(work_dir)/logs` folder if they are generated during the step. In our experience, the most common failure mode is when the targeting of the microscope is not sufficiently accurate, the ROIs of some neighboring sections are completely off and there are little to no overlaps. In that case, the user should plan to retake the images.

As a last resort when thumbnail matching fails, FEABAS allows the user to manually define matching points via Fiji [BigWarp](https://imagej.net/plugins/bigwarp). First, load the failed pair of thumbnails from `(work_dir)/thumbnail_align/thumbnails` into [Fiji](https://imagej.net/software/fiji/), launch BigWarp, and select the first section as the `moving image` and the second section as the `target image`. Manually mark some corresponding points, preferably covering most of the overlapping regions, and export the landmarks as `(section_name_0)__to__(section_name_1).csv` file to `(work_dir)/thumbnail_align/manual_matches` folder. Navigate to the FEABAS root directory, then run `python tools/convert_manual_thumbnail_matches.py`, which will convert the manually defined matches and incorporate them into the downstream process.

By default, the thumbnail-matching step assumes all the connected regions in the thumbnail should have smooth transformation fields when aligning with their neighbors. However, that is not always the case. For example, if a section is broken into multiple pieces, and all the pieces are imaged as a single montage, then these disconnected parts may need very different transformations to be pieced back together. FEABAS allows the user to address this issue by modifying the mask files saved in the `(work_dir)/thumbnail_align/material_masks`. Each section has one corresponding mask file in PNG format that should overlay with its thumbnail. In the mask images, black color defines the regions within the ROI and white defines those outside of the ROI. The user can simply use the white color to label the split site, and break the black partial sections into disconnected 'islands'. Then FEABAS will automatically find transformations distinctive to each broken piece (as long as the pieces are not too fragmented). More about the masks in the "Advanced Topic: Non-Smooth Transformation in Alignment" segment of this readme.

### Fine Alignment

The fine alignment workflow refines the matching points found in the thumbnail alignment at a higher resolution, and then computes the transformations that align the entire stack by solving the system equation constructed by the finite-element method. The user can adjust the parameters related to the fine alignment workflow in `configs/alignment_configs.yaml` in the *working directory*.

#### generate meshes

One crucial component of finite-element analysis is mesh generation. FEABAS allows versatile definitions of mesh geometries and mechanical properties, which can help address a multitude of artifacts frequently encountered in connectomic datasets, e.g. broken sections, wrinkles and folds. More about that in "Advanced Topic: Non-Smooth Transformation in Alignment". For less challenging datasets, the user can simply run:

```bash
python scripts/align_main.py --mode meshing
```
to generate the meshes to `(work_dir)/align/mesh` folder, or directly run the `matching` command which will automatically generate the meshes if it is not done already. The mesh properties (e.g. mesh_size that defines the granularity of the meshes) can be controlled from the `meshing` part in `alignment_configs.yaml`.

#### matching

To launch the fine matching step of the alignment,  which aims to get denser and more accurate matching points from the thumbnail alignment results, navigate to the FEABAS root directory, and run:

```bash
python scripts/align_main.py --mode matching
```

It reads all the matching files from the thumbnail alignment, finds the right images from the folder that stores the stitched images, and performs template matching at a higher resolution. The results will be saved to `(work_dir)/align/matches`. The working resolution for the fine matching step can be defined by `matching: working_mip_level` field in `alignment_configs.yaml`. It is advisable to select a working mipmap level that makes the xy resolution of the image closer to its section thickness, making it more isotropic. For example, with 4nm full-resolution datasets, one can elect mip2 (16nm) or mip3 (32nm) for 30nm thick sections, or mip4 (64nm) for 80 nm sections.


#### optimization

To run the optimization step of the alignment workflow, navigate to the FEABAS root directory, and run:

```bash
python scripts/align_main.py --mode optimization
```

It loads the meshes in `(work_dir)/align/mesh` folder, arranges them in alphabetical order (or follows the order in `(work_dir)/section_order.txt` if provided), and uses the matches in `(work_dir)/align/matches` to drive the meshes into alignment. The aligned meshes are saved to `(work_dir)/align/tform` folder. FEABAS follows a "sliding window" strategy to perform the optimization step. It first selects a continuous subblock in the stack (size and location defined by `window_size` and `start_loc` fields in `alignment_configs.yaml`), optimizes within the block, and then moves the "window" to a neighboring block to repeat the process. The blocks before and after the "window" movement share a fixed number of sections (defined by `buffer_size` in `alignment_configs.yaml`), so that they are anchored to the same coordinate system.

Note that if there are meshes already existing in `(work_dir)/align/tform` folder when the optimization command is executed, the program will load those meshes instead of the meshes of the corresponding sections in `(work_dir)/align/mesh` folder; and treat those sections as 'locked', i.e. not allowed to move and only serve as references to the rest of the stack during the remaining optimization process.


#### rendering

To render the aligned stack, navigate to the FEABAS root directory, and run:

```bash
python scripts/align_main.py --mode rendering
```

Again, the user can control the details of the rendering process via `rendering` settings in `alignment_configs.yaml`. As with the rendering step in the stitching workflow earlier, the output images can be directed to another storage location other than the *working directory* by providing the path to `rendering: out_dir` field in the configuration file.

Alternatively, the aligned stack can also be rendered as a [TensorStore](https://google.github.io/tensorstore/python/api/index.html) volume, instead by running:

```bash
python scripts/align_main.py --mode tensorstore_render
```

The details of the rendering process are controlled by the `tensorstore_rendering` settings in `alignment_configs.yaml`. Note that currently the `rendering` settings and `tensorstore_rendering` are independent, i.e. change to one will not affect the behavior of the other alternative rendering format. The TensorStore rendering supports both local storage and Google Cloud storage destinations.


#### generate mipmaps for VAST

The user can also run:

```bash
python scripts/align_main.py --mode downsample
```
to generate the mipmaps of the aligned stack for visualization in [VAST](https://lichtman.rc.fas.harvard.edu/vast/).

If in the previous step, the aligned stack was rendered as a Tensorstore Volume, FEABAS currently does not provide a convenient way to create mipmaps. But this can easily achieved from existing third-party tools, e.g. [igneous](https://github.com/seung-lab/igneous#downsampling-downsampletask).


### distribute works on different machines

Most of the FEABAS commands (except the alignment meshing and optimization steps) support arguments `--start`, `--stop`, `--step` to constrain the procedure on a subset of the dataset. For example:

```bash
python scripts/stitch_main.py --mode rendering --start 5 --stop 20 --step 3
```
will only render every third section, starting from section #5 until section #20.

This becomes handy for distributing works on multiple machines in e.g. an production environment managed by [Slurm](https://slurm.schedmd.com/documentation.html)


### Advanced Topic: Non-Smooth Transformation in Alignment

The strength of finite element analysis lies in its ability to handle a wide range of geometries and mechanical properties. This package harnesses this power, utilizing finite element techniques to model non-smooth deformations in the alignment of serial-sectioning datasets, thereby tackling issues caused by artifacts like wrinkles and folds.

At the core of this feature's interface are two key components: material property configuration files and material masks.

- material property configuration files: this can be either `configs/material_table.json` file in the working directory, or `configs/default_material_table.json` in the FEABAS directory if the previous one is not present. The JSON files provided a list of materials, their properties, and their corresponding labels in the material masks. Here is an example of the definition of one material called "wrinkle" in `default_material_table.json`:
	```json
	"wrinkle": {
	"mask_label": 50,
	"enable_mesh": true,
	"area_constraint": 0,
	"render": true,
	"render_weight": 1.0e-3,
	"type": "MATERIAL_MODEL_ENG",
	"poisson_ratio": 0.0,
	"stiffness_multiplier": 0.05,
	"stiffness_func_factory": "feabas.material.asymmetrical_elasticity",
	"stiffness_func_params": {
		"strain": [0.0, 0.75, 1.0, 1.01],
		"stiffness": [1.5, 1.0, 0.5, 0.0]
		},
	"uid": 3
	}
	```
	- mask_label: the label representing this material in the material mask image files. For example, here all the pixels with a grayscale value of 50 will be designated as "wrinkle" material. Label 0 is reserved for default (normal) material, and 255 is reserved for empty regions.
	- enable_mesh: whether to generate mesh on regions defined by the material. If set to false, the labeled region will be excluded and considered empty.
	- area_constraint: it defines the granularity of the meshing grids in the regions assigned to this material. It's basically a modifier multiplied to the mesh_size argument when doing the meshing step. Therefore smaller value gives finer grids, with the only exception of 0, which will give the most coarse mesh possible under the geometric constraint.
	- render: whether to render the region assigned to this material. If set to false, the material will still exert its mechanical effects during the mesh relaxation step, but will not be rendered eventually.
	- render_weight: this is a variable that defines the priority for rendering. After transforming the mesh, some regions of the mesh may collide with other regions of the same mesh. In that case, materials with larger "render_weight" values will have a higher priority to be rendered. If the colliding parts have the same "render_weight", then the overlapping region will be split along the mid-line and each colliding partner will render its own half.
	- type: the type of the element to use for this material. Options are: `"MATERIAL_MODEL_ENG"` for [engineering (linear)](https://en.wikipedia.org/wiki/Linear_elasticity) model, `"MATERIAL_MODEL_SVK"` for [Saint Venant-Kirchhoff](https://en.wikipedia.org/wiki/Hyperelastic_material#Saint_Venant.E2.80.93Kirchhoff_model) model, or `"MATERIAL_MODEL_NHK"` for [Neo-Hookean](https://en.wikipedia.org/wiki/Neo-Hookean_solid) model. In practice, we found that `"MATERIAL_MODEL_ENG"` is sufficient for most of the cases.
	- poisson_ratio: [Poisson's ratio](https://en.wikipedia.org/wiki/Poisson%27s_ratio) to use if `type` is set to `"MATERIAL_MODEL_ENG"`.
	- stiffness_multiplier: multiplier applied to the stiffness matrix. Smaller values give softer materials and larger values make more rigid ones. The default material has a multiplier of 1.0.
	- stiffness_func_factory: function used to define nonlinear stress/strain relationship. It takes the keyword arguments defined in `stiffness_func_params`, and returns a function that maps the relative area change to a stiffness multiplier. For example, here I implemented `feabas.material.asymmetrical_elasticity` which is a wrapper around a linear interolator. By providing it with the strain/stiffness sample points in `stiffness_func_params`, it describes a material that can freely expand (strain > 1) but is hard to compress (strain < 1), which is exactly what we need to model the wrinkles. If set to null, the material stiffness is not a function of its area change.
	- uid: a unique identifier for each material. can be any integer other than 0 (reserved for the default material) and -1 (reserved for empty regions).

- material masks: for each section with artifacts that requires a non-smooth deformation field to align, a material mask should be provided. A material mask is a single PNG image (or a coordinate TXT file) that sits in the stitched image space. The grayscale value (`mask_label` in `material_table.json`) of each pixel in the mask image specifies the material type of the corresponding pixels of the stitched image (of the same mipmap level). Without any special treatment, FEABAS will automatically generate a mask for each section during the `thumbnail alignment > downsample` step. Those masks are at the same mipmap level as the thumbnails, and assume the section has normal properties (smooth deformation) everywhere within the imaged ROI. As mentioned earlier, a user can modify those masks to amend severe artifacts like split sections during the thumbnail alignment process. Later during the fine alignment step, FEABAS will use these thumbnail masks by default. However, a user can provide more precise masks at a higher resolution in `(work_dir)/align/material_masks` folder and specify their mipmap levels in the `alignment_configs.yaml` file. FEABAS will automatically use the higher-quality masks if they are available during the meshing/matching steps of the fine alignment workflow.
The generation of such masks falls outside the scope of this package. It can be done manually, or the user can try out a specialized deep-learning tool [PyTorch Connectomics](https://connectomics.readthedocs.io/en/latest/tutorials/artifact.html) developed by our collaborators from Pfister Lab at Harvard University.