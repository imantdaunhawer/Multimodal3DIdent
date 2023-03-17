# Multimodal3DIdent Dataset Generation

Official code for generating the *Multimodal3DIdent* dataset introduced in the
paper [Identifiability Results for Multimodal Contrastive
Learning](http://arxiv.org/abs/2303.09166) presented at [ICLR
2023](https://iclr.cc/Conferences/2023). The dataset provides an identifiability
benchmark with image/text pairs generated from controllable ground truth
factors, some of which are shared between image and text modalities, as
illustrated in the following examples.

<p align="center">
  <img src="https://github.com/imantdaunhawer/Multimodal3DIdent//blob/master/assets/examples.jpg?raw=true" alt="Multimodal3DIdent dataset example images" width=370 />
</p>

## Description

The generated dataset contains image and text data as well as the ground
truth factors of variation for each modality. The dataset is structured as
follows:
```
.
├── images
│   ├── 000000.png
│   ├── 000001.png
│   └── etc.
├── text
│   └── text_raw.txt
├── latents_image.csv
└── latents_text.csv
```
The directories `images` and `text` contain the generated image and text data,
whereas the CSV files `latents_image.csv` and `latents_text.csv` contain the
values of the respective latent factors. There is an index-wise correspondence
between images, sentences, and latents. For example, the first line in
the file `text_raw.txt` is the sentence that corresponds to the first image in
the `images` directory.

### Latent factors
We use the following ground truth latent factors to generate image and text
data. Each factor is sampled from a uniform distribution defined on the
specified set of values for the respective factor.


| Modality | Latent Factor         | Values               | Details
| :------- | :-----------------    | :-----------         | :-------
| Image    | Object shape          | {0, 1, ..., 6}       | Mapped to Blender shapes like "Teapot", "Hare", etc.
| Image    | Object x-position     | {0, 1, 2}            | Mapped to {-3, 0, 3} for Blender
| Image    | Object y-position     | {0, 1, 2}            | Mapped to {-3, 0, 3} for Blender
| Image    | Object z-position     | {0}                  | Constant
| Image    | Object alpha-rotation | [0, 1]-interval      | Linearly transformed to [-pi/2, pi/2] for Blender
| Image    | Object beta-rotation  | [0, 1]-interval      | Linearly transformed to [-pi/2, pi/2] for Blender
| Image    | Object gamma-rotation | [0, 1]-interval      | Linearly transformed to [-pi/2, pi/2] for Blender
| Image    | Object color          | [0, 1]-interval      | Hue value in HSV transformed to RGB for Blender
| Image    | Spotlight position    | [0, 1]-interval      | Transformed to a unique position on a circle
| Image    | Spotlight color       | [0, 1]-interval      | Hue value in HSV transformed to RGB for Blender
| Image    | Background color      | [0, 1]-interval      | Hue value in HSV transformed to RGB for Blender
| Text     | Object shape          | {0, 1, ..., 6}       | Mapped to strings like "teapot", "hare", etc.
| Text     | Object x-position     | {0, 1, 2}            | Mapped to strings "left", "center", "right"
| Text     | Object y-position     | {0, 1, 2}            | Mapped to strings "top", "mid", "bottom"
| Text     | Object color          | string values        | Color names from 3 different color palettes
| Text     | Text phrasing         | {0, 1, ..., 4}       | Mapped to 5 different English sentences
<!-- TODO add examples? See README in commit 1085fb55c73-->

### Image rendering
We use the Blender rendering engine to create visually complex images
depicting a 3D scene. Each image in the dataset shows a colored 3D object of a
certain shape or class (i.e., teapot, hare, cow, armadillo, dragon, horse, or
head) in front of a colored background and illuminated by a colored spotlight
that is focused on the object and located on a circle above the scene. The
resulting RGB images are of size 224 x 224 x 3.

### Text generation
We generate a short sentence describing the respective scene. Each sentence
describes the object's shape or class (e.g., teapot), position (e.g.,
bottom-left), and color. The color is represented in a human-readable form
(e.g., "lawngreen", "xkcd:bright aqua", etc.) as the name of the color (from
a randomly sampled palette) that is closest to the sampled color value in RGB
space. The sentence is constructed from one of five pre-configured phrases
with placeholders for the respective ground truth factors.

### Relation between modalities
Three latent factors (object shape, x-position, y-position) are shared between
image/text pairs. The object color also exhibits a dependence between
modalities; however, it is not a 1-to-1 correspondence because the color palette
is sampled randomly from a set of multiple palettes. It is easy to customize
the existing dependencies or to add additional dependencies between latent
factors, see [customize dependencies](https://github.com/imantdaunhawer/Multimodal3DIdent#customize-dependencies).


## Usage

To use the dataset, you can either generate the data from scratch or download
the full dataset from [Zenodo](https://zenodo.org/record/7678231). To generate
the data from scratch, [install Blender](https://www.blender.org/download/)
(ideally, [version 2.90.1](https://download.blender.org/release/Blender2.90))
and proceed as follows:

```bash
# install dependencies (preferably, inside your conda/virtual environment)
$ pip install -r requirements.txt

# set an alias for the blender executable (adjust the path to your installation!)
$ alias blender="/home/username/downloads/blender-2.90.1-linux64/blender"

# generate ten image/text pairs and save them to the directory "example"
$ python generate_latents.py --output-folder "example" --n-points 10
$ python generate_text.py --output-folder "example"
$ blender -noaudio --background --python generate_images.py -- --use-gpu --output-folder "example"
```

Note that for the generation of images, all options prior to ` -- ` are
specific to Blender, while all arguments following ` -- ` are passed to the
Python script `generate_images.py`.

### PyTorch Dataset

To load the data using PyTorch, you can use the file
[datasets.py](datasets.py). For example, to load the example dataset generated
above, execute the following commands in Python:

```python
>>> from datasets import Multimodal3DIdent
>>> dataset = Multimodal3DIdent("example")  # init the example dataset
>>> sample = dataset[0]                     # load the first sample
>>> print(sample.keys())
dict_keys(['image', 'text', 'z_image', 'z_text'])
```
As illustrated above, each sample is a dict that contains the corresponding
`image`, `text`, and ground truth factors `z_image` and `z_text` respectively.
Note that the text is returned as a 1-hot encoding, which can easily be
adjusted in the function `__getitem__` in [datasets.py](datasets.py).

### Customize dependencies

It is easy to customize the dependencies between latent factors and to adjust
which factors are shared between modalities by changing the respective
variables in the file [generate_latents.py](generate_latents.py). 

For example, we already provide the option to impose a causal dependence of
object color on object x-position through the argument
`--position-dependent-color`. When activated, the range of hue values [0, 1]
is split into three equally sized intervals, each of which is associated with
a fixed x-position of the object.  For instance, if x-position is "left", we
sample the hue value from the interval [0, 1/3].  Consequently, the color of
the object can be predicted to some degree from the object's position. 


## BibTeX
If you find the dataset useful, please cite our paper:

```bibtex
@article{daunhawer2023multimodal,
  author = {
    Daunhawer, Imant and
    Bizeul, Alice and
    Palumbo, Emanuele and
    Marx, Alexander and
    Vogt, Julia E.
  },
  title = {
    Identifiability Results for Multimodal Contrastive Learning
  },
  booktitle = {International Conference on Learning Representations},
  year = {2023}
}
```

## Acknowledgements

This project builds on the following resources. Please cite them appropriately.
- https://github.com/brendel-group/cl-ica <3
- https://github.com/ysharma1126/ssl_identifiability <3
- https://github.com/facebookresearch/clevr-dataset-gen <3
- https://github.com/blender/blender <3
