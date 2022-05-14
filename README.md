# Repository for Point Label Aware Superpixels
The official repository for the paper: 'Point Label Aware Superpixels for Multi-species Segmentation of Underwater Imagery'

\[[arXiv](https://arxiv.org/abs/2202.134874)]

Our approach contributes to the field of segmentation of underwater imagery by enabling generation of dense, pixel-wise ground truth masks for training a model to perform semantic segmentation.  Many coral datasets are accompanied by sparse, randomly distributed point labels.  Our approach leverages the deep features at each pixel as well as the locations of the point labels to generate superpixels which conform to complex coral boundaries and encompass single species regions.  If this repository contributes to your research, please consider citing the publication below.

```
Scarlett Raine and Ross Marchant and Brano Kusy and Frederic Maire and Tobias Fischer (2022). 
Point Label Aware Superpixels for Multi-species Segmentation of Underwater Imagery.  Under Review for RAL and IROS.
```

### Bibtex
```
@article{raine2022point,
  title={Point Label Aware Superpixels for Multi-species Segmentation of Underwater Imagery},
  author={Raine, Scarlett and Marchant, Ross and Kusy, Brano and Maire, Frederic and Fischer, Tobias},
  journal={arXiv preprint arXiv:2202.13487},
  year={2022}
}

```
## Table of Contents
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Example](#example)
- [Acknowledgements](#acknowledgements)

<a name="installation"></a>
## Installation
We suggest using the Anaconda package manager to install dependencies.

1. Download Anaconda
2. Create a conda environment: conda create -n pointlabelspix python=...
3. Activate the environment: conda activate pointlabelspix
4. Install packages and libraries:

<a name="getting-started"></a>
## Getting Started
Ensure you have a folder with images and another folder with the ground truth.  This can be in the form of dense masks or sparse, randomly distributed point labels.
If your data is dense, the script will generate a set of sparse labels. 

Run the script using:

```python propagate.py```

You can alter the functionality of the method using the following arguments:


## Example Execution - UCSD Mosaics Dataset



<a name="acknowledgements"></a>
## Acknowledgements
This work was done in collaboration between QUT and CSIRO's Data61. 
