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
- [Acknowledgements](#acknowledgements)

<a name="installation"></a>
## Installation
We suggest using the Anaconda package manager to install dependencies.

1. Download Anaconda
2. Download and save the provided spix_environment.yml file in your directory
3. Create the environment using:
```conda env create -f spix_environment.yml ```

**Alternative Method**

1. Download Anaconda
2. Create a conda environment: 

```conda create -n pointlabelspix python=3.9.9 ```

3. Activate the environment: 

```conda activate pointlabelspix```

4. Install packages and libraries:


```conda install pytorch torchvision cudatoolkit=11.3 -c pytorch ```

```conda install -c conda-forge matplotlib==3.5.1 scikit-image==0.19.1 scipy==1.7.3 torchmetrics```

```conda install pyg -c pyg```


<a name="getting-started"></a>
## Getting Started
Ensure you have a folder with images and another folder with corresponding ground truth masks.  The ground truth can be dense, pixel wise masks or sparse, randomly distributed point labels.

If your data is dense, the script will generate a set of sparse labels. If you have a class for 'unlabeled' or similar, it will still be used in generating the augmented ground truth, however it will not be used in calculating the accuracy. 

The script will save the augmented ground truth masks in the specified directory as .png images, where each value indicates the class at that pixel in the correponding image.  

For the best performance, make sure to use the '--ensemble' flag, which means our approach uses an ensemble of three classifiers.  If you need to prioritize speed over accuracy, then leaving this out means that only a single classifier will be used.

Run the script using:

```python propagate.py```

You must provide the following arguments:
* '-r', '-read_im', type=str, help='the path to the images', required=True
* '-g', '-read_gt', type=str, help='the path to the provided labels', required=True
* '-l', '-save_labels', type=str, help='the destination of your propagated labels', required=True

Use the following to change the functionality:
* '--ensemble', action='store_true', dest='ensemble', help='use this flag when you would like to use an ensemble of 3 classifiers, otherwise the default is to use a single classifier'
* '--points', action='store_true', dest='points', help='use this flag when your labels are already sparse, otherwise the default is dense'

The following are optional arguments: the default values correspond to the UCSD Mosaics dataset
* '-p', '--save_rgb', type=str, help='the destination of your RGB propagated labels'
* '-x', '--xysigma', type=float, default=0.631, help='if NOT using ensemble and if you want to specify the sigma value for the xy component'
* '-f', '--cnnsigma', type=float, default=0.5534, help='if NOT using ensemble and if you want to specify the sigma value for the cnn component'
* '-a', '--alpha', type=float, default=1140, help='if NOT using ensemble and if you want to specify the alpha value for weighting the conflict loss'
* '-n', '--num_labels', type=int, default=300, help='if labels are dense, specify how many random point labels you would like to use, default is 300'
* '-y', '--height', type=int, default=512, help='height in pixels of images'
* '-w', '--width', type=int, default=512, help='width in pixels of images'
* '-c', '--num_classes', type=int, default=35, help='the number of classes in the dataset'
* '-u', '--unlabeled', type=int, default=34, help='the index of the unlabeled/unknown/background class'

An example: This is for the UCSD Mosaics dataset which is a densely labeled dataset (the script will randomly select the sparse point labels), saving RGB augmented ground truth masks, using the ensemble of classifiers and using 100 point labels per image.

```python propagate.py -r "D:\\Mosaics UCSD dataset\\Mosaics UCSD dataset\\Mosaicos\\images\\train" -g "D:\\Mosaics UCSD dataset\\Mosaics UCSD dataset\\Mosaicos\\labels\\train" -l "D:\\Mosaics UCSD dataset\\Mosaics UCSD dataset\\Mosaicos\\test" -p "D:\\Mosaics UCSD dataset\\Mosaics UCSD dataset\\Mosaicos\\test_rgb" --ensemble --num_labels 100```

The UCSD Mosaics dataset can be downloaded from the authors of the CoralSeg paper: \[[Dataset](https://sites.google.com/a/unizar.es/semanticseg/home)]

<a name="acknowledgements"></a>
## Acknowledgements
This work was done in collaboration between QUT and CSIRO's Data61. 
