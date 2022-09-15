# Repository for Point Label Aware Superpixels
The official repository for the paper: 'Point Label Aware Superpixels for Multi-species Segmentation of Underwater Imagery'

\[[arXiv](https://arxiv.org/abs/2202.134874)]  \[[IEEE Xplore](https://ieeexplore.ieee.org/document/9813385)]  \[[YouTube](https://youtu.be/elPAOIVZl-c)]

Our approach contributes to the field of segmentation of underwater imagery by enabling generation of dense, pixel-wise ground truth masks for training a model to perform semantic segmentation.  Many coral datasets are accompanied by sparse, randomly distributed point labels.  Our approach leverages the deep features at each pixel as well as the locations of the point labels to generate superpixels which conform to complex coral boundaries and encompass single species regions.  If this repository contributes to your research, please consider citing the publication below.

```
S. Raine, R. Marchant, B. Kusy, F. Maire and T. Fischer, "Point Label Aware Superpixels for Multi-Species Segmentation of Underwater Imagery," in IEEE Robotics and Automation Letters, vol. 7, no. 3, pp. 8291-8298, July 2022, doi: 10.1109/LRA.2022.3187836.
```

https://user-images.githubusercontent.com/50187455/190310353-a74940b6-ce3a-4b58-8b62-abca52391ae0.mp4


### Bibtex
```
@ARTICLE{9813385,
  author={Raine, Scarlett and Marchant, Ross and Kusy, Brano and Maire, Frederic and Fischer, Tobias},
  journal={IEEE Robotics and Automation Letters}, 
  title={Point Label Aware Superpixels for Multi-Species Segmentation of Underwater Imagery}, 
  year={2022},
  volume={7},
  number={3},
  pages={8291-8298},
  doi={10.1109/LRA.2022.3187836}}

```
## Table of Contents
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Acknowledgements](#acknowledgements)

<a name="installation"></a>
## Installation
We suggest using the Mamba (or Anaconda) package manager to install dependencies.

1. Download Mamba
2. Create a mamba environment: 

```mamba create -n pointlabelspix python pytorch matplotlib scikit-image scipy torchmetrics -c conda-forge ```

3. Activate the environment: 

```conda activate pointlabelspix```

4. Install pyg package:

```mamba install pyg -c pyg```


<a name="getting-started"></a>
## Getting Started
Ensure you have a folder with images and another folder with corresponding ground truth masks.  The ground truth can be dense, pixel wise masks or sparse, randomly distributed point labels.

If your data is dense, the script will generate a set of sparse labels. If you have a class for 'unlabeled' or similar, it will still be used in generating the augmented ground truth, however it will not be used in calculating the accuracy. 

The script will save the augmented ground truth masks in the specified directory as .png images, where each value indicates the class at that pixel in the correponding image.  

For the best performance, make sure to use the '--ensemble' flag, which means our approach uses an ensemble of three classifiers.  If you need to prioritize speed over accuracy, then leaving this out means that only a single classifier will be used.

Download the weights for our feature extractor here: \[[Feature Extractor](https://drive.google.com/file/d/1F7325ISXUTppWbO5_3eopOf7rEhgfJdP/view?usp=sharing)]
Make sure the file is in the same location as the other scripts.

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
