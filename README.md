# 3D U-Net Convolution Neural Network with Keras - Modified version to segment cells from 3D confocoal images.

## Background
Originally designed after [this paper](http://lmb.informatik.uni-freiburg.de/Publications/2016/CABR16/cicek16miccai.pdf) on 
volumetric segmentation with a 3D U-Net.

## Tutorial using the Data and Python 3
### Training
1. Data is currently not available publicly.

2. Install dependencies: 
```
nibabel,
keras,
pytables,
nilearn,
SimpleITK,
nipype
```
(nipype is required for preprocessing only) 

3. Install [ANTs N4BiasFieldCorrection](https://github.com/stnava/ANTs/releases) and add the location of the ANTs 
binaries to the PATH environmental variable.

4. Add the repository directory to the ```PYTONPATH``` system variable:
```
$ export PYTHONPATH=${PWD}:$PYTHONPATH
```
5. Convert the data to nifti format and perform image wise normalization and correction:

cd into the brats subdirectory:
```
$ cd brats
```
Import the conversion function and run the preprocessing:
```
$ python
>>> from preprocess import convert_brats_data
>>> convert_brats_data("data/original", "data/preprocessed")
```
6. Run the training:

To run training using the original UNet model:
```
$ python train.py
```

To run training using an improved UNet model (recommended): 
```
$ python train_isensee2017.py
```
**If you run out of memory during training:** try setting 
```config['patch_shape`] = (64, 64, 64)``` for starters. 
Also, read the "Configuration" notes at the bottom of this page.

```
$ python predict.py
```
The predictions will be written in the ```prediction``` folder along with the input data and ground truth labels for 
comparison.

## Citations
GBM Data Citation:
 * Spyridon Bakas, Hamed Akbari, Aristeidis Sotiras, Michel Bilello, Martin Rozycki, Justin Kirby, John Freymann, Keyvan Farahani, and Christos Davatzikos. (2017) Segmentation Labels and Radiomic Features for the Pre-operative Scans of the TCGA-GBM collection. The Cancer Imaging Archive. https://doi.org/10.7937/K9/TCIA.2017.KLXWJJ1Q

LGG Data Citation:
 * Spyridon Bakas, Hamed Akbari, Aristeidis Sotiras, Michel Bilello, Martin Rozycki, Justin Kirby, John Freymann, Keyvan Farahani, and Christos Davatzikos. (2017) Segmentation Labels and Radiomic Features for the Pre-operative Scans of the TCGA-LGG collection. The Cancer Imaging Archive. https://doi.org/10.7937/K9/TCIA.2017.GJQ7R0EF
