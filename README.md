# hippodeep-subfields
Brain Hippocampus Segmentation with Subfields

This program can quickly segment (<1min) the Hippocampus of raw brain T1 images.

This program has not been validated. It is not recommended to use it.

## Requirement

This program requires Python 3, with the PyTorch library, version > 1.0.0.

No GPU is required

ANTs is also necessary, as the program currently calls 'antsApplyTransforms'
(windows executable and dll dependencies included here)

Tested on Linux CentOS 6.x/7.x, Ubuntu 18.04 and MacOS X 10.13, using PyTorch versions 1.0.0 to 1.4.0
Windows version tested on Win7x64 and Win10x64

## Installation

Just clone or download this repository.

In addition to PyTorch, the code requires scipy and nibabel.
Windows version: additionally requires "psutil" (otionally "pywin32" is strongly recomendend)

The simplest way to install from scratch is maybe to use a Anaconda environment, then
* install scipy (`conda install scipy` or `pip install scipy`) and  nibabel (`pip install nibabel`)
* get pytorch for python from `https://pytorch.org/get-started/locally/`. CUDA is not necessary.

## Windows Installation

* pip install numpy scipy nibabel
* pip install torch==1.3.0+cpu -f https:&#8203;//download.pytorch.org/whl/torch_stable.html
* pip install psutil pywin32

to compile an executable:

* pip install pyinstaller
* pyinstaller HippoDeep_subfields.spec
* ---> find your executable under .\dist\HippoDeep_subfields.exe


## Usage:
To use the program, simply call:

`deepseg.sh example_brain_t1.nii.gz`.

Use -h for usage, in particular, -d to keep higher-resolutions images

if called without argument a file dialougue should appear
<br/>
(this is, if you have tkinter installed with: "yum install tkinter" or "apt-get install python3-tk")

To process multiple subjects, pass them as multiple arguments.
`deepseg1.sh subject_*.nii.gz`.

High resolution low FOV hipocampus images are stored as `example_brain_t1_boxL.nii.gz` and `example_brain_t1_boxR.nii.gz` for left and righ hipocampi respectively 
The resulting high resolution segmentations are stored as `example_brain_t1_boxL_hippo.nii.gz` and `example_brain_t1_boxR_hippo.nii.gz` respectively
Segmentation files transformed back to the original full FOV image are stored as `example_brain_t1_hippoL_native.nii.gz` and `example_brain_t1_hippoR_native.nii.gz`
The resulting volume quantification values (in mm^3) of all subvolumes are stored in a csv file named `example_brain_t1_hippoLR_volumes.csv`.

