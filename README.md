# hippodeep-subfields
Brain Hippocampus Segmentation with Subfields

This program can quickly segment (<1min) the Hippocampus of raw brain T1 images.

This program has not been validated. It is not recommended to use it.

## Requirement

This program requires Python 3, with the PyTorch library, version > 1.0.0.

No GPU is required

Tested on Linux CentOS 6.x/7.x, Ubuntu 18.04 and MacOS X 10.13, using PyTorch versions 1.0.0 to 1.4.0

## Installation

Just clone or download this repository.

In addition to PyTorch, the code requires scipy and nibabel.

The simplest way to install from scratch is maybe to use a Anaconda environment, then
* install scipy (`conda install scipy` or `pip install scipy`) and  nibabel (`pip install nibabel`)
* get pytorch for python from `https://pytorch.org/get-started/locally/`. CUDA is not necessary.


## Usage:
To use the program, simply call:

`deepseg1.sh example_brain_t1.nii.gz`.

To process multiple subjects, pass them as multiple arguments.
`deepseg1.sh subject_*.nii.gz`.

The resulting segmentations should be stored as `example_brain_t1_mask_L.nii.gz` (or R for right) and `example_brain_t1_brain_mask.nii.gz`.  The mask volumes (in mm^3) are stored in a csv file named `example_brain_t1_hippoLR_volumes.csv`.  If more than one input was specified, a summary table named `all_subjects_hippo_report.csv` is created.
