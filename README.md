# CT Reconstruction: Classical and Learned Methods

BEng Final Year Dissertation — University of Manchester, 2026
Student ID: 11397920

## Overview

This repository contains all code for the dissertation *"Evaluating Classical 
and Learned CT Reconstruction Methods in a Unified Ray-Tracing Framework 
Under Realistic Degradation"*.

The project implements CT image reconstruction from scratch in MATLAB using 
Siddon's ray-tracing algorithm, evaluating Filtered Back Projection (FBP), 
the Algebraic Reconstruction Technique (ART), and Ordered-Subset SART 
(OS-SART) under noiseless, sparse-angle, and Poisson noise conditions. 
A DnCNN denoising network is then trained in Python as a learned 
post-processing step, comparing MSE-only and combined MSE+SSIM loss functions.

---

## Files

| File | Description |
|------|-------------|
| `siddon_ray.m` | Core ray-tracing function. Computes intersection lengths of a ray with each pixel in the image grid using Siddon's algorithm. Used by both forward and back projection. |
| `siddon_forward.m` | Forward projection. Generates a sinogram from a given image by applying Siddon ray tracing across all angles and detector positions. |
| `siddon_backprojection.m` | Back projection. Distributes projection values back onto the image grid using the same ray-tracing geometry, forming a discrete adjoint pair with the forward projector. |
| `NoiselessComparison.m` | Experiment 1 and 2. Reconstructs the Shepp-Logan phantom using FBP and ART under noiseless conditions and across sparse angle sets (180, 90, 45, 30 angles). |
| `NoisyComparison.m` | Experiment 3 and 4. Evaluates FBP, ART, and OS-SART under Poisson noise at three dose levels (I0 = 1e3, 1e4, 1e5), including relaxation parameter sweeps and semiconvergence analysis. |
| `TrainingDataGen.m` | Generates the DnCNN training dataset. Produces 250 randomised Shepp-Logan phantoms, applies Poisson noise, reconstructs clean and noisy FBP pairs, and saves to `ct_dataset_siddon.mat`. |
| `ConvertData.py` | Converts `ct_dataset_siddon.mat` to HDF5 format for PyTorch. Adds a channel dimension and splits into train, val, and test sets saved to `ct_dataset_siddon.h5`. |
| `dncnn-arch.ipynb` | Kaggle training notebook. Contains both Network A (MSE loss) and Network B (MSE+SSIM loss, alpha=0.7). Trains for 50 epochs with Adam and step learning rate decay. |
| `DnCNNEval.py` | Dose generalisation evaluation. Loads both trained models and evaluates RMSE and SSIM across eight dose levels, three of which match training and five are unseen. |

---

## How to Run

### MATLAB Experiments

1. Add the repository folder to your MATLAB path
2. Run `NoiselessComparison.m` for FBP vs ART baseline and sparse angle results
3. Run `NoisyComparison.m` for FBP vs ART vs OS-SART under Poisson noise

> All experiments use the Shepp-Logan phantom at 256x256 resolution with 
> 180 projection angles and 365 detector positions.

### DnCNN Training

1. Run `TrainingDataGen.m` in MATLAB to generate `ct_dataset_siddon.mat`
2. Run `ConvertData.py` to produce `ct_dataset_siddon.h5`
3. Upload the `.h5` file to Kaggle as a dataset
4. Open `dncnn-arch.ipynb` on Kaggle and run all cells

### Evaluation

```bash
pip install torch h5py numpy scikit-image matplotlib pandas
python DnCNNEval.py
```

---

## Dataset

The dataset is not included in this repository due to file size (~200MB). 
To regenerate it, run `TrainingDataGen.m` followed by `ConvertData.py`. 
The dataset consists of 250 randomised Shepp-Logan phantoms reconstructed 
under three dose levels (I0 = 1e3, 1e4, 1e5), split into 200 training, 
25 validation, and 25 test samples.

---

## Dependencies

**MATLAB:** R2024a or later (no additional toolboxes required)

**Python:**
