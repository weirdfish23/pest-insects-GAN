# pest-insects-GAN

A PyTorch framework for synthetic data generation and data augmentation of greenhouse pest insects on chromatic sticky traps. The project implements class-conditional deep generative models (Conditional DCGAN, Conditional WGAN-GP, and Conditional VAE) to mitigate acute class imbalance and synthesize annotated full-scale sticky trap images for training object detectors such as YOLOv5.

---

## Table of Contents

- [Overview](#overview)
- [Target Pest Species](#target-pest-species)
- [Pipeline Architecture](#pipeline-architecture)
- [Generative Model Architectures](#generative-model-architectures)
  - [Joint Class-Conditional Training](#joint-class-conditional-training)
  - [Conditional DCGAN](#1-conditional-dcgan)
  - [Conditional WGAN-GP](#2-conditional-wgan-gp)
  - [Conditional VAE](#3-conditional-vae)
- [Repository Structure](#repository-structure)
- [Environment Setup](#environment-setup)
- [Configuration Reference (`config.json`)](#configuration-reference-configjson)
- [Step-by-Step Usage Guide](#step-by-step-usage-guide)
  - [1. Data Acquisition](#1-data-acquisition)
  - [2. Offline Traditional Balancing](#2-offline-traditional-balancing)
  - [3. Training Conditional Generative Models](#3-training-conditional-generative-models)
  - [4. Sticky Trap Image Synthesis](#4-sticky-trap-image-synthesis)
  - [5. YOLOv5 Dataset Formatting & Detection](#5-yolov5-dataset-formatting--detection)
- [Checkpoints and Model Selection](#checkpoints-and-model-selection)
- [Operational Notes and Troubleshooting](#operational-notes-and-troubleshooting)

---

## Overview

Automated greenhouse pest monitoring relies heavily on chromatic (yellow and blue) sticky traps. While computer vision object detectors can automate insect counts, real-world trap datasets suffer from extreme class imbalance: common pests appear in thousands of instances, whereas quarantine or low-density pests appear only in dozens of instances.

This repository provides an end-to-end framework to resolve this imbalance:
1. **Patch-level Augmentation**: Combines traditional spatial/photometric transforms with class-conditional deep generative models (cDCGAN, cWGAN-GP, cVAE) to generate realistic 64x64 RGB insect patches conditioned on species class.
2. **Scene-level Composition**: Composites generated and real insect patches onto high-resolution sticky trap background templates with hue matching and non-overlapping spatial placement.
3. **Detection Export**: Automatically records bounding boxes and exports datasets directly into normalized YOLOv5 annotation format (`[class_id, x_center, y_center, width, height]`).

---

## Target Pest Species

The dataset contains seven pest species indexed alphabetically for one-hot encoding ($C = 7$):

| Class Index | Scientific Name | Common Name | Typical Natural Count | Baseline Augmentation Multiplier |
|---|---|---|---|---|
| 0 | *Bemisia tabaci* | Sweetpotato whitefly | High (~5,800) | 0 (No extra offline copies) |
| 1 | *Macrolophus pygmaeus* | Mirid predatory bug | High (~1,600) | 0 (No extra offline copies) |
| 2 | *Nesidiocoris tenuis* | Tomato bug / mirid | Medium (~680) | 1 (1 new copy per original) |
| 3 | *Brevicoryne brassicae* | Cabbage aphid | Low (~58) | 23 (23 new copies per original) |
| 4 | *Liriomyza huidobrensis* | Serpentine leafminer | Low (~112) | 23 (23 new copies per original) |
| 5 | *Prodiplosis longifila* | Citrus gall midge | Low (~35) | 23 (23 new copies per original) |
| 6 | *Trips tabaci* | Onion thrips | Low (~53) | 23 (23 new copies per original) |

The initial class balancing step equalizes sample representation before generative model training.

---

## Pipeline Architecture

The system operates across five sequential stages:

```
[Google Drive Base Data]
           |
           v
  dowload_base_data.py
           |
           v
[data/base: 64x64 crops & CSV]
           |
           v
  traditional_data_augmentation.py  --> Multipliers configured in config.json
           |
           v
[data/augmented: balanced 64x64 crops]
           |
           +----------------------------------+----------------------------------+
           |                                  |                                  |
           v                                  v                                  v
 train_conditional_GAN.py           train_conditional_WGAN.py          train_conditional_VAE.py
           |                                  |                                  |
           +----------------------------------+----------------------------------+
                                              |
                                              v
                                      [Saved Weights]
                                              |
                                              v
                       pipeline_recreate_sticky_traps_images.ipynb
                       (Selects best model per class + pastes on templates)
                                              |
                                              v
                                 [Composite Full-Trap Images]
                                              |
                                              v
                                   prepare_train_yolov5.py
                                              |
                                              v
                              [YOLOv5 train/val dataset structure]
                                              |
                                              v
                                    YOLOv5 Object Detector
```

---

## Generative Model Architectures

All generative models operate on 64x64 RGB patches ($3 \times 64 \times 64$) with pixel values normalized to $[-1, 1]$.

### Joint Class-Conditional Training

Each model family (cDCGAN, cWGAN-GP, and cVAE) is trained as a single unified network jointly across all seven insect classes simultaneously, rather than training seven individual models per species. The shared backbone learns general representations of trap backgrounds, lighting, and insect features, while class conditioning modulates species-specific attributes:
- **Latent Conditioning (Generator / Decoder)**: The 7-dimensional one-hot class vector $y \in \{0, 1\}^7$ is concatenated directly with the 64-dimensional latent noise vector $z \sim \mathcal{N}(0, I_{64})$, creating a 71-dimensional input $(B, 71, 1, 1)$.
- **Spatial Conditioning (Discriminator / Critic / Encoder)**: The one-hot class vector is spatially broadcast to $(B, 7, 64, 64)$ and channel-concatenated with the $3 \times 64 \times 64$ RGB image, producing a 10-channel input tensor $(B, 10, 64, 64)$.

### 1. Conditional DCGAN

Defined in `models/conditional_GAN.py` and trained via `train_conditional_GAN.py`:

- **Generator**:
  - Input: Latent vector $z \sim \mathcal{N}(0, I_{64})$ concatenated with one-hot vector $y \in \{0, 1\}^7$ ($d_{\text{in}} = 71$).
  - Reshaped to $(B, 71, 1, 1)$ and upsampled through 5 transposed convolutional layers with `BatchNorm2d` and `ReLU`.
  - Output activation: `Tanh` producing $(B, 3, 64, 64)$.
- **Discriminator**:
  - Input: Image tensor $(B, 3, 64, 64)$ concatenated channel-wise with the one-hot label tensor spatially tiled to $(B, 7, 64, 64)$, yielding $(B, 10, 64, 64)$.
  - Downsampled through 5 convolutional layers with `BatchNorm2d` and `LeakyReLU(0.2)`.
  - Final layer outputs a raw scalar logit per sample.
- **Loss**: Binary Cross-Entropy with Logits (`nn.BCEWithLogitsLoss`).

### 2. Conditional WGAN-GP

Defined in `models/conditional_WGAN.py` and trained via `train_conditional_WGAN.py`:

- **Generator**: Same transposed convolutional architecture as DCGAN.
- **Critic**: Similar convolutional backbone to the discriminator, outputting an unconstrained Wasserstein score (no sigmoid).
- **Critic Objective**:
  $$\mathcal{L}_{\text{critic}} = \mathbb{E}[\mathcal{D}(G(z, y), y)] - \mathbb{E}[\mathcal{D}(x, y)] + \lambda \mathbb{E}\left[\left(\|\nabla_{\hat{x}} \mathcal{D}(\hat{x}, y)\|_2 - 1\right)^2\right]$$
  where $\hat{x} = \epsilon x + (1 - \epsilon) G(z, y)$ with $\epsilon \sim \mathcal{U}(0, 1)$, and penalty coefficient $\lambda = 10$.
- **Training Schedule**: The critic is updated 5 times (`crit_repeats = 5`) per generator update. Adam optimizer configured with $\beta_1 = 0.5, \beta_2 = 0.999$.

### 3. Conditional VAE

Defined in `models/conditionalVAE.py` and trained via `train_conditional_VAE.py`:

- **Encoder**:
  - Input: Image concatenated with spatially replicated one-hot label $(B, 10, 64, 64)$.
  - Downsampled via 5 convolutional layers to 512 channels, flattened and projected via linear layers to mean $\mu \in \mathbb{R}^{64}$ and log-variance $\log(\sigma^2) \in \mathbb{R}^{64}$.
  - Latent sampling: $z = \mu + \sigma \odot \epsilon$, where $\epsilon \sim \mathcal{N}(0, I_{64})$.
- **Decoder**:
  - Input: Latent $z$ concatenated with one-hot vector $y$ ($d_{\text{in}} = 71$).
  - Linear projection to $(B, 512, 1, 1)$ followed by 6 transposed-convolutional blocks, outputting $(B, 3, 64, 64)$ with `Tanh`.
- **Loss**:
  $$\mathcal{L}_{\text{VAE}} = \text{MSE}(\hat{x}, x) + \beta_{\text{KLD}} \cdot D_{\text{KL}}\left(\mathcal{N}(\mu, \sigma^2) \,\|\, \mathcal{N}(0, I)\right)$$
  where $\beta_{\text{KLD}} = 10^{-4}$ (`kld_weight`).

---

## Repository Structure

```
pest-insects-GAN/
|-- config.json                               # Central configuration file
|-- dataset.yaml                              # YOLOv5 dataset description (classes and paths)
|-- requirements.txt                          # Python dependencies
|-- dowload_base_data.py                      # Google Drive dataset downloader and extractor
|-- traditional_data_augmentation.py          # Class-wise geometric and photometric balancer
|-- prepare_train_yolov5.py                   # YOLOv5 bounding box format converter
|-- train_conditional_GAN.py                  # Training pipeline for Conditional DCGAN
|-- train_conditional_WGAN.py                 # Training pipeline for Conditional WGAN-GP
|-- train_conditional_VAE.py                  # Training pipeline for Conditional VAE
|
|-- models/
|   |-- conditional_GAN.py                    # DCGAN Generator and Discriminator modules
|   |-- conditional_WGAN.py                   # WGAN-GP Generator, Critic, and GP functions
|   +-- conditionalVAE.py                     # CVAE Encoder, Decoder, and ELBO loss
|
|-- utils/
|   |-- read.py                               # Configuration JSON loader
|   |-- dataloader.py                         # InsectsDataset, ToTensor, ToTensorNorm transforms
|   +-- imgs.py                               # Tensor visualization and augmentation helpers
|
|-- notebooks/
|   |-- pipeline_recreate_sticky_traps_images.ipynb # Trap synthesis, blending, and export
|   |-- test_traditional_data_augmentation.ipynb   # Augmentation testing and visualization
|   +-- test_train_conditional_GAN.ipynb           # Model debugging and sanity checks
|
+-- weights/                                  # Pretrained checkpoints directory
    |-- conditional_DCGAN_*                   # DCGAN generator and discriminator checkpoints
    |-- conditional_WGAN_*                    # WGAN generator and critic checkpoints
    +-- conditional_VAE_*                     # CVAE encoder and decoder checkpoints
```

---

## Environment Setup

### 1. Create and Activate Virtual Environment

Using Conda:

```bash
conda create -n pests python=3.8.5 -y
conda activate pests
```

Or using standard `venv`:

```bash
python3 -m venv pests-env
source pests-env/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

*Note on PyTorch versioning*: `requirements.txt` specifies `torch==1.5.1` and `torchvision==0.6.1` for legacy CUDA environments. The code is compatible with modern PyTorch releases (1.8+ through 2.x). If using modern GPUs (CUDA 11.x / 12.x or Apple Silicon MPS), install the appropriate PyTorch build from the official PyTorch channel before installing remaining dependencies.

### 3. WandB Configuration

Training scripts log metrics and generated image grids to Weights & Biases (`wandb`). Set your API key or enable offline mode:

```bash
# To run offline without a WandB account:
export WANDB_MODE=offline

# Or log in with your credentials:
wandb login
```

---

## Configuration Reference (`config.json`)

All workflows are controlled by `config.json`. Key parameters include:

| Section | Parameter | Description |
|---|---|---|
| `base_data` | `url`, `dest_dir` | Google Drive source URL and destination directory (`data`) |
| `n_data_augmentation` | Key-value pairs | Number of augmented crops generated per original crop per species |
| `conditional_GAN` | `img_shape` | Target crop resolution `[3, 64, 64]` |
| | `z_dim`, `batch_size`, `lr` | Latent vector dimension (64), batch size (128), Adam learning rate (2e-4) |
| | `device` | Compute target: `"cuda"` or `"cpu"` |
| | `load_weights`, `w_to_load` | Boolean flag and checkpoint paths for resumption |
| `conditional_WGAN` | `c_lambda`, `crit_repeats` | Gradient penalty coefficient (10) and critic updates per generator step (5) |
| | `beta_1`, `beta_2` | Adam momentum parameters (0.5, 0.999) |
| `conditional_VAE` | `kld_weight` | Coefficient $\beta$ weighting the KL-divergence term (0.0001) |
| `detection` | `data_url`, `dataset_version`| Source URL for full trap evaluation dataset and target version directory |

---

## Step-by-Step Usage Guide

### 1. Data Acquisition

Ensure destination folder exists and download base patches:

```bash
mkdir -p data
python3 dowload_base_data.py
```

This extracts raw insect patches and annotations into `data/base/`.

### 2. Offline Traditional Balancing

Generate augmented versions of minority classes to build a balanced dataset:

```bash
python3 traditional_data_augmentation.py
```

This script:
- Reads original crops from `data/base/images/` using `data/base/data_info.csv`.
- Applies random horizontal flips, rotations (degrees up to 180 filled with mean image color), and color jitter (brightness, contrast, saturation, hue).
- Saves augmented crops to `data/augmented/images/` and outputs `data/augmented/data_info.csv`.

### 3. Training Conditional Generative Models

Set `"device": "cuda"` (or `"cpu"`) in `config.json`.

**Train Conditional DCGAN:**
```bash
python3 train_conditional_GAN.py
```

**Train Conditional WGAN-GP:**
```bash
python3 train_conditional_WGAN.py
```

**Train Conditional VAE:**
```bash
python3 train_conditional_VAE.py
```

Checkpoints are saved in `weights/<model_name>_<timestamp>/` at intervals defined by `save_epoch`.

### 4. Sticky Trap Image Synthesis

Use `pipeline_recreate_sticky_traps_images.ipynb` to synthesize high-resolution sticky traps:
1. Downloads empty sticky trap background templates (`templates_url`).
2. Loads trained generators for each species according to empirical image quality:
   - *Liriomyza huidobrensis* -> DCGAN
   - *Brevicoryne brassicae* -> WGAN-GP
   - *Prodiplosis longifila* -> VAE
   - *Trips tabaci* -> VAE
   - *Bemisia tabaci* -> VAE
   - *Macrolophus pygmaeus* -> DCGAN
   - *Nesidiocoris tenuis* -> WGAN-GP
3. Samples insect patches (mixing real patches and synthetic generations via `gen_imgs_ratio`).
4. Performs hue adjustments to match the template background lighting.
5. Randomly pastes non-overlapping patches onto the template image and computes bounding boxes `[xmin, ymin, xmax, ymax]`.
6. Exports composite images along with a metadata DataFrame containing annotations and provenance flags (`generated` vs `notGenerated`).

### 5. YOLOv5 Dataset Formatting & Detection

Convert detection annotations into YOLO format:

```bash
python3 prepare_train_yolov5.py
```

This script:
- Splits dataset into 90% train and 10% validation.
- Normalizes bounding boxes:
  $$x_{\text{center}} = \frac{x_1 + x_2}{2 \cdot W}, \quad y_{\text{center}} = \frac{y_1 + y_2}{2 \cdot H}, \quad w = \frac{x_2 - x_1}{W}, \quad h = \frac{y_2 - y_1}{H}$$
- Populates `data/images/train/`, `data/images/val/`, `data/labels/train/`, and `data/labels/val/`.

To train YOLOv5 on the generated dataset:

```bash
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
pip install -r requirements.txt
python train.py --img 640 --batch 16 --epochs 100 --data ../dataset.yaml --weights yolov5s.pt
```

---

## Checkpoints and Model Selection

The repository provides pretrained weights under `weights/`:
- `conditional_DCGAN_17_04_21__23_47_12/`: DCGAN checkpoints up to epoch 550.
- `conditional_WGAN_19_04_21__01_08_04/`: WGAN-GP checkpoints up to epoch 600.
- `conditional_VAE_23_04_21__14_36_25/`: CVAE checkpoints up to epoch 500.

To resume training or evaluate a specific checkpoint, set `"load_weights": true` and specify the relative checkpoint path in `config.json` under `"w_to_load"`.

Each checkpoint dictionary stores:
- `epoch`: Epoch number at checkpoint time.
- `state_dict`: Model parameters.
- `optimizer`: Optimizer state.

---

## Operational Notes and Troubleshooting

1. **Target Device**: `config.json` defaults to `"cuda"`. If running locally without a dedicated GPU, change `"device"` to `"cpu"` in all model sections of `config.json`.
2. **Missing `data` Directory**: If `dowload_base_data.py` fails with `FileNotFoundError`, create the target directory manually (`mkdir -p data`) before executing the script.
3. **WandB In Headless Environments**: If running in non-interactive batch jobs or environments without browser access, set `export WANDB_MODE=offline` to prevent blocking on authentication prompts.
4. **Coordinate Convention**: The pipeline uses standard computer vision coordinates: $(x_1, y_1)$ represents top-left and $(x_2, y_2)$ represents bottom-right before conversion to normalized YOLO center coordinates.

---

By Joel Cabrera Rios
