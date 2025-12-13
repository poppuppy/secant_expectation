# Learning to Integrate Diffusion ODEs by Averaging the Derivatives

**Wenze Liu**, **Xiangyu Yue**

MMLab, The Chinese University of Hong Kong

[![Paper](https://img.shields.io/badge/arXiv-PDF-b31b1b)](https://arxiv.org/abs/2505.14502)

This is the official implementation for the NeurIPS 2025 paper.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/poppuppy/secant-expectation.git
   cd secant-expectation
   ```

2. Install the required dependencies.
   ```bash
   pip install torch torchvision numpy tqdm diffusers timm lmdb Pillow accelerate
   ```

3. Install `torch-fidelity` for evaluation:
   ```bash
   pip install -e git+https://github.com/LTH14/torch-fidelity.git@master#egg=torch-fidelity
   ```

## Data and Model Preparation

### Dataset
We use pre-extracted VAE latent features for ImageNet-256 provided by [yuchuantian/imagenet_vae_256](https://huggingface.co/datasets/yuchuantian/imagenet_vae_256).

### Pretrained Checkpoints
Our method involves fine-tuning or distilling from a pretrained SiT model. You need to prepare the initial weights as follows:

1. **Download the SiT Checkpoint:**
   Download `SiT-XL-2-256.pt` from the official SiT repository link:
   [Download Link (Dropbox)](https://www.dl.dropboxusercontent.com/scl/fi/as9oeomcbub47de5g4be0/SiT-XL-2-256.pt?rlkey=uxzxmpicu46coq3msb17b9ofa&dl=0), and move the downloaded SiT-XL-2-256.pt to `pretrained/`.

2. **Convert Checkpoint:**
   Run the transfer script in `pretrained` to convert the weights for dual time-point input support:
   ```bash
   python ckpt_transfer.py
   ```

## Training

To train the model (e.g., training the SDEI variant for 1-step generation), you can use the provided shell script.

```bash
bash train.sh
```

*Note: Please ensure the data path in `train.sh` points to your downloaded `imagenet_vae_256` dataset.*

## Evaluation

To evaluate the FID score of the models, run:

```bash
bash test.sh
```

## Checkpoints

We provide trained checkpoints for different settings (SDEI, STEE, SDEE, STEE, various step counts) on [Hugging Face](https://huggingface.co/poppuppy/secant-expectation).

## Acknowledgments

This repository is built upon [fast-DiT](https://github.com/chuanyangjin/fast-DiT) and [SiT](https://github.com/willisma/SiT).

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@article{liu2025learning,
  title={Learning to Integrate Diffusion ODEs by Averaging the Derivatives},
  author={Liu, Wenze and Yue, Xiangyu},
  journal={arXiv preprint arXiv:2505.14502},
  year={2025}
}
```
