# Merging Library (merging_lib)

A production-ready, modular Python library for advanced Stable Diffusion and FLUX model merging, surgery, and analysis.

## 🚀 Features

### 1. Advanced Merging
- **Multi-Model TIES Merging**: Trim, Elect Sign, and Merge for base + unlimited models.
- **Multi-Model DARE Merging**: Drop And REscale (Hydra) for base + unlimited models.
- **Universal MBW**: Multi-Block Weighted merging with SLERP support for FLUX, SDXL, Pony, and Z-Image.
- **SLERP Interpolation**: High-quality spherical linear interpolation for tensors.

### 2. Model Surgery
- **Frankenstein Block Swapper**: Swap entire architectural blocks (Input, Output, Middle, etc.) between models.
- **VAE Fixer**: Adjust contrast and brightness of VAE decoder layers.
- **Model Modifier**: Scale specific blocks or layers by a multiplier.
- **VAE Baker**: Inject or replace VAE in a checkpoint.
- **LoRA Baker**: Stream-bake LoRA into a checkpoint with adjustable strength.

### 3. LoRA Tools
- **Reverse LoRA Extractor**: Extract deltas between two models as a full-rank LoRA.
- **LoRA Compressor**: SVD-based compression to reduce LoRA rank/size.
- **SVD LoRA Merger**: Merge two LoRAs with different ranks into a single target rank.

### 4. Analysis & Utilities
- **The Analyst**: Cosine Similarity and MSE layer-by-layer plotter.
- **Architecture X-Ray**: Detailed structural reporting for FLUX, SDXL, SD1.5, and Z-Image.
- **Inspect Model**: Metadata display, size checking, and SHA256 hashing.
- **Converters**: Diffusers-to-Safetensors and bulk precision conversion (fp8, fp16, bf16, fp32).

## 📦 Installation

```bash
pip install torch safetensors tqdm numpy ml_dtypes matplotlib
```

## 🛠️ Usage

Refer to `merging_v7.ipynb` for a comprehensive Interactive Notebook interface.

```python
from merging_lib import ties_merge

ties_merge(
    base_model_path="base.safetensors",
    models_paths=["model1.safetensors", "model2.safetensors"],
    weights=[0.5, 0.5],
    output_path="merged.safetensors"
)
```
