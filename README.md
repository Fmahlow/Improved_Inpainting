# Improved Inpainting

Image editing framework for **automatic background and attire replacement** using **monocular depth estimation** and **text-guided inpainting**. The pipeline combines MiDaS to generate masks automatically and diffusion models (Stable Diffusion + LCMs/SDXL Inpainting) to produce realistic, prompt-aligned results.

> This repository hosts the experimental artifacts and quantitative analysis for the paper *“Automated Inpainting for Attire and Background Replacement Leveraging Depth Estimation”*.

## Overview
The method follows a three-stage flow:
1. **Depth estimation + mask creation**: MiDaS generates depth maps and a mean-based threshold separates subject and background. Face detection removes facial regions from the mask to preserve identity.
2. **Background generation and replacement**: Stable Diffusion with Latent Consistency Models (LCMs) synthesizes a new background in a few steps.
3. **Attire inpainting**: SDXL Inpainting applies a dedicated mask guided by a text prompt while preserving face and background.

## Key contributions
- **Automatic depth-based mask generation** (no manual segmentation).
- **Integrated background + attire replacement** in a single pipeline.
- **LCM acceleration** for fast generation with high visual quality.

## Repository structure
- `experiment/`: generated images (prompt/threshold combinations).
- `reference_images/`: reference images used by metrics (background/attire).
- `experiment_metrics.csv`: metrics table produced during evaluation.
- `metrics.ipynb`: notebook for **PSNR** and **SSIM**.
- `parse_experiment.ipynb`: notebook for **CLIP Score**, **FID**, and plots.

## Requirements (metrics analysis)
To reproduce the notebooks, you will need:
- Python 3.9+
- Jupyter Notebook/Lab
- Core libraries:
  - `pandas`, `numpy`, `Pillow`, `regex`
  - `matplotlib`
  - `torch`, `torchvision`, `torchmetrics`
  - `clip` (OpenAI CLIP)

## Running the metrics
1. Start Jupyter:
   ```bash
   jupyter notebook
   ```
2. Run:
   - `metrics.ipynb` for **PSNR** and **SSIM**.
   - `parse_experiment.ipynb` for **CLIP Score**, **FID**, and analysis plots.

> The notebooks expect the `experiment/` and `reference_images/` folders plus `experiment_metrics.csv`.

## Notes on prompts and thresholds
- The threshold controls the subject/background split; values between **1.1 and 1.3** typically balance background preservation and inpainting quality.
- Creative prompts (e.g., “a person with arms like a tree branch”) are more challenging and can introduce anatomical artifacts.

## Citation
If you use this work, please cite the paper:

> **Automated Inpainting for Attire and Background Replacement Leveraging Depth Estimation**

## Contact
For questions or access to the full pipeline implementation, please contact the authors of the paper.
