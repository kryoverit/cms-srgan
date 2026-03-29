# SRGAN for CMS Calorimeter Super-Resolution

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

## Project Overview

This repository contains a Super-Resolution GAN (SRGAN) implementation for enhancing the resolution of calorimeter images from the CMS detector at CERN. The project aims to map low-resolution (64×64) particle collision data to high-resolution (125×125) representations to improve jet classification and particle identification.

**Key Features:**
- SRGAN architecture optimized for sparse scientific imaging data
- Custom loss functions for sparsity preservation and edge enhancement
- Channel-aware training for balanced multi-channel reconstruction
- Comprehensive evaluation metrics and KPIs

## Results Summary

| Metric | Value | Notes |
|--------|-------|-------|
| **PSNR** | 23.71 ± 1.20 dB | +4.7 dB over bicubic baseline |
| **SSIM** | 0.9092 ± 0.0280 | Excellent structural similarity |
| **Overall KPI** | 89.6/100 | Strong performance across all metrics |

**Channel Performance:**
- Red: 20.19 dB (needs improvement)
- Green: 26.18 dB (acceptable)  
- Blue: 40.62 dB (excellent)

## Repository Structure

```
srgan_cms_github/
├── srgan.py              # Training script
├── inference.py          # Inference and evaluation
├── srgan.pth             # Trained model checkpoint
├── README.md             # This file
└── requirements.txt      # Python dependencies
```

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.0+ (for GPU acceleration)

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/srgan-cms.git
cd srgan-cms

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.21.0
pyarrow>=10.0.0
matplotlib>=3.5.0
scipy>=1.9.0
scikit-image>=0.19.0
tqdm>=4.64.0
```

## Usage

### Training
```python
# Basic training
python srgan.py

# Training with custom parameters
python srgan.py \
    --data_dir /path/to/parquet/files \
    --max_samples 10000 \
    --batch_size 8 \
    --epochs_pretrain 10 \
    --epochs_gan 20
```

### Inference
```python
# Generate super-resolved images
python inference.py \
    --model srgan.pth \
    --num_samples 10 \
    --output_dir results/

# Evaluate model performance
python inference.py --evaluate
```

### TensorBoard Logging
```bash
# View training metrics
tensorboard --logdir runs/
```

## Model Architecture

### Generator
- **Input:** 64×64×3 low-resolution calorimeter image
- **Encoder:** 12 residual blocks with 96 features each
- **Upsampling:** Pixel shuffle (2×) followed by bilinear resize
- **Output:** 125×125×3 super-resolved image
- **Activation:** Sigmoid (normalized to [0,1])

### Discriminator
- 6 convolutional layers with spectral normalization
- Progressive downsampling with LeakyReLU activations
- Adaptive average pooling for final classification

### Loss Functions
1. **Sparsity-Weighted L1 Loss:** Emphasizes non-zero pixels (energy deposits)
2. **Edge Preservation Loss:** Sobel-based edge detection
3. **Feature Matching Loss:** Prevents mode collapse
4. **Adversarial Loss:** WGAN-GP with gradient penalty

## Evaluation Metrics

### Key Performance Indicators (KPIs)
| KPI | Score | Interpretation |
|-----|-------|----------------|
| PSNR | 79.0/100 | Good pixel accuracy |
| SSIM | 90.9/100 | Excellent structural similarity |
| Channel Balance | 100.0/100 | All channels >20 dB |
| Sparsity Preservation | 90.7/100 | Good sparsity maintenance |
| Distribution Match | 77.1/100 | Reasonable distribution alignment |
| Overall | **89.6/100** | **Strong performance** |

### Channel-wise Analysis
- **Red Channel:** 20.19 dB (low energy deposits)
- **Green Channel:** 26.18 dB (medium energy)
- **Blue Channel:** 40.62 dB (high energy)

## Dataset

The model is trained on simulated CMS detector data containing:
- **Low Resolution:** 64×64 pixel calorimeter images
- **High Resolution:** 125×125 pixel ground truth images
- **Format:** Parquet files with paired LR/HR data
- **Channels:** 3 (representing different energy deposits)

### Data Preprocessing
1. Clip negative values to zero
2. Apply log1p transformation: `log(1 + x)`
3. Per-image percentile normalization (99.9th)
4. Convert to float32 in [0,1] range

## Results and Visualization

### Sample Outputs
![Sample Comparison](images/comparison_grid.png)
*Comparison of low-resolution input, ground truth, and super-resolved output*

### Training Stability
- **Generator Loss:** 0.0639 → 0.0449 (29.7% reduction)
- **Discriminator Loss:** 2.6766 → 0.2826 (89.4% reduction)
- **G/D Ratio:** 0.024 → 0.159 (healthy range >0.1)

## Future Work

### Short-term Improvements
1. **Channel-specific weighting:** Increase red channel importance
2. **Sparsity penalty:** Explicit constraint on zero/nonzero patterns
3. **Extended training:** 30-50 GAN epochs for better convergence

### Long-term Goals
1. **Transformer architectures:** VAR, JEPA, Diffusion models
2. **Physics integration:** Detector response embedding
3. **Real-time processing:** Optimized for online reconstruction
4. **3D extension:** Volumetric calorimeter data

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **ML4Sci** for mentorship and project guidance
- **CMS Collaboration** for providing detector data and physics expertise
- **CERN** for computational resources and support

## Contact

**Author:** Akshat Jalan  
**Email:** akshatjalan204@gmail.com  
**LinkedIn:** linkedin.com/in/akshatjalan04  
**GitHub:** [Your GitHub Username]  

**Project:** GSoC 2026 - Super Resolution at the CMS Detector  
**Organization:** ML4Sci  
**Mentors:** Eric Reinhardt, Diptarko Choudhury, Ruchi Chudasama, Emanuele Usai, Sergei Gleyzer

---

**Citation:** If you use this code in your research, please cite:
```bibtex
@software{srgan_cms_2026,
  author = {Akshat Jalan},
  title = {SRGAN for CMS Calorimeter Super-Resolution},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/yourusername/srgan-cms}}
}
```
