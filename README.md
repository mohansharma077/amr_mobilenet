# Bacterial Antimicrobial Susceptibility Testing via Deep Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.15](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Replication and extension of the paper: *"Deep learning and single-cell phenotyping for rapid antimicrobial susceptibility detection in Escherichia coli"* ([Zagajewski et al., 2023](https://doi.org/10.1038/s42003-023-05524-4))

## 🎯 Project Overview

This project implements a deep learning pipeline for rapid antimicrobial susceptibility testing (AST) using single-cell bacterial microscopy images. By analyzing morphological changes in bacterial cells exposed to antibiotics, we can classify them as resistant or susceptible in **30 minutes** instead of the traditional 18-24 hours.

### Key Features
- 🔬 Single-cell phenotyping using fluorescence microscopy
- 🧠 Deep learning classification (MobileNetV2/DenseNet121)
- ⚡ Rapid results (~30 minutes vs 18-24 hours)
- 📊 Achieves ~75% accuracy on small dataset (Phase 1)
- 🎯 Target: 80-84% accuracy on full dataset (Phase 2)

## 📂 Dataset

This project uses the **DASP (Deep Antimicrobial Susceptibility Phenotyping)** dataset:
- **Organism:** *Escherichia coli* (MG1655 strain + clinical isolates)
- **Antibiotics:** Ciprofloxacin, Gentamicin, Rifampicin, Co-amoxiclav
- **Imaging:** Dual-channel fluorescence (DAPI for DNA, Nile Red for membrane)
- **Resolution:** 30 frames per field of view

> **Note:** Dataset not included in repository due to size. Contact  for access.

## 🚀 Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/mohansharma077/amr_mobilenet.git
cd bacterial-ast-classification

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Usage

**Option 1: Jupyter Notebook (Recommended for beginners)**
```bash
jupyter notebook notebooks/phase1_pipeline.ipynb
```



## 📊 Results (Phase 1)





### Sample Results

![Training Curves](results/figures/training_curves.png)
![Confusion Matrix](results/figures/confusion_matrix.png)
![Example Predictions](results/figures/example_predictions.png)

## 🏗️ Architecture

### Preprocessing Pipeline
1. **Multi-frame averaging** (30 frames → 1 image)
2. **Channel registration** (align DAPI to Nile Red)
3. **Cell segmentation** (using provided masks)
4. **Normalization** (64×64 pixels, histogram equalization)

### Model
- **Base:** MobileNetV2 (Phase 1) / DenseNet121 (Phase 2)
- **Pretraining:** ImageNet weights
- **Input:** (64, 64, 3) RGB images
- **Output:** Binary classification (Resistant/Susceptible)

# Simplified architecture
base = MobileNetV2(weights='imagenet', include_top=False)
x = GlobalAveragePooling2D()(base.output)
x = Dense(128, activation='relu')(x)
output = Dense(2, activation='softmax')(x)
```

## 📈 Roadmap

### ✅ Phase 1 (Complete)
- [x] Data inspection and preprocessing
- [x] Extract 1,000 cell dataset
- [x] Train MobileNetV2 classifier
- [x] Achieve ~72% accuracy
- [x] Generate evaluation figures

### 🚧 Phase 2 (In Progress)
- [ ] Scale to full dataset (20,000+ cells)
- [ ] Switch to DenseNet121 architecture
- [ ] Train with GPU acceleration
- [ ] Test on clinical isolates (EC1-EC6)
- [ ] Target: 80-84% accuracy

### 🔮 Phase 3 (Future)
- [ ] Implement dose-response curves
- [ ] Add remaining antibiotics (Gentamicin, Rifampicin, Co-amoxiclav)
- [ ] Develop web interface for predictions
- [ ] Real-time inference optimization

## 📝 Methodology

### Key Differences from Original Paper

| Aspect | Original Paper | This Implementation |
|--------|---------------|---------------------|
| **Dataset size** | ~30,000 cells | 1,000 cells (Phase 1) |
| **Architecture** | DenseNet121 | MobileNetV2 (Phase 1) |
| **Training time** | Not specified | 25 min (CPU) |
| **Segmentation** | Mask R-CNN (trained) | Pre-computed masks |
| **Hardware** | GPU | CPU (Phase 1) |

### Preprocessing Details
```python
# Example: Extract single cell
def extract_cell(dapi_stack, nile_red_stack, mask):
    # 1. Average 30 frames
    dapi_avg = np.mean(dapi_stack, axis=0)
    nile_red_avg = np.mean(nile_red_stack, axis=0)
    
    # 2. Register channels
    shift = cv2.phaseCorrelate(nile_red_avg, dapi_avg)
    dapi_registered = cv2.warpAffine(dapi_avg, shift)
    
    # 3. Crop to cell bounding box
    # 4. Resize to 64×64
    # 5. Histogram equalization
    # 6. Return RGB image
```

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use this code in your research, please cite:
```bibtex
@article{zagajewski2023deep,
  title={Deep learning and single-cell phenotyping for rapid antimicrobial susceptibility detection in Escherichia coli},
  author={Zagajewski, Alexander and Turner, Piers and others},
  journal={Communications Biology},
  volume={6},
  number={1},
  pages={1164},
  year={2023},
  publisher={Nature Publishing Group}
}

@misc{mohansharma077,
  title={Bacterial AST Classification: Implementation and Extension},
  author={Mohan Sharma},
  year={2026},
  }
}
```

## 🙏 Acknowledgments

- Original paper authors: Zagajewski et al., 2023
- DASP dataset providers
- University of East London for computational resources
- TensorFlow and Keras teams

## 📧 Contact

**Mohan Sharma** - u2976733@uel.ac.uk

Project Link: [https://github.com/mohansharma077/amr_mobilenet])

---

**Note:** This is an academic replication project for educational purposes.
EOF

```python
