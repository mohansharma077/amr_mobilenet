# Bacterial Antimicrobial Susceptibility Testing via Deep Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.15](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)](https://www.tensorflow.org/)


## 🎯 Project Overview

This project implements a deep learning pipeline for rapid antimicrobial susceptibility testing (AST) using single-cell bacterial microscopy images. By analyzing morphological changes in bacterial cells exposed to antibiotics, we can classify them as resistant or susceptible in **30 minutes** instead of the traditional 18-24 hours.

### Key Features
- 🔬 Single-cell phenotyping using fluorescence microscopy
- 🧠 Deep learning classification (MobileNetV2/DenseNet121)
- ⚡ Rapid results (~30 minutes vs 18-24 hours)


## 📂 Dataset

This project uses the **DASP (Deep Antimicrobial Susceptibility Phenotyping)** dataset:
- **Organism:** *Escherichia coli* (MG1655 strain + clinical isolates)
- **Antibiotics:** Ciprofloxacin, Gentamicin, Rifampicin, Co-amoxiclav
- **Imaging:** Dual-channel fluorescence (DAPI for DNA, Nile Red for membrane)
- **Resolution:** 30 frames per field of view

> **Note:** Dataset not included in repository due to size. Contact  to access.




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
```
base = MobileNetV2(weights='imagenet', include_top=False)
x = GlobalAveragePooling2D()(base.output)
x = Dense(128, activation='relu')(x)
output = Dense(2, activation='softmax')(x)
```



## 📝 Methodology


### Preprocessing Details
```python
# Extract single cell
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

Contributions are welcome! 



## 📧 Contact

**Mohan Sharma** - u2976733@uel.ac.uk

Project Link: [https://github.com/mohansharma077/amr_mobilenet])

---

**Note:** This is an academic replication project for educational purposes.
EOF

```python
