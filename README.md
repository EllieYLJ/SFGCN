# SFGCN: Stable Fusion Graph Convolutional Network for EEG Emotion Recognition

This repository contains the official implementation of the Stable Fusion Graph Convolutional Network (**SFGCN**) proposed in the paper： 

>K. Shi, Y. Gong, R. Ouyang, et al. Electroencephalogram emotion recognition and analysis based on feature fusion and stable learning[J]. Biomedical Signal Processing and Control, 2025, 108: 107929.

SFGCN is designed to address limitations of existing Graph Convolutional Network (GCN)-based methods by integrating multi-graph topology construction, feature fusion, and stable learning.

---

## 1. Model Introduction

The **Stable Fusion Graph Convolutional Network (SFGCN)** is a novel model designed for EEG-based emotion recognition. It integrates **multiple graph topologies** and **stable learning** to enhance feature representation and generalization. Key components include:

- **Graph Structure Construction Block**: Constructs adjacency matrices using three metrics: Pearson Correlation Coefficient (PCC), Partial Directed Coherence (PDC), and Directed Transfer Function (DTF).

- **Feature Fusion Block**: Uses a Squeeze-and-Excitation (SE) block to adaptively fuse features from multiple graph structures.

- **Stable Learning Block**: Employs Random Fourier Features (RFF) and feature decorrelation to eliminate spurious correlations and improve feature independence.
---

## 2. Experimental Results


SFGCN achieves state-of-the-art performance on two benchmark EEG emotion datasets: DEAP and DREAMER. Below are the detailed classification accuracies (mean ± standard deviation).

**DEAP Dataset (4 Emotional Dimensions, 32 Channels)**

| Model       | Valence (%)      | Arousal (%)       | Dominance (%)     | Liking (%)        |
|-------------|------------------|-------------------|-------------------|-------------------|
| SFGCN (Ours)| 98.02 ± 1.26     | 98.36 ± 1.02      | 98.21 ± 0.96      | 97.89 ± 1.61      |

**DREAMER Dataset (3 Emotional Dimensions, 14 Channels)**

| Model       | Valence (%)      | Arousal (%)       | Dominance (%)     |
|-------------|------------------|-------------------|-------------------|
| SFGCN (Ours)| 92.42 ± 3.02     | 92.17 ± 3.74      | 91.92 ± 3.51      |


**Critical Channel Performance Analysis**

The 20 selected critical channels achieve comparable accuracy to the full 32-channel setup, with significantly reduced computational cost:

| Number of Channels | Valence Accuracy (%) | Arousal Accuracy (%) | Training Time (s/epoch) |
|-------------------|---------------------|---------------------|------------------------|
| 20 (Selected)     | 98.03 ± 1.20        | 98.18 ± 1.07        | 0.2722                 |
| 32 (Full Set)     | 98.02 ± 1.26        | 98.36 ± 1.02        | 0.3883                 |

---

## 3. Environment Dependencies

- Python ≥ 3.9  
- GPU: NVIDIA Quadro RTX 5000 (single GPU training)
- torch>=1.10.0
- numpy>=1.21.0
- scipy>=1.7.0
- scikit-learn>=1.0.0
- matplotlib>=3.4.0
- seaborn>=0.11.0
- plotly>=5.0.0
- mne>=1.0.0 

## 4. Experimental Data Processing
**DEAP Dataset**: A public multi-channel dataset with 32 subjects, 40 music video stimuli, and 32 EEG channels. The sampling rate is 128 Hz, and each sample is segmented into 1-second intervals (size: C×128, C=32). Emotional labels are binarized using a threshold of 5 (≥5 = 1, <5 = 0).

**DREAMER Dataset**: Contains 23 subjects, 18 video clips, and 14 EEG channels. The sampling rate is 128 Hz, with samples segmented into 1-second intervals (size: C×128, C=14). Labels are binarized using a threshold of 3 (≥3 = 1, <3 = 0).

## 5. Citation

If you use this code or model in your research, please cite the original paper:

### BibTeX
```bibtex
@article{shi2025eeg,
  title={Electroencephalogram emotion recognition and analysis based on feature fusion and stable learning},
  author={Shi, Kaiting and Gong, Yifan and Ouyang, Rujie and Yang, Lijun and Yang, Xiaohui and Zheng, Chen},
  journal={Biomedical Signal Processing and Control},
  volume={108},
  pages={107929},
  year={2025},
  publisher={Elsevier}
}