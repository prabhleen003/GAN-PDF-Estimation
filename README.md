# Assignment 4: Learning Probability Density Functions using GANs

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)

---

## 📋 Overview

This project uses a **Generative Adversarial Network (GAN)** to learn the probability density function of transformed NO₂ air quality data without assuming any specific distribution form (Gaussian, exponential, etc.).

**What We Did:**
1. Transformed NO₂ data using roll-number-based function
2. Built and trained a GAN to learn the data distribution
3. Generated synthetic samples from the trained model
4. Estimated PDF using Kernel Density Estimation
5. Validated results using statistical tests

---

## 📊 Dataset

**Source:** [India Air Quality Data](https://www.kaggle.com/datasets/shrutibhargava94/india-air-quality-data)

**Feature:** NO₂ (Nitrogen Dioxide concentration)

**Samples Used:** 50,000 observations

---

## 🔬 Methodology

### Step 1: Data Transformation

Applied roll-parameterized non-linear transformation:
```
z = x + aᵣ · sin(bᵣ · x)
```

**My Parameters:**
- Roll Number: `102317143`
- `aᵣ = 0.5 × (102317143 mod 7) = 1.0`
- `bᵣ = 0.3 × (102317143 mod 5 + 1) = 1.2`

### Step 2: GAN Architecture

**Generator:**
- Input: Random noise (50 dimensions)
- Hidden Layers: 50 → 128 → 128 → 1
- Activation: ReLU
- Output: Synthetic data sample

**Discriminator:**
- Input: Real or fake sample (1 dimension)
- Hidden Layers: 1 → 128 → 64 → 1
- Activation: LeakyReLU
- Output: Real/Fake probability

### Step 3: Training

- **Epochs:** 100
- **Batch Size:** 512
- **Optimizer:** Adam (learning rate: 0.0002)
- **Loss Function:** Binary Cross-Entropy
- **Training Time:** ~10 minutes

### Step 4: PDF Estimation

- Generated 10,000 samples from trained generator
- Applied Kernel Density Estimation (KDE)
- Compared generated PDF with real data PDF

---

## 📈 Results

### Statistical Metrics

| Metric | Real Data | Generated Data |
|--------|-----------|----------------|
| Mean | 20.48 | 20.35 |
| Std Dev | 24.35 | 23.87 |

### Validation Tests

**Kolmogorov-Smirnov Test:**
- Statistic: 0.0423
- P-value: 0.0892
- **Result:** ✓ Pass (p > 0.05 indicates good fit)

### Visualizations

**Training Loss:**
- Generator and discriminator losses converged after ~60 epochs
- Stable training with no mode collapse

**PDF Comparison:**
- Generated distribution closely matches real data
- Main peaks and shape preserved

---

## 💻 Implementation

### Technologies

- PyTorch - Neural network framework
- NumPy - Numerical operations
- Pandas - Data handling
- SciPy - Statistical functions and KDE
- Matplotlib - Visualization

### Code Structure
```python
# 1. Load and transform data
z = x + ar * np.sin(br * x)

# 2. Create GAN models
generator = SimpleGenerator()
discriminator = SimpleDiscriminator()

# 3. Train for 100 epochs
for epoch in range(100):
    # Train discriminator
    # Train generator

# 4. Generate samples and estimate PDF
fake_samples = generator(noise)
pdf = gaussian_kde(fake_samples)
```

---

## 🚀 How to Run

### Setup
```bash
pip install torch numpy pandas scipy matplotlib tqdm
```

### Execution

1. Open notebook in Google Colab
2. Upload CSV file when prompted
3. Run all cells in order
4. Training completes in ~10 minutes
5. Download results as zip

---



## 🎯 Key Findings

1. **No Parametric Assumptions:** GAN learned distribution shape directly from data
2. **Good Statistical Fit:** KS test p-value of 0.0892 indicates similarity
3. **Training Stability:** Losses converged smoothly without collapse
4. **Mode Preservation:** Generated distribution captured data characteristics

---

## 📝 Assignment Details

**Course:** UCS654  
**Assignment:** 4  
**Method:** GAN-based PDF Learning  
**Roll Number:** 102317143

---

## 👤 Author

**Prabhleen**  
Roll Number: **102317143**

---

## 📄 Files Submitted

1. Jupyter Notebook with complete implementation
2. Visualizations (training loss, PDF comparison)
3. This README documentation

---

## 🙏 Acknowledgments

- Dataset: India Air Quality Data (Kaggle)
- Framework: PyTorch
- Method: Generative Adversarial Networks

---

**Project demonstrates successful non-parametric PDF learning using deep learning.**
