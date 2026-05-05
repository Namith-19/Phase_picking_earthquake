# Phase Picking in Earthquake Signals using Deep Learning

## 🔍 Project Overview
This project focuses on **automatic earthquake detection and seismic phase picking (P-wave and S-wave)** using a deep learning framework built on convolutional and temporal architectures.

Accurate phase picking is a critical step in seismic analysis, enabling:
- Precise earthquake localization  
- Faster early warning systems  
- Improved seismic monitoring in noisy environments  

The model is designed to process **raw seismic waveforms** and:
1. Detect whether an earthquake signal is present  
2. Identify and localize P and S wave arrival times  

The system is based on a **Residual Causal Dilated CNN–TCN architecture**, which captures:
- Local waveform features (via CNN)  
- Long-range temporal dependencies (via dilated TCN)  

This results in:
- High detection accuracy  
- Strong robustness to noise  
- Low inference latency suitable for real-time systems  

---

## ⚙️ Technologies & Libraries Used

### 🧠 Core Frameworks
- Python 3.x  
- PyTorch  

### 📊 Data Handling & Processing
- NumPy  
- Pandas   

### 🌍 Seismic Data Tools
- SeisBench (STEAD dataset handling)    

### 📈 Visualization & Evaluation
- Matplotlib  
- Scikit-learn  

### ⚡ Utilities
- tqdm  

---

## 🧱 Model Architecture Summary
- Input: 3-channel seismic waveform (Z, N, E) + STA/LTA input  
- 1×1 Convolution (channel projection: 3 → 32)  
- Residual Causal Dilated TCN:
  - Kernel size: 5  
  - Dilations: [1, 2, 4, 8, 16, 32, 64, 128]  
- Residual connections + normalization + dropout  
- Global Average Pooling  
- Fully Connected Layer  

### Outputs:
- Earthquake detection probability  
- P-wave and S-wave phase predictions  

---

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/Namith-19/Phase_picking_earthquake.git

# Navigate into the project directory
cd Phase_picking_earthquake

# Create virtual environment (recommended)
python -m venv venv

# Activate environment
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
