# 📈 VietAI Learning Module 6: Time Series Forecasting

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/PhucVinhDEV/VietAI-Learning-Module6)
[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Latest-orange?logo=pytorch)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)](https://streamlit.io/)

Repository cho module học tập về Time Series Forecasting, bao gồm implementation của **GRU model** dự đoán giá cổ phiếu FPT với giao diện Streamlit tương tác.

**Team**: VietAI-Learning  
**Course**: AI VIET NAM - AI COURSE 2025

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Project Structure](#-project-structure)
- [FPT GRU Stock Prediction](#-fpt-gru-stock-prediction)
  - [Train Model](#1-train-model)
  - [Run Streamlit App](#2-run-streamlit-app)
  - [Usage Guide](#3-usage-guide)
- [Deployment](#-deployment)
- [Configuration](#-configuration)
- [Reproducibility](#-reproducibility)
- [Troubleshooting](#-troubleshooting)
- [Team](#-team)
- [References](#-references)

---

## 🎯 Overview

Project này implement **GRU (Gated Recurrent Unit)** model cho time series forecasting, được áp dụng để dự đoán giá cổ phiếu FPT. Project bao gồm:

- ✅ **Modular codebase** với cấu trúc rõ ràng
- ✅ **PyTorch-based** model training
- ✅ **Streamlit web app** với giao diện tương tác
- ✅ **Checkpoint system** để save/load model
- ✅ **Reproducible training** với random seed
- ✅ **Comprehensive evaluation** metrics (MAPE, MSE, etc.)

---

## ✨ Features

### Model Features

- **GRU Architecture**: Multi-layer GRU với dropout regularization
- **Early Stopping**: Tự động dừng training khi validation loss không cải thiện
- **Data Preprocessing**: Log transformation và StandardScaler normalization
- **Future Prediction**: Dự đoán giá cổ phiếu cho N ngày tới

### App Features

- **Data Visualization**:
  - Interactive price charts với date range selection
  - Moving average overlay
  - Data preview với customizable rows
- **Model Management**:
  - Load trained checkpoints
  - View model metrics và architecture
  - Training curves visualization
- **Prediction**:
  - Generate future predictions
  - Visualize historical vs predicted prices
  - Download predictions as CSV

---

## 🚀 Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/PhucVinhDEV/VietAI-Learning-Module6.git
cd VietAI-Learning-Module6
```

### 2. Create Virtual Environment

```bash
# Create venv
python -m venv venv

# Activate
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare Data

Đảm bảo có file dữ liệu:

- `data/raw/FPT_train.csv`

### 5. Train Model

```bash
python scripts/train_fpt_gru.py
```

Model sẽ được lưu tại: `models/fpt_gru/best_model.pt`

### 6. Run Streamlit App

```bash
streamlit run streamlit_app.py
```

App sẽ mở tại: `http://localhost:8501`

---

## 📦 Installation

### Prerequisites

- Python 3.8+
- pip
- (Optional) CUDA-capable GPU for faster training

### Step-by-Step Installation

```bash
# 1. Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate    # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify installation
python -c "import torch; import streamlit; print('✓ Installation successful')"
```

### Dependencies

Key packages:

- `torch` - Deep learning framework
- `streamlit` - Web app framework
- `pandas`, `numpy` - Data manipulation
- `scikit-learn` - Data preprocessing
- `matplotlib` - Visualization
- `tqdm` - Progress bars

Xem đầy đủ trong `requirements.txt`.

---

## 📁 Project Structure

```
VietAI-Learning-Module6/
├── README.md                 # This file
├── DEPLOY.md                 # Deployment guide
├── requirements.txt          # Python dependencies
├── setup.py                  # Package setup
│
├── data/
│   ├── raw/
│   │   └── FPT_train.csv     # FPT stock data
│   └── processed/            # Processed data (if any)
│
├── src/
│   ├── data/
│   │   ├── loader.py         # Data loading utilities
│   │   └── dataset.py        # PyTorch Dataset class
│   ├── model/
│   │   └── gru_model.py      # GRU model definition
│   ├── training/
│   │   └── trainer.py        # Training loop with early stopping
│   ├── utils/
│   │   ├── config.py         # Configuration & seed setting
│   │   ├── checkpoint.py     # Save/load model checkpoints
│   │   └── predict.py        # Prediction utilities
│   └── streamlit_app.py     # Streamlit application
│
├── scripts/
│   └── train_fpt_gru.py     # Training script
│
├── models/
│   └── fpt_gru/
│       └── best_model.pt    # Trained model checkpoint
│
├── notebooks/
│   └── kaggle-final-version.ipynb  # Jupyter notebook
│
├── .streamlit/
│   └── config.toml          # Streamlit configuration
│
└── streamlit_app.py         # Entry point for Streamlit Cloud
```

---

## 📈 FPT GRU Stock Prediction

### 1. Train Model

**Bước quan trọng**: Train model trước khi chạy Streamlit app.

```bash
python scripts/train_fpt_gru.py
```

Script này sẽ:

1. Load data từ `data/raw/FPT_train.csv`
2. Prepare data (log transform, normalization)
3. Split train/validation sets
4. Initialize GRU model
5. Train với early stopping
6. Evaluate và tính MAPE
7. Save checkpoint vào `models/fpt_gru/best_model.pt`

**Output mẫu:**

```
============================================================
FPT GRU Model Training
============================================================

📊 Loading data...
✅ Data loaded: 1149 records

🔧 Creating datasets...
✅ Train samples: 999, Val samples: 120

🏗️  Creating model...
✅ Model created: 25089 parameters

🚀 Training model...
Training: 100%|████████| 45/45 [00:30<00:00, 1.48it/s, train_loss=0.0012, val_loss=0.0015]
Early stopping!

📊 Evaluating model...
✅ Validation MAPE: 2.45%
✅ Best Val Loss: 0.001456

💾 Saving checkpoint to: models/fpt_gru/best_model.pt
✅ Checkpoint saved to: models/fpt_gru/best_model.pt
```

### 2. Run Streamlit App

```bash
streamlit run streamlit_app.py
```

Hoặc từ thư mục `src/`:

```bash
streamlit run src/streamlit_app.py
```

### 3. Usage Guide

#### 📊 Tab 1: Data Overview

1. **Load Data**: Click button "Load Data" để load `FPT_train.csv`
2. **View Metrics**:
   - Total Records
   - Date Range
   - Current Price
3. **Visualization Options**:
   - **Date Range**: Chọn "All", "Last 6 months", "Last 1 year", "Last 2 years", hoặc "Last N days"
   - **Moving Average**: Toggle để hiển thị MA với window size tùy chỉnh
4. **Price Chart**: Biểu đồ giá đóng cửa theo thời gian
5. **Data Preview**: Xem preview dữ liệu với số rows có thể chỉnh

#### 📥 Tab 2: Load Model

1. **Load Checkpoint**: Click "Load Checkpoint" để load model từ `models/fpt_gru/best_model.pt`
2. **Model Metrics**:
   - Validation MAPE
   - Best Validation Loss
   - Final Train/Val Loss
3. **Model Architecture**:
   - Input/Output Length
   - Hidden Size
   - Number of Layers
   - Dropout Rate
4. **Training Configuration**:
   - Learning Rate
   - Batch Size
   - Number of Epochs
   - Device (CPU/GPU)
5. **Training Curves**: Biểu đồ training và validation loss

#### 🔮 Tab 3: Predict

1. **Generate Prediction**:
   - Chỉnh "Days to Predict" trong sidebar (10-200 days)
   - Click "Generate Prediction"
2. **Results**:
   - **Chart**: Biểu đồ Historical vs Predicted prices
   - **Metrics**:
     - Current Price
     - Predicted (Day 1)
     - Predicted (Final Day)
     - Total Change %
   - **Download**: Download predictions dưới dạng CSV

---

## 🚀 Deployment

### Deploy lên Streamlit Cloud

Xem hướng dẫn chi tiết trong [`DEPLOY.md`](DEPLOY.md).

**Tóm tắt nhanh:**

1. **Commit code lên GitHub**:

   ```bash
   git add .
   git commit -m "Prepare for deployment"
   git push origin main
   ```

2. **Deploy trên Streamlit Cloud**:

   - Truy cập [Streamlit Cloud](https://streamlit.io/cloud)
   - Đăng nhập bằng GitHub
   - Click "New app"
   - Chọn repository và branch
   - Main file: `streamlit_app.py`
   - Click "Deploy"

3. **Xử lý Model File**:
   - Option 1: Commit model vào Git (nếu < 100MB)
   - Option 2: Dùng Git LFS (cho model lớn)
   - Option 3: Download model khi deploy (xem `DEPLOY.md`)

---

## ⚙️ Configuration

Model được cấu hình trong `src/utils/config.py`:

```python
CONFIG = {
    "input_len": 30,              # Input sequence length
    "output_len": 1,               # Output length (single step)
    "total_predict_days": 100,     # Days to predict in future
    "batch_size": 32,
    "hidden_size": 64,             # GRU hidden units
    "num_layers": 2,               # Number of GRU layers
    "dropout": 0.2,                # Dropout rate
    "learning_rate": 1e-3,         # Learning rate
    "num_epochs": 80,              # Max epochs
    "early_stop_patience": 15,     # Early stopping patience
    "val_size": 120,               # Validation set size
    "device": "cuda" or "cpu",     # Auto-detect
}
```

**Có thể chỉnh:**

- `total_predict_days`: Trong Streamlit sidebar khi predict
- Các tham số khác: Sửa trong `src/utils/config.py` và train lại

---

## 🔒 Reproducibility

Model training sử dụng **random seed = 42** để đảm bảo kết quả giống nhau mỗi lần train.

```python
from src.utils import set_seed, SEED

# Set seed cho tất cả random generators
set_seed(SEED)  # Sets: random, numpy, torch, cuda, cudnn
```

**Seed được set cho:**

- Python `random` module
- NumPy random
- PyTorch random
- CUDA random (nếu có GPU)
- CuDNN deterministic mode

---

## 🐛 Troubleshooting

### Model không load được

**Lỗi**: `FileNotFoundError: Checkpoint not found`

**Giải pháp**:

```bash
# Train model trước
python scripts/train_fpt_gru.py

# Kiểm tra file tồn tại
ls models/fpt_gru/best_model.pt
```

### Data không load được

**Lỗi**: `FileNotFoundError: Không tìm thấy FPT_train.csv`

**Giải pháp**:

- Đảm bảo file `data/raw/FPT_train.csv` tồn tại
- App sẽ tự tìm file trong project structure
- Kiểm tra đường dẫn trong error message

### Import errors

**Lỗi**: `ModuleNotFoundError: No module named 'src'`

**Giải pháp**:

```bash
# Đảm bảo đang ở thư mục root của project
cd VietAI-Learning-Module6

# Cài đặt package (nếu có setup.py)
pip install -e .

# Hoặc chạy với PYTHONPATH
PYTHONPATH=. streamlit run streamlit_app.py
```

### CUDA/GPU issues

**Lỗi**: CUDA out of memory hoặc CUDA not available

**Giải pháp**:

```bash
# Kiểm tra CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Nếu không có GPU, model sẽ tự động dùng CPU
# Có thể force CPU trong config:
# "device": torch.device("cpu")
```

### Streamlit app không chạy

**Lỗi**: Port 8501 already in use

**Giải pháp**:

```bash
# Dùng port khác
streamlit run streamlit_app.py --server.port 8502

# Hoặc kill process đang dùng port 8501
# Windows:
netstat -ano | findstr :8501
taskkill /PID <PID> /F
```

---

## 👥 Team

**VietAI-Learning**

- Nguyễn Tấn Dũng
- Nguyễn Quốc Huy
- Ngô Ngọc Anh
- Trần Phúc Vinh
- Vũ Nguyệt Hằng

**Repository**: [https://github.com/PhucVinhDEV/VietAI-Learning-Module6](https://github.com/PhucVinhDEV/VietAI-Learning-Module6)

---

## 📚 References

### Papers

- [Are Transformers Effective for Time Series Forecasting?](https://arxiv.org/abs/2205.13504) - Paper về Linear models cho time series

### Documentation

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)

### Course

- **AI VIET NAM - AI COURSE 2025**

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Dataset: FPT stock data
- Course: AI VIET NAM - AI COURSE 2025
- Framework: PyTorch, Streamlit

---

## 📝 Additional Resources

- [`DEPLOY.md`](DEPLOY.md) - Chi tiết về deployment
- [`README_STREAMLIT.md`](README_STREAMLIT.md) - Hướng dẫn Streamlit app (nếu có)
- `notebooks/kaggle-final-version.ipynb` - Jupyter notebook với code gốc

---

**Happy Forecasting! 📈**

_Nếu có câu hỏi hoặc gặp vấn đề, vui lòng mở issue trên GitHub repository._
