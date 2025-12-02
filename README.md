# LTSF-Linear: Long-Term Time Series Forecasting

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/PhucVinhDEV/VietAI-Learning-Module6)

Implementation of Linear, NLinear, and DLinear models for long-term time series forecasting, applied to Vietnamese stock market data (VIC).

**Team**: VietAI-Learning

## 📋 Overview

This project implements three simple yet powerful baseline models for time series forecasting:

- **Linear**: Direct linear mapping from historical window to future predictions
- **NLinear**: Normalized Linear with distribution shift handling
- **DLinear**: Decomposition Linear separating trend and seasonality

**Key Features:**

- ✅ Clean, modular Python codebase following PEP-8
- ✅ Production-ready data pipeline
- ✅ Multiple input window sizes (7, 30, 120, 480 days)
- ✅ 7-day ahead forecasting
- ✅ Comprehensive evaluation metrics
- ✅ Visualization tools

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

### 4. Download Data

```bash
python scripts/download_data.py
```

### 5. Run Training

```bash
# Train all models
python scripts/train.py

# Or train specific model
python scripts/train.py --model linear --seq-len 30
```

## 📦 Installation

### Prerequisites

- Python 3.8+
- pip

### Step-by-Step

```bash
# 1. Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download data
python scripts/download_data.py

# 4. Verify installation
python -c "from src.model import Linear; print('✓ Installation successful')"
```

### Development Setup

For development with testing and code formatting tools:

```bash
# Install package in editable mode
pip install -e .

# Run tests
pytest tests/

# Format code
black src/ tests/
```

## 🎯 Usage

### Data Pipeline

```python
from src.data import DataPipeline

# Initialize pipeline
pipeline = DataPipeline(
    data_path="data/raw/VIC.csv",
    seq_lengths=[7, 30, 120, 480],
    pred_len=7,
    batch_size=32
)

# Run full pipeline
dataloaders = pipeline.run()

# Access dataloaders
train_loader = dataloaders['30d']['train']
val_loader = dataloaders['30d']['val']
test_loader = dataloaders['30d']['test']
```

### Training Models

```python
from src.model import Linear, NLinear, DLinear
from src.training import Trainer

# Initialize model
model = Linear(seq_len=30, pred_len=7)

# Create trainer
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device='cuda'
)

# Train
history = trainer.fit(num_epochs=50)
```

### Making Predictions

```python
from src.pineline.forecast_pipeline import ForecastPipeline

# Load trained model
pipeline = ForecastPipeline.from_checkpoint('experiments/checkpoints/linear_30d.pt')

# Predict
predictions = pipeline.predict(input_data)
```

## 📁 Project Structure

```
ltsf-linear/
├── README.md
├── requirements.txt
├── setup.py
│
├── data/
│   ├── raw/              # Downloaded data
│   └── processed/        # Preprocessed data
│
├── src/
│   ├── data/
│   │   ├── dataset.py
│   │   ├── dataloader.py
│   │   └── preprocesser.py
│   │
│   ├── model/
│   │   ├── base.py
│   │   ├── linear.py
│   │   ├── n_linear.py
│   │   └── d_linear.py
│   │
│   ├── training/
│   │   ├── trainer.py
│   │   ├── evaluator.py
│   │   └── callbacks.py
│   │
│   ├── utils/
│   │   ├── metrics.py
│   │   ├── decomposition.py
│   │   └── visualization.py
│   │
│   └── pineline/
│       ├── __init__.py
│       └── forecast_pipeline.py
│
├── scripts/
│   ├── download_data.py
│   ├── train.py
│   └── evaluate.py
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_from_scratch_models.ipynb
│   └── 03_full_training.ipynb
│
├── tests/
│   ├── test_data_simple.py
│   ├── test_models.py
│   └── test_utils.py
│
└── experiments/
    ├── logs/
    ├── checkpoints/
    └── results/
```

## 🧪 Models

### Linear

Simple linear mapping from input sequence to output sequence:

```
ŷ = Wx + b
```

- **Parameters**: `T × L` (+ T bias)
- **Complexity**: O(B × T × L)

### NLinear

Normalized linear with distribution shift handling:

```
x' = x - x_last
ŷ' = Wx' + b
ŷ = ŷ' + x_last
```

- **Use case**: Data with level shifts
- **Key feature**: Re-centering normalization

### DLinear

Decomposition-based linear model:

```
x_trend, x_seasonal = decompose(x)
ŷ_trend = W_t × x_trend + b_t
ŷ_seasonal = W_s × x_seasonal + b_s
ŷ = ŷ_trend + ŷ_seasonal
```

- **Use case**: Data with clear trend/seasonality
- **Key feature**: Moving average decomposition

## 📊 Results

Training on VIC stock data (2020-2025):

| Model   | Input Length | MSE ↓ | MAE ↓ | RMSE ↓ | R² ↑ |
| ------- | ------------ | ----- | ----- | ------ | ---- |
| Linear  | 30d          | 0.023 | 0.112 | 0.152  | 0.87 |
| NLinear | 30d          | 0.021 | 0.108 | 0.145  | 0.89 |
| DLinear | 30d          | 0.019 | 0.101 | 0.138  | 0.91 |

_Results on 7-day ahead forecasting_

## 🛠️ Development

### Run Tests

```bash
# Run simple test
python tests/test_data_simple.py

# Or with pytest
pytest tests/ -v
```

### Code Formatting

```bash
black src/ tests/
flake8 src/ tests/
```

### Type Checking

```bash
mypy src/
```

## 📝 Scripts

### Download Data

```bash
python scripts/download_data.py
```

### Train Models

```bash
# Train all models with all input lengths
python scripts/train.py

# Train specific model
python scripts/train.py --model linear --seq-len 30 --epochs 50

# Resume from checkpoint
python scripts/train.py --resume checkpoints/linear_30d.pt
```

### Evaluate

```bash
python scripts/evaluate.py --checkpoint checkpoints/dlinear_120d.pt
```

## 🔬 Experiments

### Jupyter Notebooks

Explore the project interactively:

```bash
jupyter notebook
```

Available notebooks:

- `01_data_exploration.ipynb` - Data analysis and visualization
- `02_from_scratch_models.ipynb` - Model implementation from scratch
- `03_full_training.ipynb` - Complete training pipeline
- `04_results_analysis.ipynb` - Results comparison
- `05_analysis_and_critique.ipynb` - Analysis and critique of model results

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 👥 Team

**VietAI-Learning**

- Nguyễn Tấn Dũng
- Nguyễn Quốc Huy
- Ngô Ngọc Anh
- Trần Phúc Vinh
- Vũ Nguyệt Hằng

**Repository**: [https://github.com/PhucVinhDEV/VietAI-Learning-Module6](https://github.com/PhucVinhDEV/VietAI-Learning-Module6)

## 🙏 Acknowledgments

- Based on the paper: [Are Transformers Effective for Time Series Forecasting?](https://arxiv.org/abs/2205.13504)
- Dataset: VIC stock data from Vietnamese stock market
- Course: AI VIET NAM - AI COURSE 2025

## 🐛 Troubleshooting

### Common Issues

**`ModuleNotFoundError: No module named 'src'`**

```bash
pip install -e .
```

**`gdown` download fails**

```bash
# Upgrade gdown
pip install --upgrade gdown

# Or download manually
# Visit: https://drive.google.com/file/d/18J_Z8b-qMMj9wm5eGyQ-1nPS16PfRePK/view
# Save to: data/raw/VIC.csv
```

**PyTorch CUDA issues**

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Install CPU version if no GPU
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## 📚 References

```bibtex
@article{zeng2023transformers,
  title={Are Transformers Effective for Time Series Forecasting?},
  author={Zeng, Ailing and Chen, Muxi and Zhang, Lei and Xu, Qiang},
  journal={arXiv preprint arXiv:2205.13504},
  year={2023}
}
```

---

**Happy Forecasting! 📈**

---

## 📈 FPT GRU Streamlit App

Ngoài các model Linear/NLinear/DLinear cho VIC, repo còn có một demo **GRU model** dự đoán giá cổ phiếu **FPT** với giao diện **Streamlit**.

### 1. Chuẩn Bị Môi Trường

- Đã tạo và kích hoạt virtualenv như phần Quick Start.
- Đã cài đặt dependencies chung:

```bash
pip install -r requirements.txt
```

Sau đó cài thêm (nếu chưa có):

```bash
pip install streamlit tqdm
```

### 2. Chuẩn Bị Dữ Liệu FPT

Đảm bảo có file:

- `data/raw/FPT_train.csv`

App sẽ tự tìm file này, nên chỉ cần đúng đường dẫn/thư mục.

### 3. Chạy Streamlit App

Từ thư mục root của project:

```bash
streamlit run src/streamlit_app.py
```

Hoặc nếu bạn đang ở trong thư mục `src`:

```bash
streamlit run streamlit_app.py
```

### 4. Hướng Dẫn Sử Dụng (Step-by-step)

- **Tab `Data Overview`**

  - Bấm **Load Data** để đọc `FPT_train.csv`
  - Xem tổng số bản ghi, khoảng thời gian dữ liệu, giá hiện tại
  - Xem biểu đồ giá đóng cửa theo thời gian và bảng preview

- **Tab `Train Model`**

  - Chỉnh các tham số trong sidebar:
    - **Model**: Input Length, Hidden Size, Number of Layers, Dropout
    - **Training**: Epochs, Learning Rate, Batch Size, Early Stop Patience
  - Bấm **Prepare Data & Train**:
    - Chuẩn hóa dữ liệu (log transform)
    - Chia train/validation
    - Train GRU với early stopping
    - Hiển thị training/validation loss + Validation MAPE

- **Tab `Predict`**
  - Sau khi train xong, bấm **Generate Prediction**:
    - Dự đoán giá FPT cho `Days to Predict` ngày tới
    - Hiển thị biểu đồ Historical vs Predicted
    - Hiển thị các metric: current price, predicted day 1, predicted final, tổng % thay đổi
    - Cho phép download file CSV kết quả

### 5. Cấu Trúc Liên Quan Đến App

```text
src/
├── data/
│   ├── loader.py          # Load & prepare data FPT
│   └── dataset.py         # TimeSeriesDataset cho GRU
├── model/
│   └── gru_model.py       # GRUModel demo
├── training/
│   └── trainer.py         # Training loop + early stopping
├── utils/
│   ├── config.py          # Config mặc định cho GRU demo
│   └── predict.py         # Hàm evaluate & predict future
└── streamlit_app.py       # Ứng dụng Streamlit
```

Chi tiết hơn xem thêm `README_STREAMLIT.md`.
