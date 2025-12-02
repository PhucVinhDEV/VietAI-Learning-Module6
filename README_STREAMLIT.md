# FPT Stock Prediction - Streamlit App

Ứng dụng Streamlit để dự đoán giá cổ phiếu FPT sử dụng GRU Model.

## 📋 Yêu cầu

Cài đặt các dependencies:

```bash
pip install streamlit torch pandas numpy matplotlib scikit-learn tqdm
```

## 🚀 Chạy ứng dụng

Từ thư mục root của project:

```bash
streamlit run src/streamlit_app.py
```

Hoặc nếu bạn đang ở trong thư mục `src`:

```bash
streamlit run streamlit_app.py
```

## 📖 Hướng dẫn sử dụng

### 1. Tab "Data Overview"

- Click nút **"Load Data"** để load dữ liệu FPT từ file CSV
- Xem overview về dữ liệu: số records, date range, current price
- Xem biểu đồ giá cổ phiếu theo thời gian
- Xem preview dữ liệu

### 2. Tab "Train Model"

- Điều chỉnh các tham số model và training ở sidebar
- Click **"Prepare Data & Train"** để:
  - Chuẩn bị dữ liệu (log transform, split train/val)
  - Train model với các tham số đã chọn
  - Hiển thị training curves và validation results
  - Hiển thị MAPE (Mean Absolute Percentage Error)

### 3. Tab "Predict"

- Sau khi train xong, click **"Generate Prediction"** để:
  - Dự đoán giá cổ phiếu trong tương lai
  - Xem biểu đồ so sánh historical vs predicted
  - Xem các metrics: current price, predicted prices, % change
  - Download kết quả dự đoán dưới dạng CSV

## ⚙️ Configuration

Các tham số có thể điều chỉnh trong sidebar:

### Model Parameters

- **Input Length**: Độ dài input sequence (10-60)
- **Hidden Size**: Kích thước hidden layer (32-128)
- **Number of Layers**: Số layers GRU (1-4)
- **Dropout**: Dropout rate (0.0-0.5)

### Training Parameters

- **Epochs**: Số epochs training (10-200)
- **Learning Rate**: Learning rate (1e-4 đến 1e-2)
- **Batch Size**: Batch size (16-64)
- **Early Stop Patience**: Số epochs không cải thiện trước khi dừng (5-30)

### Prediction Parameters

- **Days to Predict**: Số ngày cần dự đoán (10-200)
- **Validation Size**: Kích thước validation set (50-200)

## 📁 Cấu trúc Code

```
src/
├── data/
│   ├── loader.py          # Load và prepare data
│   └── dataset.py         # TimeSeriesDataset class
├── model/
│   └── gru_model.py       # GRUModel class
├── training/
│   └── trainer.py         # Training utilities
├── utils/
│   ├── config.py          # Configuration
│   └── predict.py         # Prediction utilities
└── streamlit_app.py       # Streamlit app chính
```

## 🔧 Troubleshooting

### Lỗi "Không tìm thấy FPT_train.csv"

- Đảm bảo file `data/raw/FPT_train.csv` tồn tại
- Hoặc chỉnh sửa path trong `src/data/loader.py`

### Lỗi import

- Đảm bảo bạn đang chạy từ root project
- Hoặc thêm project root vào PYTHONPATH

### Model training chậm

- Giảm số epochs hoặc batch size
- Sử dụng GPU nếu có (tự động detect)

## 📝 Notes

- Model sử dụng soft clipping để tránh giá trị quá lớn/nhỏ
- Validation MAPE thường khoảng 8-12%
- Predictions được clip trong range 80%-125% của giá hiện tại
