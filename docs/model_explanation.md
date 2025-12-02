# Giải Thích Chi Tiết Các Model Time Series Forecasting

## 📋 Tổng Quan

Project này implement 3 model đơn giản nhưng mạnh mẽ cho time series forecasting:
- **Linear**: Model đơn giản nhất, baseline
- **NLinear**: Xử lý distribution shift (thay đổi mức độ)
- **DLinear**: Tách trend và seasonality

Tất cả đều kế thừa từ `BaseForecastModel` để đảm bảo interface nhất quán.

---

## 🏗️ Kiến Trúc Tổng Thể: BaseForecastModel

### Mục đích
`BaseForecastModel` là abstract base class đảm bảo tất cả models có:
- Interface nhất quán: `forward()`, `get_model_info()`
- Quản lý tham số: `seq_len` (input length), `pred_len` (output length)

### Code Structure

```python
class BaseForecastModel(nn.Module, ABC):
    def __init__(self, seq_len: int, pred_len: int):
        self.seq_len = seq_len  # Ví dụ: 30 (nhìn lại 30 ngày)
        self.pred_len = pred_len  # Ví dụ: 7 (dự đoán 7 ngày)
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Mỗi model implement riêng
        pass
```

### Input/Output Format
- **Input**: `(batch_size, seq_len, 1)` hoặc `(batch_size, seq_len)`
  - Ví dụ: `(32, 30, 1)` = 32 samples, mỗi sample có 30 giá trị
- **Output**: `(batch_size, pred_len)`
  - Ví dụ: `(32, 7)` = 32 predictions, mỗi prediction có 7 giá trị tương lai

---

## 1️⃣ Linear Model - Model Đơn Giản Nhất

### Ý Tưởng
**Linear** là model đơn giản nhất: ánh xạ trực tiếp từ input sequence sang output sequence bằng một phép nhân ma trận.

### Công Thức Toán Học

```
ŷ = Wx + b
```

Trong đó:
- `x`: Input sequence `(seq_len,)` - ví dụ: giá cổ phiếu 30 ngày qua
- `W`: Weight matrix `(pred_len, seq_len)` - ma trận học được
- `b`: Bias vector `(pred_len,)` - bias học được
- `ŷ`: Output predictions `(pred_len,)` - dự đoán 7 ngày tới

### Code Implementation

```python
class Linear(BaseForecastModel):
    def __init__(self, seq_len: int, pred_len: int):
        super().__init__(seq_len, pred_len)
        # Một linear layer duy nhất
        self.linear = nn.Linear(seq_len, pred_len)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.squeeze(-1)  # (batch, seq_len, 1) -> (batch, seq_len)
        return self.linear(x)  # (batch, seq_len) -> (batch, pred_len)
```

### Ví Dụ Minh Họa

```
Input:  [100, 102, 101, 103, 105, ...]  (30 giá trị)
        ↓
        [W matrix transformation]
        ↓
Output: [107, 108, 109, 110, 111, 112, 113]  (7 giá trị dự đoán)
```

### Ưu Điểm
- ✅ Đơn giản, dễ hiểu
- ✅ Ít tham số (30 × 7 = 210 weights + 7 bias = 217 parameters)
- ✅ Training nhanh
- ✅ Baseline tốt để so sánh

### Nhược Điểm
- ❌ Không xử lý được distribution shift (thay đổi mức độ)
- ❌ Không tách được trend/seasonality
- ❌ Giả định dữ liệu stationary (ổn định)

### Khi Nào Dùng?
- Baseline để so sánh với các model phức tạp hơn
- Dữ liệu ổn định, không có thay đổi mức độ lớn
- Cần model đơn giản, nhanh

---

## 2️⃣ NLinear Model - Xử Lý Distribution Shift

### Vấn Đề Linear Model Gặp Phải

Giả sử giá cổ phiếu có **level shift** (thay đổi mức độ):
```
Ngày 1-30:  Giá dao động quanh 100
Ngày 31-60: Giá dao động quanh 150  ← Level shift!
```

Linear model sẽ gặp khó khăn vì nó học trên dữ liệu quanh mức 100, nhưng phải dự đoán quanh mức 150.

### Ý Tưởng NLinear

**NLinear** giải quyết bằng cách **normalize** input về mức 0, sau đó **denormalize** output về mức ban đầu.

### Công Thức Toán Học

```
1. Normalize:   x' = x - x_last
2. Predict:     ŷ' = Wx' + b
3. Denormalize: ŷ = ŷ' + x_last
```

Trong đó:
- `x_last`: Giá trị cuối cùng của input sequence (điểm tham chiếu)
- `x'`: Input đã normalize (trừ đi x_last)
- `ŷ'`: Prediction trên normalized input
- `ŷ`: Final prediction (cộng lại x_last)

### Code Implementation

```python
class NLinear(BaseForecastModel):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.squeeze(-1)
        
        # Lấy giá trị cuối cùng
        x_last = x[:, -1]  # (batch_size,)
        
        # Normalize: trừ đi x_last
        x_normalized = x - x_last.unsqueeze(-1)  # (batch, seq_len)
        
        # Predict trên normalized input
        y_pred_normalized = self.linear(x_normalized)  # (batch, pred_len)
        
        # Denormalize: cộng lại x_last
        y_pred = y_pred_normalized + x_last.unsqueeze(-1)  # (batch, pred_len)
        
        return y_pred
```

### Ví Dụ Minh Họa

```
Input:  [100, 102, 101, 103, 105, 107, 109]  (x_last = 109)
        ↓ Normalize (trừ 109)
        [-9, -7, -8, -6, -4, -2, 0]
        ↓ Linear transformation
        [-2, -1, 0, 1, 2, 3, 4]  (ŷ')
        ↓ Denormalize (cộng 109)
Output: [107, 108, 109, 110, 111, 112, 113]  (ŷ)
```

### Tại Sao NLinear Hoạt Động Tốt?

1. **Re-centering**: Normalize về mức 0 giúp model học pattern thay vì absolute values
2. **Adaptive**: Tự động điều chỉnh theo level shift bằng cách dùng `x_last` làm reference
3. **Simple**: Vẫn chỉ là một linear layer, nhưng thông minh hơn

### Ưu Điểm
- ✅ Xử lý được distribution shift
- ✅ Tự động adapt với level changes
- ✅ Vẫn đơn giản (chỉ thêm normalize/denormalize)
- ✅ Thường tốt hơn Linear trên dữ liệu thực tế

### Nhược Điểm
- ❌ Vẫn không tách được trend/seasonality
- ❌ Phụ thuộc vào giá trị cuối cùng (nếu outlier thì ảnh hưởng)

### Khi Nào Dùng?
- Dữ liệu có level shifts (thay đổi mức độ)
- Stock prices, exchange rates (thường có level shifts)
- Muốn cải thiện Linear mà không phức tạp quá

---

## 3️⃣ DLinear Model - Tách Trend và Seasonality

### Vấn Đề Với Linear và NLinear

Cả hai model trên đều xử lý time series như một chuỗi đơn giản, không tách được:
- **Trend**: Xu hướng dài hạn (tăng/giảm)
- **Seasonality**: Chu kỳ lặp lại (tuần, tháng, năm)

### Ý Tưởng DLinear

**DLinear** tách input thành 2 components:
1. **Trend**: Xu hướng dài hạn (dùng moving average)
2. **Seasonal**: Phần còn lại sau khi trừ trend

Sau đó áp dụng **2 linear layers riêng biệt** cho mỗi component, rồi cộng lại.

### Công Thức Toán Học

```
1. Decompose:    x_trend, x_seasonal = decompose(x)
2. Predict trend:    ŷ_trend = W_t × x_trend + b_t
3. Predict seasonal: ŷ_seasonal = W_s × x_seasonal + b_s
4. Combine:      ŷ = ŷ_trend + ŷ_seasonal
```

### Decomposition Process

```python
def decompose_trend_seasonal(x, kernel_size=25):
    # Trend = Moving average (làm mịn)
    trend = moving_average(x, kernel_size)
    
    # Seasonal = Original - Trend
    seasonal = x - trend
    
    return trend, seasonal
```

**Moving Average** là gì?
- Lấy trung bình của một cửa sổ (window) để làm mịn dữ liệu
- Ví dụ: kernel_size=25 → lấy trung bình 25 điểm
- Trend = phần mịn, dài hạn
- Seasonal = phần còn lại, ngắn hạn, có chu kỳ

### Code Implementation

```python
class DLinear(BaseForecastModel):
    def __init__(self, seq_len: int, pred_len: int, kernel_size: int = 25):
        super().__init__(seq_len, pred_len)
        self.kernel_size = kernel_size
        
        # 2 linear layers riêng biệt
        self.linear_trend = nn.Linear(seq_len, pred_len)
        self.linear_seasonal = nn.Linear(seq_len, pred_len)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.squeeze(-1)
        
        # Tách thành trend và seasonal
        x_trend, x_seasonal = decompose_trend_seasonal(x, self.kernel_size)
        
        # Predict riêng từng component
        y_trend = self.linear_trend(x_trend)
        y_seasonal = self.linear_seasonal(x_seasonal)
        
        # Cộng lại
        y_pred = y_trend + y_seasonal
        
        return y_pred
```

### Ví Dụ Minh Họa

```
Input:  [100, 102, 98, 105, 103, 101, 107, ...]
        ↓ Decompose
        ↓
Trend:  [101, 101.5, 102, 102.5, 103, ...]  (moving average, mịn)
Seasonal: [-1, 0.5, -4, 2.5, 0, -2, 4, ...]  (original - trend)
        ↓
        ↓ Linear transformation riêng
        ↓
ŷ_trend:    [104, 104.5, 105, 105.5, ...]
ŷ_seasonal: [-0.5, 1, -3, 2, ...]
        ↓ Combine
Output: [103.5, 105.5, 102, 107.5, ...]
```

### Tại Sao DLinear Hoạt Động Tốt?

1. **Separation of Concerns**: Tách trend và seasonal giúp model học từng pattern riêng
2. **Trend Handling**: Moving average capture xu hướng dài hạn tốt
3. **Seasonal Patterns**: Phần seasonal capture chu kỳ ngắn hạn
4. **Combination**: Cộng lại cho prediction toàn diện

### Ưu Điểm
- ✅ Tách được trend và seasonality
- ✅ Hiệu quả với dữ liệu có pattern rõ ràng
- ✅ Thường tốt nhất trong 3 model
- ✅ Vẫn đơn giản (chỉ 2 linear layers)

### Nhược Điểm
- ❌ Cần chọn `kernel_size` phù hợp (default 25)
- ❌ Moving average có thể làm mất thông tin nếu kernel_size quá lớn
- ❌ Phức tạp hơn Linear và NLinear một chút

### Khi Nào Dùng?
- Dữ liệu có trend rõ ràng (tăng/giảm dài hạn)
- Dữ liệu có seasonality (chu kỳ tuần, tháng)
- Muốn model tốt nhất trong 3 model
- Stock prices, sales data, temperature data

---

## 📊 So Sánh 3 Models

| Tiêu Chí | Linear | NLinear | DLinear |
|----------|--------|---------|---------|
| **Độ phức tạp** | Đơn giản nhất | Đơn giản | Phức tạp hơn một chút |
| **Số parameters** | 217 (30×7+7) | 217 | 434 (2×217) |
| **Xử lý level shift** | ❌ | ✅ | ❌ |
| **Tách trend/seasonal** | ❌ | ❌ | ✅ |
| **Tốc độ training** | Nhanh nhất | Nhanh | Nhanh |
| **Hiệu suất** | Baseline | Tốt hơn Linear | Thường tốt nhất |
| **Use case** | Baseline, stable data | Data với level shifts | Data có trend/seasonal |

---

## 🎯 Kết Luận và Khuyến Nghị

### Thứ Tự Thử Nghiệm

1. **Bắt đầu với Linear**: Baseline để so sánh
2. **Nếu có level shifts → NLinear**: Cải thiện đáng kể
3. **Nếu có trend/seasonal → DLinear**: Thường tốt nhất

### Lưu Ý Quan Trọng

- **Không phải model phức tạp = tốt hơn**: 3 model này đơn giản nhưng rất mạnh
- **Phụ thuộc vào dữ liệu**: Mỗi model phù hợp với loại dữ liệu khác nhau
- **Experiment**: Thử cả 3 và so sánh trên validation set

### Next Steps

Sau khi hiểu models, bạn có thể:
1. **Training**: Dùng `Trainer` class để train models
2. **Evaluation**: Dùng `Evaluator` class để đánh giá
3. **Experiment**: Thử các `seq_len` khác nhau (7, 30, 120, 480)
4. **Improve**: Cải thiện dựa trên insights từ results

---

## 📚 Tài Liệu Tham Khảo

- Paper gốc: "Are Transformers Effective for Time Series Forecasting?"
- Code implementation: `src/model/`
- Test examples: `tests/test_models_simple.py`

