# Giải Thích: `import torch.nn as nn`

## 🔍 `torch.nn` là gì?

`torch.nn` là **Neural Network module** của PyTorch - một thư viện chứa các building blocks (khối xây dựng) để tạo neural networks.

### Tại sao dùng `as nn`?

```python
import torch.nn as nn  # Thay vì import torch.nn
```

**Lý do:**
- **Ngắn gọn**: `nn.Linear` thay vì `torch.nn.Linear`
- **Convention**: Cộng đồng PyTorch dùng `nn` như một chuẩn
- **Dễ đọc**: Code ngắn gọn, dễ hiểu hơn

---

## 📦 Các Thành Phần Quan Trọng Trong `torch.nn`

### 1. `nn.Module` - Base Class cho Tất Cả Models

**Là gì?**
- Base class mà tất cả neural network layers/models phải kế thừa
- Cung cấp các tính năng: parameter management, device placement, training/eval mode

**Ví dụ trong project:**

```python
import torch.nn as nn

class BaseForecastModel(nn.Module, ABC):  # ← Kế thừa từ nn.Module
    def __init__(self, seq_len: int, pred_len: int):
        super().__init__()  # ← Gọi constructor của nn.Module
        self.seq_len = seq_len
        self.pred_len = pred_len
```

**Tại sao cần `nn.Module`?**
- ✅ Tự động quản lý parameters (weights, biases)
- ✅ Hỗ trợ `.to(device)` để chuyển model lên GPU
- ✅ Hỗ trợ `.train()` và `.eval()` modes
- ✅ Hỗ trợ save/load models

**Ví dụ sử dụng:**

```python
model = Linear(seq_len=30, pred_len=7)

# Xem tất cả parameters
for param in model.parameters():
    print(param.shape)

# Chuyển lên GPU
model = model.to('cuda')

# Chuyển sang eval mode (tắt dropout, batch norm updates)
model.eval()
```

---

### 2. `nn.Linear` - Linear Layer

**Là gì?**
- Một fully-connected layer (dense layer) thực hiện phép toán: `y = Wx + b`
- Đây là layer cơ bản nhất trong neural networks

**Công thức:**
```
output = input × weight^T + bias
```

**Ví dụ trong project:**

```python
import torch.nn as nn

class Linear(BaseForecastModel):
    def __init__(self, seq_len: int, pred_len: int):
        super().__init__(seq_len, pred_len)
        
        # Tạo một linear layer
        # Input size: seq_len (ví dụ: 30)
        # Output size: pred_len (ví dụ: 7)
        self.linear = nn.Linear(seq_len, pred_len)
        # ↑
        # Tương đương với:
        # - Weight matrix W: shape (pred_len, seq_len) = (7, 30)
        # - Bias vector b: shape (pred_len,) = (7,)
```

**Giải thích chi tiết:**

```python
# Khi bạn viết:
self.linear = nn.Linear(30, 7)

# PyTorch tự động tạo:
# - W (weight): tensor shape (7, 30) - khởi tạo ngẫu nhiên
# - b (bias): tensor shape (7,) - khởi tạo ngẫu nhiên

# Khi forward:
x = torch.randn(8, 30)  # (batch_size=8, input_size=30)
y = self.linear(x)      # (batch_size=8, output_size=7)

# Thực chất là:
# y = x @ W.T + b
#   = (8, 30) @ (30, 7) + (7,)
#   = (8, 7)
```

**Ví dụ cụ thể:**

```python
import torch
import torch.nn as nn

# Tạo linear layer
linear = nn.Linear(in_features=30, out_features=7)

# Input: 8 samples, mỗi sample có 30 features
x = torch.randn(8, 30)
print(f"Input shape: {x.shape}")  # (8, 30)

# Forward pass
y = linear(x)
print(f"Output shape: {y.shape}")  # (8, 7)

# Xem weights và bias
print(f"Weight shape: {linear.weight.shape}")  # (7, 30)
print(f"Bias shape: {linear.bias.shape}")      # (7,)
```

---

### 3. Các Layer Khác Trong `torch.nn` (Tham Khảo)

Mặc dù project này chỉ dùng `nn.Linear`, nhưng `torch.nn` còn có nhiều layer khác:

```python
import torch.nn as nn

# Convolutional layers
nn.Conv1d()  # 1D convolution (cho time series)
nn.Conv2d()  # 2D convolution (cho images)

# Activation functions
nn.ReLU()    # ReLU activation
nn.Sigmoid() # Sigmoid activation
nn.Tanh()    # Tanh activation

# Normalization
nn.BatchNorm1d()  # Batch normalization
nn.LayerNorm()    # Layer normalization

# Dropout (regularization)
nn.Dropout()      # Dropout layer

# Recurrent layers
nn.LSTM()         # LSTM layer
nn.GRU()          # GRU layer
nn.RNN()          # RNN layer

# Loss functions
nn.MSELoss()      # Mean Squared Error
nn.CrossEntropyLoss()  # Cross Entropy Loss
```

---

## 📝 Ví Dụ Đầy Đủ: Từ Import Đến Sử Dụng

### Ví dụ 1: Linear Model

```python
import torch
import torch.nn as nn  # ← Import module

class Linear(nn.Module):  # ← Kế thừa từ nn.Module
    def __init__(self, seq_len: int, pred_len: int):
        super().__init__()
        
        # Tạo linear layer
        self.linear = nn.Linear(seq_len, pred_len)
        # ↑
        # nn.Linear là class trong torch.nn module
        # Tạo một layer với:
        # - Input: seq_len features
        # - Output: pred_len features
    
    def forward(self, x):
        return self.linear(x)  # ← Gọi forward của linear layer

# Sử dụng
model = Linear(seq_len=30, pred_len=7)
x = torch.randn(8, 30)  # 8 samples, 30 features
y = model(x)            # 8 samples, 7 predictions
```

### Ví dụ 2: DLinear Model (Dùng 2 Linear Layers)

```python
import torch
import torch.nn as nn

class DLinear(nn.Module):
    def __init__(self, seq_len: int, pred_len: int):
        super().__init__()
        
        # 2 linear layers riêng biệt
        self.linear_trend = nn.Linear(seq_len, pred_len)      # ← Layer 1
        self.linear_seasonal = nn.Linear(seq_len, pred_len)  # ← Layer 2
    
    def forward(self, x_trend, x_seasonal):
        y_trend = self.linear_trend(x_trend)
        y_seasonal = self.linear_seasonal(x_seasonal)
        return y_trend + y_seasonal
```

---

## 🎯 Tóm Tắt

### `import torch.nn as nn` là gì?

1. **`torch.nn`**: Module chứa các building blocks cho neural networks
2. **`as nn`**: Alias (bí danh) để code ngắn gọn hơn
3. **Mục đích**: Cung cấp các class như `nn.Module`, `nn.Linear`, etc.

### Các Class Quan Trọng Trong Project

| Class | Mục Đích | Ví Dụ Sử Dụng |
|-------|----------|---------------|
| `nn.Module` | Base class cho tất cả models | `class MyModel(nn.Module):` |
| `nn.Linear` | Linear transformation layer | `nn.Linear(30, 7)` |

### So Sánh: Có và Không Có `as nn`

```python
# Không dùng as nn (dài dòng)
import torch.nn
class Model(torch.nn.Module):
    def __init__(self):
        self.layer = torch.nn.Linear(30, 7)

# Dùng as nn (ngắn gọn - RECOMMENDED)
import torch.nn as nn
class Model(nn.Module):
    def __init__(self):
        self.layer = nn.Linear(30, 7)
```

---

## 💡 Lưu Ý Quan Trọng

1. **Luôn kế thừa `nn.Module`**: Tất cả models phải kế thừa từ `nn.Module`
2. **Gọi `super().__init__()`**: Luôn gọi trong `__init__` của model
3. **Định nghĩa `forward()`**: Method này được gọi khi bạn gọi `model(x)`
4. **Parameters tự động**: `nn.Module` tự động quản lý tất cả parameters

---

## 🔗 Liên Kết

- PyTorch Documentation: https://pytorch.org/docs/stable/nn.html
- Code trong project: `src/model/linear.py`, `src/model/n_linear.py`, `src/model/d_linear.py`

