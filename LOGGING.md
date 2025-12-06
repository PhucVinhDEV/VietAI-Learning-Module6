# 📋 Hướng dẫn xem Logs trong Streamlit

## 🔍 Xem Logs trên Streamlit Cloud

### ⚠️ Lưu ý quan trọng

Streamlit Cloud hiển thị logs từ:
- **stderr** (standard error) - được capture tốt nhất
- **print() statements** - hiển thị trong logs
- **Python logging module** - nếu output đến stderr

Code đã được cấu hình để:
- ✅ Logging output đến `sys.stderr`
- ✅ Sử dụng `print()` với `flush=True` để đảm bảo logs hiển thị ngay
- ✅ Format: `[INFO]` hoặc `[ERROR]` prefix cho dễ nhận biết

### Cách 1: Qua Dashboard (Khuyến nghị)

1. **Truy cập Streamlit Cloud Dashboard**
   - Vào [https://share.streamlit.io](https://share.streamlit.io)
   - Đăng nhập và chọn app của bạn

2. **Xem Logs**
   - Click vào app → **"Manage app"** (hoặc icon ⚙️)
   - Chọn tab **"Logs"** hoặc **"Runtime logs"**
   - Xem real-time logs hoặc scroll để xem logs cũ

3. **Tìm logs của bạn**
   - Tìm các dòng có prefix `[INFO]` hoặc `[ERROR]`
   - Ví dụ:
     ```
     [INFO] Loading data from project root: /mount/src/vietai-learning-module6
     [INFO] Data loaded: 1149 records
     [ERROR] FileNotFoundError: Checkpoint not found
     ```

4. **Filter Logs**
   - Có thể search logs bằng từ khóa: `[INFO]`, `[ERROR]`, `Loading`, etc.
   - Streamlit Cloud logs thường hiển thị cả stdout và stderr

### Cách 2: Debug trong App (Nếu logs không hiển thị)

Nếu logs không hiển thị trên Streamlit Cloud, có thể thêm debug panel trong app:

```python
# Thêm vào sidebar
if st.sidebar.checkbox("Show Debug Logs"):
    st.sidebar.subheader("📋 Recent Logs")
    # Hiển thị logs từ session state hoặc file
    if 'app_logs' in st.session_state:
        st.sidebar.text_area("Logs", st.session_state['app_logs'], height=200)
```

### Cách 3: Kiểm tra Logs qua Browser Console

1. Mở Developer Tools (F12)
2. Vào tab **Console**
3. Streamlit có thể log một số thông tin ở đây

### Troubleshooting: Logs không hiển thị

**Vấn đề**: Logs không xuất hiện trên Streamlit Cloud

**Giải pháp**:

1. **Kiểm tra code đã dùng `print()` với `flush=True`**:
   ```python
   print(f"[INFO] Message", file=sys.stderr, flush=True)
   ```

2. **Đảm bảo logging output đến stderr**:
   ```python
   logging.StreamHandler(sys.stderr)  # Not stdout
   ```

3. **Kiểm tra trong Streamlit Cloud Dashboard**:
   - Vào "Manage app" → "Logs"
   - Scroll xuống để xem logs cũ
   - Refresh page nếu cần

4. **Thử thêm explicit print statements**:
   ```python
   print("=" * 50, file=sys.stderr, flush=True)
   print("APP STARTED", file=sys.stderr, flush=True)
   print("=" * 50, file=sys.stderr, flush=True)
   ```

## 🖥️ Xem Logs khi chạy Local

### Chạy Streamlit với logging

```bash
# Chạy bình thường (logs hiển thị trong terminal)
streamlit run streamlit_app.py

# Hoặc redirect logs ra file
streamlit run streamlit_app.py 2>&1 | tee streamlit.log
```

### Xem logs trong terminal

Khi chạy `streamlit run`, logs sẽ hiển thị trực tiếp trong terminal:
- ✅ INFO logs: Thông tin về các operations
- ⚠️ WARNING logs: Cảnh báo
- ❌ ERROR logs: Lỗi với full traceback

## 📝 Logging trong Code

Code đã được setup logging với format:

```
YYYY-MM-DD HH:MM:SS - logger_name - LEVEL - message
```

### Các điểm logging chính:

1. **Data Loading**
   - `Loading data from project root: ...`
   - `Data loaded: X records`
   - `Data prepared: X records`

2. **Checkpoint Loading**
   - `Loading checkpoint from: ...`
   - `Loading checkpoint on device: ...`
   - `Checkpoint loaded successfully`
   - `Model loaded: MAPE=X.XX%`

3. **Prediction**
   - `Starting prediction generation...`
   - `History length: X, Predicting Y days`
   - `Using device: ...`
   - `Prediction completed: X predictions generated`

4. **Errors**
   - Tất cả errors đều được log với `exc_info=True` (full traceback)
   - Format: `ERROR - Error message - [full traceback]`

## 🐛 Debug Tips

### 1. Enable Debug Mode

Trong code, có thể thêm debug logging:

```python
# Trong src/streamlit_app.py
logger.setLevel(logging.DEBUG)  # Thêm dòng này để xem DEBUG logs
```

### 2. Xem Logs trong Streamlit UI

Có thể thêm một tab để xem logs trong app:

```python
# Thêm vào sidebar
if st.sidebar.checkbox("Show Logs"):
    with st.expander("Application Logs"):
        # Hiển thị logs từ memory hoặc file
        pass
```

### 3. Common Issues và Logs

| Issue | Log Message | Solution |
|-------|-------------|----------|
| File not found | `FileNotFoundError: ...` | Kiểm tra path và file tồn tại |
| Checkpoint error | `Checkpoint file missing key: ...` | Kiểm tra checkpoint format |
| Import error | `ModuleNotFoundError: ...` | Kiểm tra requirements.txt |
| Memory error | `RuntimeError: ...` | Giảm batch size hoặc data size |

## 📊 Log Levels

- **DEBUG**: Chi tiết nhất, dùng để debug
- **INFO**: Thông tin chung về operations (default)
- **WARNING**: Cảnh báo nhưng không dừng execution
- **ERROR**: Lỗi nghiêm trọng, có thể dừng execution
- **CRITICAL**: Lỗi cực kỳ nghiêm trọng

## 🔧 Customize Logging

Nếu muốn thay đổi logging format hoặc level:

```python
# Trong src/streamlit_app.py
logging.basicConfig(
    level=logging.DEBUG,  # Thay đổi level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('streamlit.log')  # Thêm file handler
    ]
)
```

## 📚 References

- [Python Logging Documentation](https://docs.python.org/3/library/logging.html)
- [Streamlit Cloud Logs](https://docs.streamlit.io/streamlit-cloud/get-started/manage-your-app#view-logs)

