# 🚀 Hướng dẫn Deploy lên Streamlit Cloud

## Bước 1: Chuẩn bị Repository

1. **Đảm bảo code đã được commit lên GitHub/GitLab/Bitbucket**

   ```bash
   git add .
   git commit -m "Prepare for Streamlit Cloud deployment"
   git push origin main
   ```

2. **Kiểm tra các file cần thiết:**
   - ✅ `streamlit_app.py` (entry point ở root)
   - ✅ `requirements.txt` (đã có streamlit)
   - ✅ `.streamlit/config.toml` (config cho Streamlit)
   - ✅ `models/fpt_gru/best_model.pt` (model checkpoint - cần commit hoặc dùng Git LFS)

## Bước 2: Deploy lên Streamlit Cloud

### Cách 1: Deploy từ GitHub (Khuyến nghị)

1. **Truy cập [Streamlit Cloud](https://streamlit.io/cloud)**

   - Đăng nhập bằng GitHub account
   - Click "New app"

2. **Điền thông tin:**

   - **Repository**: Chọn repo của bạn
   - **Branch**: `main` (hoặc branch bạn muốn)
   - **Main file path**: `streamlit_app.py`
   - **App URL**: Tự động tạo (ví dụ: `your-app-name.streamlit.app`)

3. **Click "Deploy"**

### Cách 2: Deploy từ GitLab/Bitbucket

1. Kết nối GitLab/Bitbucket account với Streamlit Cloud
2. Chọn repository và branch
3. Điền main file path: `streamlit_app.py`
4. Click "Deploy"

## Bước 3: Xử lý Model Files (Quan trọng!)

### Option 1: Commit model vào Git (cho model nhỏ < 100MB)

```bash
git add models/fpt_gru/best_model.pt
git commit -m "Add trained model"
git push
```

### Option 2: Dùng Git LFS (cho model lớn)

1. **Cài đặt Git LFS:**

   ```bash
   git lfs install
   ```

2. **Track model files:**
   ```bash
   git lfs track "*.pt"
   git add .gitattributes
   git add models/fpt_gru/best_model.pt
   git commit -m "Add model with Git LFS"
   git push
   ```

### Option 3: Download model khi deploy (Khuyến nghị cho model lớn)

Nếu model quá lớn, bạn có thể:

- Lưu model trên Google Drive / Dropbox
- Download trong `streamlit_app.py` khi app khởi động
- Hoặc dùng Streamlit Secrets để lưu download link

**Ví dụ code download model:**

```python
# Thêm vào đầu streamlit_app.py
import gdown

model_path = Path("models/fpt_gru/best_model.pt")
if not model_path.exists():
    # Download từ Google Drive
    url = "https://drive.google.com/uc?id=YOUR_FILE_ID"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    gdown.download(url, str(model_path), quiet=False)
```

## Bước 4: Cấu hình Secrets (Nếu cần)

Nếu cần API keys hoặc sensitive data:

1. Trong Streamlit Cloud dashboard, vào "Settings" → "Secrets"
2. Thêm secrets dạng TOML:

   ```toml
   [secrets]
   API_KEY = "your-api-key"
   MODEL_URL = "https://drive.google.com/..."
   ```

3. Sử dụng trong code:
   ```python
   import streamlit as st
   api_key = st.secrets["secrets"]["API_KEY"]
   ```

## Bước 5: Kiểm tra Deployment

1. **Xem logs:**

   - Vào Streamlit Cloud dashboard
   - Click vào app → "Manage app" → "Logs"
   - Kiểm tra lỗi nếu có

2. **Common issues:**
   - **ModuleNotFoundError**: Kiểm tra `requirements.txt` đã có đủ packages
   - **FileNotFoundError**: Đảm bảo model file đã được commit hoặc download được
   - **Memory error**: Model quá lớn, cần dùng Git LFS hoặc download

## Bước 6: Update App

Mỗi khi push code mới lên repository:

- Streamlit Cloud sẽ tự động rebuild app
- Hoặc có thể manual trigger rebuild trong dashboard

## 📝 Checklist trước khi deploy

- [ ] Code đã được push lên Git repository
- [ ] `requirements.txt` đã có `streamlit` và tất cả dependencies
- [ ] `streamlit_app.py` ở root directory
- [ ] Model file đã được xử lý (commit/Git LFS/download)
- [ ] Test chạy local: `streamlit run streamlit_app.py`
- [ ] Không có hardcoded paths (dùng relative paths)
- [ ] Không có secrets trong code (dùng Streamlit Secrets)

## 🔗 Links hữu ích

- [Streamlit Cloud Documentation](https://docs.streamlit.io/streamlit-community-cloud)
- [Streamlit Secrets](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app/secrets-management)
- [Git LFS](https://git-lfs.github.com/)

## 💡 Tips

1. **Optimize model size**: Có thể quantize model để giảm kích thước
2. **Caching**: Dùng `@st.cache_data` và `@st.cache_resource` để cache data/model
3. **Error handling**: Thêm try-except để handle lỗi gracefully
4. **Loading states**: Dùng `st.spinner()` để show loading state
