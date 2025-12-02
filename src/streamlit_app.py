"""Streamlit app cho FPT Stock Prediction."""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend for Streamlit Cloud
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import sys
import os

# Add project root to path
# Always calculate from file location: src/streamlit_app.py -> go up 2 levels
_current_file = Path(__file__).resolve()
# If we're in src/streamlit_app.py, go up 2 levels to project root
if _current_file.parent.name == 'src':
    project_root = _current_file.parent.parent
else:
    # Fallback: assume we're at project root already
    project_root = Path.cwd()

# Ensure project root is in path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data import load_fpt_data, prepare_data, TimeSeriesDataset
from src.model import GRUModel
from src.utils import CONFIG, predict_future, evaluate_model, load_checkpoint
from torch.utils.data import DataLoader

# Page config
st.set_page_config(
    page_title="FPT Stock Prediction",
    page_icon="📈",
    layout="wide"
)

st.title("📈 FPT Stock Price Prediction")
st.markdown("Dự đoán giá cổ phiếu FPT sử dụng GRU Model")

# Sidebar cho configuration
st.sidebar.header("⚙️ Configuration")

# Checkpoint selection
st.sidebar.subheader("Model Checkpoint")
default_checkpoint = project_root / "models" / "fpt_gru" / "best_model.pt"
checkpoint_path = st.sidebar.text_input(
    "Checkpoint Path",
    value=str(default_checkpoint),
    help="Đường dẫn tới file checkpoint đã train"
)

# Prediction parameters
st.sidebar.subheader("Prediction Parameters")
total_predict_days = st.sidebar.slider("Days to Predict", 10, 200, CONFIG["total_predict_days"], 10)

# Main content
tab1, tab2, tab3 = st.tabs(["📊 Data Overview", "📥 Load Model", "🔮 Predict"])

# Tab 1: Data Overview
with tab1:
    st.header("Data Overview")
    
    if st.button("Load Data", key="load_data"):
        with st.spinner("Loading data..."):
            try:
                df = load_fpt_data()
                st.session_state['df'] = df
                # Prepare data ngay khi load để đảm bảo consistency với notebook
                df_processed = prepare_data(df)
                st.session_state['df_processed'] = df_processed
                st.success("Data loaded successfully!")
            except FileNotFoundError as e:
                st.error(f"❌ File not found: {e}")
                st.info("💡 Đảm bảo file `data/raw/FPT_train.csv` tồn tại trong repository.")
                st.code(f"Current directory: {Path.cwd()}\nProject root: {project_root}")
            except Exception as e:
                st.error(f"❌ Error loading data: {e}")
                import traceback
                with st.expander("🔍 Error Details"):
                    st.code(traceback.format_exc())
    
    if 'df' in st.session_state:
        try:
            df = st.session_state['df']
            
            # Prepare data ngay khi load để đảm bảo consistency
            if 'df_processed' not in st.session_state:
                df_processed = prepare_data(df)
                st.session_state['df_processed'] = df_processed
            else:
                df_processed = st.session_state['df_processed']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", len(df_processed))
            with col2:
                st.metric("Date Range", f"{df_processed['time'].min().date()} to {df_processed['time'].max().date()}")
            with col3:
                st.metric("Current Price", f"${df_processed['close'].iloc[-1]:.2f}")
            
            # View options
            col_view1, col_view2 = st.columns(2)

            with col_view1:
                range_mode = st.selectbox(
                    "Date range",
                    ["All", "Last 6 months", "Last 1 year", "Last 2 years", "Last N days"],
                    index=1,
                )
                n_days = None
                if range_mode == "Last N days":
                    n_days = st.slider("N days", min_value=30, max_value=365 * 3, value=180, step=30)

            with col_view2:
                show_ma = st.checkbox("Show Moving Average", value=True)
                ma_window = st.slider("MA window (days)", min_value=5, max_value=60, value=20, step=5)

            # Filter by date range
            df_plot = df_processed.copy()
            max_date = df_plot["time"].max()
            if range_mode == "Last 6 months":
                df_plot = df_plot[df_plot["time"] >= max_date - pd.Timedelta(days=180)]
            elif range_mode == "Last 1 year":
                df_plot = df_plot[df_plot["time"] >= max_date - pd.Timedelta(days=365)]
            elif range_mode == "Last 2 years":
                df_plot = df_plot[df_plot["time"] >= max_date - pd.Timedelta(days=365 * 2)]
            elif range_mode == "Last N days" and n_days is not None:
                df_plot = df_plot[df_plot["time"] >= max_date - pd.Timedelta(days=n_days)]

            # Compute moving average if needed
            if show_ma and len(df_plot) >= ma_window:
                df_plot["close_ma"] = df_plot["close"].rolling(window=ma_window).mean()
            else:
                df_plot["close_ma"] = np.nan

            # Price chart with options
            st.subheader("Price Chart")
            try:
                if len(df_plot) == 0:
                    st.warning("⚠️ No data to display for selected date range.")
                else:
                    fig, ax = plt.subplots(figsize=(12, 5))
                    ax.plot(df_plot["time"], df_plot["close"], label="Close Price", linewidth=1.5, color="tab:blue")
                    if show_ma and df_plot["close_ma"].notna().any():
                        ax.plot(
                            df_plot["time"],
                            df_plot["close_ma"],
                            label=f"MA ({ma_window}d)",
                            linewidth=1.5,
                            color="tab:orange",
                        )
                    ax.set_xlabel("Date")
                    ax.set_ylabel("Price (VND)")
                    ax.set_title("FPT Stock Close Price Over Time")
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)  # Close figure to free memory
            except Exception as e:
                st.error(f"❌ Error rendering chart: {e}")
                import traceback
                with st.expander("🔍 Error Details"):
                    st.code(traceback.format_exc())

            # Data preview controls
            st.subheader("Data Preview")
            try:
                preview_rows = st.slider("Number of rows to preview", min_value=10, max_value=200, value=20, step=10)
                
                # Dùng df gốc cho preview (có đầy đủ các cột)
                df = st.session_state['df']
                df_preview = df[df["time"].isin(df_plot["time"])].tail(preview_rows)
                
                # Chỉ hiển thị các cột có sẵn
                available_cols = ["time", "close"]
                if "open" in df_preview.columns:
                    available_cols.extend(["open", "high", "low", "volume"])
                
                st.dataframe(
                    df_preview[available_cols],
                    use_container_width=True,
                )
            except Exception as e:
                st.error(f"❌ Error displaying data preview: {e}")
                import traceback
                with st.expander("🔍 Error Details"):
                    st.code(traceback.format_exc())
        except Exception as e:
            st.error(f"❌ Error processing data: {e}")
            import traceback
            with st.expander("🔍 Error Details"):
                st.code(traceback.format_exc())

# Tab 2: Load Model
with tab2:
    st.header("Load Trained Model")
    st.markdown("Load model đã được train từ checkpoint file")
    
    if st.button("Load Checkpoint", key="load_checkpoint"):
        with st.spinner("Loading checkpoint..."):
            try:
                checkpoint_path_obj = Path(checkpoint_path)
                if not checkpoint_path_obj.exists():
                    st.error(f"❌ Checkpoint not found: {checkpoint_path}")
                    st.info("💡 Hãy chạy script train trước: `python scripts/train_fpt_gru.py`")
                else:
                    # Load checkpoint
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    checkpoint = load_checkpoint(checkpoint_path_obj, device=device)
                    
                    # Extract info
                    model_config = checkpoint['config']
                    scaler = checkpoint['scaler']
                    metrics = checkpoint['metrics']
                    train_losses = checkpoint['train_losses']
                    val_losses = checkpoint['val_losses']
                    
                    # Create model với config từ checkpoint
                    model = GRUModel(
                        input_size=1,  # close_log only
                        hidden_size=model_config['hidden_size'],
                        num_layers=model_config['num_layers'],
                        dropout=model_config['dropout'],
                        output_len=model_config['output_len']
                    ).to(device)
                    
                    # Load weights
                    model.load_state_dict(checkpoint['model_state_dict'])
                    model.eval()
                    
                    # Save to session state
                    st.session_state['model'] = model
                    st.session_state['scaler'] = scaler
                    st.session_state['model_config'] = model_config
                    st.session_state['metrics'] = metrics
                    st.session_state['train_losses'] = train_losses
                    st.session_state['val_losses'] = val_losses
                    
                    st.success(f"✅ Model loaded successfully!")
                    
                    # Show model metrics
                    st.subheader("📊 Model Metrics")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Validation MAPE", f"{metrics['mape']:.2f}%")
                    with col2:
                        st.metric("Best Val Loss", f"{metrics['best_val_loss']:.6f}")
                    with col3:
                        st.metric("Final Train Loss", f"{metrics['final_train_loss']:.6f}")
                    with col4:
                        st.metric("Final Val Loss", f"{metrics['final_val_loss']:.6f}")
                    
                    # Show model architecture
                    st.subheader("🏗️ Model Architecture")
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        st.metric("Input Length", model_config['input_len'])
                    with col2:
                        st.metric("Output Length", model_config['output_len'])
                    with col3:
                        st.metric("Hidden Size", model_config['hidden_size'])
                    with col4:
                        st.metric("Layers", model_config['num_layers'])
                    with col5:
                        st.metric("Dropout", f"{model_config['dropout']:.2f}")
                    
                    # Training info
                    st.subheader("⚙️ Training Configuration")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Learning Rate", f"{model_config['learning_rate']:.0e}")
                    with col2:
                        st.metric("Batch Size", model_config['batch_size'])
                    with col3:
                        st.metric("Epochs", len(train_losses))
                    with col4:
                        st.metric("Device", str(model_config['device']))
                    
                    # Show training curves
                    st.subheader("Training Curves")
                    fig, ax = plt.subplots(figsize=(12, 5))
                    ax.plot(train_losses, label='Train Loss', linewidth=2)
                    ax.plot(val_losses, label='Val Loss', linewidth=2)
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel('Loss')
                    ax.set_title('Training & Validation Loss')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    
                    
            except Exception as e:
                st.error(f"Error loading checkpoint: {e}")
                import traceback
                st.code(traceback.format_exc())
    
    # Show instructions
    if 'model' not in st.session_state:
        st.info("""
        **📝 Hướng dẫn:**
        
        1. **Train model trước** bằng script:
           ```bash
           python scripts/train_fpt_gru.py
           ```
        
        2. Model sẽ được lưu tại: `models/fpt_gru/best_model.pt`
        
        3. Click **"Load Checkpoint"** để load model đã train
        
        4. Sau đó chuyển sang tab **"Predict"** để dự đoán
        """)

# Tab 3: Predict
with tab3:
    st.header("Future Prediction")
    
    if 'model' not in st.session_state:
        st.warning("⚠️ Please load model first in 'Load Model' tab!")
    elif 'df' not in st.session_state:
        st.warning("⚠️ Please load data first in 'Data Overview' tab!")
    else:
        if st.button("Generate Prediction", key="predict"):
            with st.spinner("Generating predictions..."):
                try:
                    model = st.session_state['model']
                    scaler = st.session_state['scaler']
                    model_config = st.session_state['model_config']
                    
                    # Sử dụng df_processed đã được prepare (giống notebook)
                    if 'df_processed' not in st.session_state:
                        df = st.session_state['df']
                        df_processed = prepare_data(df)
                        st.session_state['df_processed'] = df_processed
                    else:
                        df_processed = st.session_state['df_processed']
                    
                    # Get history - dùng toàn bộ df_processed giống notebook
                    history = df_processed["close_log"].tolist()
                    
                    # Predict
                    device = model_config['device']
                    pred_close = predict_future(
                        model, history, scaler,
                        model_config["input_len"], total_predict_days,
                        device
                    )
                    
                
                    st.session_state['pred_close'] = pred_close
                    st.session_state['df_processed'] = df_processed
                    
                    # Show results
                    st.success(f"✅ Predicted {len(pred_close)} days ahead!")
                    
                    # Chart
                    st.subheader("Future Price Prediction")
                    fig, ax = plt.subplots(figsize=(14, 6))
                    
                    # Plot historical data
                    last_n = min(100, len(df_processed))
                    historical = df_processed['close'].iloc[-last_n:].values
                    historical_dates = range(-last_n, 0)
                    ax.plot(historical_dates, historical, label='Historical', linewidth=2, color='blue')
                    
                    # Plot predictions
                    pred_dates = range(0, len(pred_close))
                    ax.plot(pred_dates, pred_close, label='Predicted', linewidth=2, color='red', linestyle='--')
                    
                    ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5, label='Now')
                    ax.set_xlabel('Days (relative)')
                    ax.set_ylabel('Price (VND)')
                    ax.set_title(f'FPT Stock Price: Historical & {len(pred_close)}-Day Forecast')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    
                    # Statistics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Current Price", f"${df_processed['close'].iloc[-1]:.2f}")
                    with col2:
                        st.metric("Predicted (Day 1)", f"${pred_close[0]:.2f}")
                    with col3:
                        st.metric("Predicted (Final)", f"${pred_close[-1]:.2f}")
                    with col4:
                        change = ((pred_close[-1] - df_processed['close'].iloc[-1]) / df_processed['close'].iloc[-1]) * 100
                        st.metric("Total Change", f"{change:+.2f}%")
                    
                    # Download CSV
                    st.subheader("Download Predictions")
                    pred_df = pd.DataFrame({
                        "id": range(1, len(pred_close) + 1),
                        "close": pred_close
                    })
                    csv = pred_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name="fpt_predictions.csv",
                        mime="text/csv"
                    )
                    
                except Exception as e:
                    st.error(f"Error during prediction: {e}")
                    import traceback
                    st.code(traceback.format_exc())

