"""Streamlit app entry point for deployment."""
# This file is the entry point for Streamlit Cloud
# It imports and runs the actual app from src/streamlit_app.py

import sys
from pathlib import Path
import os

# Add project root to path
project_root = Path(__file__).parent.resolve()
sys.path.insert(0, str(project_root))

# Change to project root directory to ensure relative paths work
os.chdir(project_root)

# Import and run the actual Streamlit app
try:
    # Import the actual app module - this will execute all code in src/streamlit_app.py
    from src import streamlit_app
except ImportError as e:
    # If import fails, try to show helpful error
    import streamlit as st
    import traceback
    
    st.set_page_config(page_title="Error", page_icon="❌", layout="wide")
    st.error(f"❌ Error importing app module: {e}")
    
    with st.expander("🔍 Error Details"):
        st.code(traceback.format_exc())
    
    st.info(f"""
    **Debug Information:**
    - Project root: `{project_root}`
    - Current dir: `{os.getcwd()}`
    - Python path: `{sys.path[:3]}`
    - src/ exists: `{(project_root / 'src').exists()}`
    - streamlit_app.py exists: `{(project_root / 'src' / 'streamlit_app.py').exists()}`
    """)
    
    st.stop()
except Exception as e:
    # Other errors - let the actual app handle them
    import streamlit as st
    import traceback
    
    st.set_page_config(page_title="Error", page_icon="❌", layout="wide")
    st.error(f"❌ Unexpected error: {e}")
    
    with st.expander("🔍 Error Details"):
        st.code(traceback.format_exc())
    
    st.stop()

