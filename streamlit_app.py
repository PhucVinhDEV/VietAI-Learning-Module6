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
    from src.streamlit_app import *
except Exception as e:
    import streamlit as st
    st.error(f"Error importing app: {e}")
    st.code(f"""
    Project root: {project_root}
    Current dir: {os.getcwd()}
    Python path: {sys.path[:3]}
    """)
    raise

