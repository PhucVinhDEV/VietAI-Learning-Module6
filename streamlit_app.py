"""Streamlit app entry point for deployment."""
# This file is the entry point for Streamlit Cloud
# It imports and runs the actual app from src/streamlit_app.py

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import and run the actual Streamlit app
from src.streamlit_app import *

