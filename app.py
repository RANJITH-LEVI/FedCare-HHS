"""
FedCare-HHS Streamlit Cloud Entry Point
Delegates execution to frontend/app.py for Streamlit Community Cloud deployments.
"""

import os
import sys

# Ensure repository root is on sys.path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Import and execute frontend app
import frontend.app
