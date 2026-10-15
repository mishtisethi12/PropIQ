"""
PropIQ - Streamlit Community Cloud Entry Point
Executes the main application from the repository root.
"""

import os
import sys
import runpy

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

app_path = os.path.join(BASE_DIR, "src", "ui", "app.py")
runpy.run_path(app_path, run_name="__main__")
