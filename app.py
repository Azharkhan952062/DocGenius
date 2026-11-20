"""
DocGenius Application Entrypoint.
Ensures 100% deployment compatibility with Render, Streamlit Cloud, and Local execution.
Dispatches directly to the DocGenius application engine.
"""

import os
import sys
import runpy

# Ensure root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

target_script = os.path.join(PROJECT_ROOT, "DocGenius", "PDFChat.py")

if __name__ == "__main__" or "streamlit" in sys.modules:
    runpy.run_path(target_script, run_name="__main__")
