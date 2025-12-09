#!/usr/bin/env python3
"""
HuggingFace Spaces entry point for Ideal Polyhedron Volume Toolkit.

This file is the entry point for HuggingFace Spaces deployment.
HF Spaces automatically runs app.py and calls demo.launch().
"""

import os
import sys

# Add bin directory to path so we can import gui module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bin.gui import create_gui

# Create the Gradio demo
demo = create_gui()

# HuggingFace Spaces will call demo.launch() automatically,
# but we can also run this directly for testing
if __name__ == "__main__":
    demo.launch()
