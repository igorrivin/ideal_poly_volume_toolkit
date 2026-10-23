#!/usr/bin/env python3
"""
HuggingFace Spaces entry point for Ideal Polyhedron Volume Toolkit.

This file is the entry point for HuggingFace Spaces deployment.
HF Spaces automatically runs app.py and calls demo.launch().
"""

import os
import sys

# Debug: print versions before anything else
try:
    import gradio
    print(f"Gradio version: {gradio.__version__}")
except Exception as e:
    print(f"Could not import gradio: {e}")

try:
    import huggingface_hub
    print(f"HuggingFace Hub version: {huggingface_hub.__version__}")
except Exception as e:
    print(f"Could not import huggingface_hub: {e}")

# Add the repo root to path so imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from bin.gui import APP_THEME, create_gui
    demo = create_gui()
    print("Successfully loaded Ideal Polyhedron Volume Toolkit GUI")
except Exception as e:
    # Fallback: show error in a simple Gradio interface
    import gradio as gr
    import traceback

    APP_THEME = gr.themes.Soft()

    error_msg = f"Failed to load the main GUI:\n\n{str(e)}\n\n{traceback.format_exc()}"
    print(error_msg)

    with gr.Blocks() as demo:
        gr.Markdown("# Error Loading Application")
        gr.Markdown(f"```\n{error_msg}\n```")

# For Docker SDK on HF Spaces, we must explicitly launch
# with server_name="0.0.0.0" to bind to all interfaces
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=APP_THEME)
