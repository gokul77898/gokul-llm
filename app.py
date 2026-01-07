#!/usr/bin/env python3
"""
Hugging Face Space - Legal AI Assistant
Gradio interface for legal document analysis
"""

import gradio as gr

def process_text(input_text):
    """Placeholder function - training comes later."""
    if not input_text.strip():
        return "Please enter some text to process."
    return "Space is working. Training comes later."

# Create Gradio interface
demo = gr.Interface(
    fn=process_text,
    inputs=gr.Textbox(
        label="Input Text",
        placeholder="Enter legal text or question here...",
        lines=5
    ),
    outputs=gr.Textbox(
        label="Response",
        lines=5
    ),
    title="Legal AI Assistant",
    description="Upload legal documents or ask legal questions. AI responses coming soon.",
    allow_flagging="never"
)

# Launch server
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        quiet=False
    )
