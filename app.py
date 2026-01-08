import gradio as gr

def chat(text):
    return "OmilosLLM Space is live. Training runs via Hugging Face Jobs."

demo = gr.Interface(
    fn=chat,
    inputs="text",
    outputs="text",
    title="Omilos LLM"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
