import gradio as gr

def chat(text):
    return "HF Space is working. Training is separate."

demo = gr.Interface(
    fn=chat,
    inputs="text",
    outputs="text",
    title="Omilos Legal AI"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

