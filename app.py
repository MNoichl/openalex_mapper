from pathlib import Path
import gradio as gr
from datetime import datetime
import sys


import spaces # necessary to run on Zero.
from spaces.zero.client import _get_token


# create a static directory to store the static files
static_dir = Path('./static')
static_dir.mkdir(parents=True, exist_ok=True)

@spaces.GPU(duration=4*60)
def predict(request: gr.Request,text_input):
    token = _get_token(request)
    file_name = f"{datetime.utcnow().strftime('%s')}.html"
    file_path = static_dir / file_name
    print(file_path)
    with open(file_path, "w") as f:
        f.write(f"""
        <script src="https://cdn.tailwindcss.com"></script>
        <body class="bg-gray-200 dark:text-white dark:bg-gray-900">
        <h1 class="text-3xl font-bold">
        Hello <i>{text_input}</i> From Gradio Iframe
        </h1>
        <h3>Filename: {file_name}</h3>
        """)
    iframe = f"""<iframe src="file={file_path}" width="100%" height="500px"></iframe>"""
    link = f'<a href="file={file_path}" target="_blank">{file_name}</a>'
    return link, iframe


with gr.Blocks() as block:
    gr.Markdown("""
## Gradio + FastAPI + Static Server
This is a demo of how to use Gradio with FastAPI and a static server.
The Gradio app generates dynamic HTML files and stores them in a static directory. FastAPI serves the static files.
""")
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(label="Name")
            markdown = gr.Markdown(label="Output Box")
            new_btn = gr.Button("New")
        with gr.Column():
            html = gr.HTML(label="HTML preview", show_label=True)

    new_btn.click(fn=predict, inputs=[text_input], outputs=[markdown, html])

block.launch(debug=True, share=False,ssr_mode=False)