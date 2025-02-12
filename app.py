from pathlib import Path
import tempfile
import gradio as gr
from datetime import datetime
import os
import sys

import spaces  # necessary to run on Zero.
from spaces.zero.client import _get_token

# Instead of a static folder, use the system temporary directory.
# You can also create a subfolder within tempfile.gettempdir() if needed.
rtemp_dir = Path(tempfile.gettempdir()) / "gradio_generated_files"
rtemp_dir.mkdir(parents=True, exist_ok=True)
# Optionally, set GRADIO_ALLOWED_PATHS to this directory if required.
os.environ["GRADIO_ALLOWED_PATHS"] = str(rtemp_dir.resolve())

@spaces.GPU(duration=10)
def predict(request: gr.Request, text_input):
    token = _get_token(request)
    file_name = f"{datetime.utcnow().strftime('%s')}.html"
    file_path = rtemp_dir / file_name
    print("Writing file to:", file_path)
    with open(file_path, "w") as f:
        f.write(f"""<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-200 dark:text-white dark:bg-gray-900">
    <h1 class="text-3xl font-bold">
        Hello <i>{text_input}</i> From Gradio Iframe
    </h1>
    <h3>Filename: {file_name}</h3>
</body>
</html>
        """)
    os.chmod(file_path, 0o644)
    # Construct the URL relative to the repo root.
    # If Gradio automatically serves files from the temporary directory,
    # you might need to check what URL path it uses.
    # Here we assume it serves files from /file=<basename>.
    iframe = f'<iframe src="/file={file_name}" width="100%" height="500px"></iframe>'
    link = f'<a href="/file={file_name}" target="_blank">{file_name}</a>'
    print("Serving file at URL:", f"/file={file_name}")
    return link, iframe

with gr.Blocks() as block:
    gr.Markdown("""
## Gradio + Temporary Files Demo
This demo generates dynamic HTML files and stores them in the temporary directory.
""")
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(label="Name")
            markdown = gr.Markdown(label="Output Link")
            new_btn = gr.Button("New")
        with gr.Column():
            html = gr.HTML(label="HTML Preview", show_label=True)

    new_btn.click(fn=predict, inputs=[text_input], outputs=[markdown, html])

block.launch(debug=True, share=False, ssr_mode=False)