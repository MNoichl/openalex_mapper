import base64
import json
import gradio as gr
import spaces
import torch
from spaces.zero.client import _get_token
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pathlib import Path

# FastAPI setup
app = FastAPI()

static_dir = Path('./static')
static_dir.mkdir(parents=True, exist_ok=True)

# Create a sample HTML file in the static directory for demonstration
with open(static_dir / "test.html", "w", encoding="utf-8") as f:
    f.write("<html><body><h1>Hello from static!</h1><p>We're serving this file without uvicorn!</p></body></html>")

app.mount("/static", StaticFiles(directory=static_dir), name="static")


@spaces.GPU(duration=4*60)  # Not possible with IP-based quotas on certain Spaces
def inner():
    return "ok"


def greet(request: gr.Request, n):
    """
    Example function that verifies GPU usage and decodes a token.
    """
    from spaces.zero.client import _get_token
    token = _get_token(request)
    print("Token:", token)
    # Check that the GPU-decorated function still works
    assert inner() == "ok"

    # A small example of token decoding
    payload = token.split('.')[1]
    payload = f"{payload}{'=' * ((4 - len(payload) % 4) % 4)}"
    try:
        decoded = base64.urlsafe_b64decode(payload).decode()
        return json.loads(decoded)
    except Exception as e:
        return {"error": str(e)}

@spaces.GPU(duration=4*60)  # Not possible with IP-based quotas on certain Spaces
def greet_wrapper(request: gr.Request, n):
    """
    Simple wrapper function that passes through inputs to greet function
    """
    return greet(request, n)


# Build a simple Gradio Blocks interface that also shows a static HTML iframe
with gr.Blocks() as demo:
    gr.Markdown("## Testing Static File Serving Without uvicorn")

    # Show the static HTML file inside an iframe
    gr.HTML(
        value='<iframe src="/static/test.html" width="100%" height="300px"></iframe>',
        label="Static HTML Demo"
    )

    # Add the original demonstration interface
    # (just a numeric input feeding into the greet() function)
    greet_interface = gr.Interface(fn=greet_wrapper, inputs=gr.Number(), outputs=gr.JSON())
    greet_interface.render()


# Mount the Gradio app
app = gr.mount_gradio_app(app, demo, path="/", ssr_mode=False)
app.zerogpu = True

__all__ = ['app']

# # We do NOT manually run uvicorn here; HF Spaces will serve the FastAPI app automatically.
# if __name__ == "__main__":
#     # This pass ensures that if you run it locally (e.g., python app.py),
#     # nothing breaks, but on Spaces it's auto-served via the 'app' object.
#     pass