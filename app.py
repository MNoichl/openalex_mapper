import base64
import json
import gradio as gr
import spaces
import torch
from spaces.zero.client import _get_token
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn
from pathlib import Path

# FastAPI setup
app = FastAPI()
static_dir = Path('./static')
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@spaces.GPU(duration=4*60) # Not possible with IP-based quotas
def inner():
    return "ok"

def greet(request: gr.Request, n):
    token = _get_token(request)
    print(token)
    assert inner() ==  "ok"
    payload = token.split('.')[1]
    payload = f"{payload}{'=' * ((4 - len(payload) % 4) % 4)}"
    return json.loads(base64.urlsafe_b64decode(payload).decode())

demo = gr.Interface(fn=greet, inputs=gr.Number(), outputs=gr.JSON())

# Mount Gradio app
app = gr.mount_gradio_app(app, demo, path="/", ssr_mode=False)

app.zerogpu = True

def start_server(app):
    uvicorn.run(app, host="0.0.0.0", port=7860)

start_server.zerogpu = True

if __name__ == "__main__":
    start_server(app)