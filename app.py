import base64
import json

import gradio as gr
import spaces
import torch
from spaces.zero.client import _get_token

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
demo.launch()