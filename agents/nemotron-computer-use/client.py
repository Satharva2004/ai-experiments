"""
Talk to the deployed endpoint.

  python client.py                     # text-only test
  python client.py path/to/image.jpg   # image + text test
"""

import base64
import mimetypes
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url=os.environ["MODAL_ENDPOINT_URL"].rstrip("/") + "/v1",
    api_key=os.environ["VLLM_API_KEY"],
)

if len(sys.argv) > 1:
    path = sys.argv[1]
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    b64 = base64.b64encode(open(path, "rb").read()).decode()
    content = [
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
        {"type": "text", "text": "Describe this image."},
    ]
    temperature = 0.6  # model card's recommendation for vision
else:
    content = "Explain mixture-of-experts models in two sentences."
    temperature = 1.0

stream = client.chat.completions.create(
    model="nemotron-3.5-super-vl",
    messages=[{"role": "user", "content": content}],
    temperature=temperature,
    top_p=0.95,
    max_tokens=4096,
    stream=True,
)

for chunk in stream:
    delta = chunk.choices[0].delta if chunk.choices else None
    if delta and delta.content:
        print(delta.content, end="", flush=True)
print()
