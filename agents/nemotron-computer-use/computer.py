"""
Eyes and hands: take a screenshot, and click / type / press keys / scroll / open apps.

The screenshot is scaled to the screen's size in points (what pyautogui uses),
so an (x, y) the model reads off the image is exactly where we click.
"""

import base64
import io
import os
import subprocess
import tempfile
import time

import pyautogui
from PIL import Image, ImageDraw

pyautogui.FAILSAFE = True  # emergency stop: push the mouse into any screen corner
pyautogui.PAUSE = 0.1

# Key names the model might use -> pyautogui names
KEY_NAMES = {"cmd": "command", "opt": "option", "alt": "option", "control": "ctrl", "return": "enter", "escape": "esc"}


def screenshot():
    """Returns (base64 JPEG, width, height) of the main screen, with the mouse cursor."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "screen.png")
        subprocess.run(["screencapture", "-x", "-C", path], check=True)  # -x: no sound, -C: show cursor
        img = Image.open(path).convert("RGB")
    w, h = pyautogui.size()
    img = img.resize((w, h))  # Retina pixels -> points
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode(), w, h


def to_screen(action):
    """The model gives x, y on a 0-1000 scale (0,0 = top-left, 1000,1000 = bottom-right). Convert to screen points."""
    w, h = pyautogui.size()
    return round(float(action["x"]) / 1000 * w), round(float(action["y"]) / 1000 * h)


def save_step(image_b64, action, path):
    """Save what the model saw, with a red ring where it chose to click, and the action written on top."""
    img = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
    d = ImageDraw.Draw(img)
    if "x" in action and "y" in action:
        x, y = float(action["x"]) / 1000 * img.width, float(action["y"]) / 1000 * img.height
        d.ellipse([x - 14, y - 14, x + 14, y + 14], outline="red", width=4)
    label = {k: v for k, v in action.items() if k != "why"}
    d.rectangle([0, 0, img.width, 28], fill="black")
    d.text((8, 6), str(label), fill="white")
    img.save(path, quality=85)


def do(action):
    """Run one action dict from the model, e.g. {"action": "click", "x": 500, "y": 250}."""
    kind = action["action"]

    if kind in ("click", "double_click", "right_click"):
        x, y = to_screen(action)
        {"click": pyautogui.click, "double_click": pyautogui.doubleClick, "right_click": pyautogui.rightClick}[kind](x, y)

    elif kind == "type":
        # Paste instead of typing key by key: faster, and works for any language or emoji.
        subprocess.run(["pbcopy"], input=action["text"].encode(), check=True)
        pyautogui.hotkey("command", "v", interval=0.05)

    elif kind == "key":
        keys = [KEY_NAMES.get(k.strip().lower(), k.strip().lower()) for k in action["keys"].split("+")]
        pyautogui.hotkey(*keys, interval=0.05)

    elif kind == "scroll":
        pyautogui.moveTo(*to_screen(action))
        pyautogui.scroll(int(action.get("amount", -5)))

    elif kind == "open_app":
        subprocess.run(["open", "-a", action["name"]], check=True)
        time.sleep(2)  # give the app time to open

    elif kind == "wait":
        time.sleep(2)

    else:
        raise ValueError(f"unknown action: {kind}")

    time.sleep(1)  # let the screen update before the next screenshot
