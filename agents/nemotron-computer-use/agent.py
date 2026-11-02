"""
Computer-use agent on our own Nemotron server (Modal).

Loop: screenshot -> model picks ONE action -> do it -> repeat, until the model says "done".

  python agent.py "open Calculator and compute 19 times 37"
  python agent.py "what is on my screen?"       # screenshot understanding only
  python agent.py --voice                       # talk to it instead of typing
  python agent.py --dry-run "open Safari"       # show the first action, do nothing

Emergency stop: Ctrl+C, or push the mouse into a screen corner.
"""

import argparse
import json
import os
import time

from dotenv import load_dotenv
from openai import OpenAI

import computer
import voice

load_dotenv()

MODEL = "nemotron-3.5-super-vl"
MAX_STEPS = 25

client = OpenAI(
    base_url=os.environ["MODAL_ENDPOINT_URL"].rstrip("/") + "/v1",
    api_key=os.environ["VLLM_API_KEY"],
    timeout=120,
)

SYSTEM = """Your name is Gandu, You control a Mac to do the user's task. Each turn you get a screenshot of the screen and the steps done so far.

Reply with ONE JSON object and nothing else. Choose exactly one action:
{"action": "open_app", "name": "Calculator", "why": "..."}   opens or switches to an app
{"action": "click", "x": 500, "y": 250, "why": "..."}
{"action": "double_click", "x": 500, "y": 250, "why": "..."}
{"action": "right_click", "x": 500, "y": 250, "why": "..."}
{"action": "type", "text": "hello", "why": "..."}            types into the focused field
{"action": "key", "keys": "cmd+l", "why": "..."}              e.g. enter, esc, tab, cmd+t, cmd+a
{"action": "scroll", "x": 500, "y": 500, "amount": -5, "why": "..."}   negative = down
{"action": "wait", "why": "..."}                              the screen is still loading
{"action": "done", "answer": "...", "why": "..."}             task finished; "answer" is said out loud

Rules:
- x, y are on a 0-1000 scale: (0, 0) is the top-left of the screenshot, (1000, 1000) the bottom-right.
- To open an app, always use open_app. Do not click the Dock or use Spotlight.
- Prefer the keyboard when it is reliable: in Calculator, type "19*37" then press enter; in a browser, cmd+l, type the URL, enter.
- Click a text field before typing in it. Press enter after typing a search or URL.
- After each step, look at the screenshot to check it worked. If not, try a different way. Do not give up early.
- If the task is only a question about the screen, look and reply with "done" and the answer.
- Keep "answer" short: one or two sentences, it is spoken out loud.
- Never type into the Terminal or a code editor unless the task asks for it.
- Before sending, deleting, buying or posting anything, reply "done" and ask the user instead."""


def wake_server():
    """Modal turns the GPUs off when idle. A cold start takes ~8 minutes; wait for it."""
    for attempt in range(40):
        try:
            client.with_options(timeout=60).models.list()
            return
        except Exception:
            if attempt == 0:
                voice.say("Waking up the model server. This takes a few minutes.")
            time.sleep(10)
    raise RuntimeError("Model server did not start. Check: modal app logs nemotron-vl")


def next_action(task, history):
    """Send the screenshot + task + history, get one action dict back."""
    image, _, _ = computer.screenshot()
    steps = "\n".join(history) or "(none yet)"
    reply = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}},
                {"type": "text", "text": f"Task: {task}\n\nSteps so far:\n{steps}\n\nWhat is the next action?"},
            ]},
        ],
        max_tokens=512,
        temperature=0.2,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},  # fast mode
    ).choices[0].message.content or ""
    try:
        return json.loads(reply[reply.index("{"): reply.rindex("}") + 1]), image
    except ValueError:
        return None, image  # not valid JSON; we tell the model and retry


def run(task, dry_run=False):
    print(f"\n▶ Task: {task}")
    wake_server()
    run_dir = os.path.join("runs", time.strftime("%Y%m%d-%H%M%S"))  # what the model saw at each step
    os.makedirs(run_dir, exist_ok=True)
    history = []
    for step in range(1, MAX_STEPS + 1):
        action, image = next_action(task, history)
        if action is None or "action" not in action:
            history.append(f"{step}. (your reply was not valid JSON; reply with one JSON object only)")
            continue

        computer.save_step(image, action, os.path.join(run_dir, f"step-{step:02d}.jpg"))
        where = f" -> screen {computer.to_screen(action)}" if "x" in action else ""
        print(f"  {step}. {action['action']:12} {json.dumps({k: v for k, v in action.items() if k not in ('action', 'why')})}"
              f"{where}  — {action.get('why', '')}")

        if action["action"] == "done":
            print(f"  Screenshots of each step: {run_dir}/")
            voice.say(action.get("answer") or "Done.")
            return
        if dry_run:
            print("  (dry run: nothing was clicked)")
            return

        try:
            computer.do(action)
            history.append(f"{step}. {json.dumps(action)}")
        except Exception as e:
            history.append(f"{step}. {json.dumps(action)} -> FAILED: {e}")

    voice.say(f"I stopped after {MAX_STEPS} steps without finishing.")
    print(f"  Screenshots of each step: {run_dir}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Computer-use agent on Nemotron 3.5 Super VL")
    p.add_argument("task", nargs="*", help="what to do (leave empty with --voice)")
    p.add_argument("--voice", action="store_true", help="speak tasks instead of typing them")
    p.add_argument("--dry-run", action="store_true", help="show the first action without doing it")
    args = p.parse_args()

    try:
        if args.voice:
            while True:
                task = voice.listen()
                if task:
                    run(task, args.dry_run)
        else:
            run(" ".join(args.task) or input("Task: "), args.dry_run)
    except KeyboardInterrupt:
        print("\nStopped.")
