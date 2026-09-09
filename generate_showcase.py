"""
Pull a handful of real held-out validation cards and run them through the
actual pushed fine-tuned model (atharvasawant99/smolvlm2-pokemon-cards-v1)
to get genuine before/after showcase examples for the README - not
fabricated ones. Uses the exact same seed=42 shuffle/split as train.py /
modal_train.py so these are real samples the model never trained on.

Run with: modal run generate_showcase.py
"""

import modal

app = modal.App("smolvlm2-pokemon-showcase")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch", "transformers", "accelerate", "torchvision", "datasets", "pillow", "requests", "num2words"
)


@app.function(image=image, gpu="A10G", timeout=900)
def generate_showcase(n: int = 6):
    import io
    import json
    import random

    import requests
    import torch
    from datasets import load_dataset
    from PIL import Image
    from transformers import pipeline

    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.bfloat16
    FT_MODEL_ID = "atharvasawant99/smolvlm2-pokemon-cards-v1"
    BASE_MODEL_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"

    USER_PROMPT = """This is a Pokemon TCG card. Reply with ONLY a raw JSON object:
{"name": "pokemon name", "hp": "HP digits only", "set_name": "the set name"}
"""

    # same sampling/shuffle/split logic as the training script, so these are
    # genuinely part of the held-out validation split, not cherry-picked
    raw_dataset = load_dataset("TheFusion21/PokemonCards")["train"]
    raw_dataset = raw_dataset.shuffle(seed=42).select(range(min(1500, len(raw_dataset))))

    session = requests.Session()

    def download_image(url, timeout=10, retries=2):
        for attempt in range(retries):
            try:
                resp = session.get(url, timeout=timeout)
                resp.raise_for_status()
                return Image.open(io.BytesIO(resp.content)).convert("RGB")
            except Exception:
                if attempt == retries - 1:
                    return None
        return None

    formatted = []
    for sample in raw_dataset:
        img = download_image(sample["image_url"])
        if img is None:
            continue
        formatted.append((img, {"name": sample["name"], "hp": str(sample["hp"]), "set_name": sample["set_name"]}))

    random.seed(42)
    shuffled = random.sample(formatted, k=len(formatted))
    train_split = round(0.85 * len(shuffled))
    val_samples = shuffled[train_split:]  # same val split boundary as training

    chosen = random.sample(val_samples, k=min(n, len(val_samples)))

    ft_pipe = pipeline("image-text-to-text", model=FT_MODEL_ID, dtype=DTYPE, device=DEVICE)
    ft_pipe.model.generation_config.do_sample = False
    ft_pipe.model.generation_config.pad_token_id = ft_pipe.processor.tokenizer.eos_token_id

    base_pipe = pipeline("image-text-to-text", model=BASE_MODEL_ID, dtype=DTYPE, device=DEVICE)
    base_pipe.model.generation_config.do_sample = False
    base_pipe.model.generation_config.pad_token_id = base_pipe.processor.tokenizer.eos_token_id

    results = []
    for img, ground_truth in chosen:
        msg = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": USER_PROMPT}]}]

        ft_out = ft_pipe(text=msg, max_new_tokens=100)[0]["generated_text"][-1]["content"]
        base_out = base_pipe(text=msg, max_new_tokens=100)[0]["generated_text"][-1]["content"]

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)

        results.append(
            {
                "image_bytes": buf.getvalue(),
                "ground_truth": ground_truth,
                "base_output": base_out,
                "fine_tuned_output": ft_out,
            }
        )
        print(f"[INFO] {ground_truth['name']}: base={base_out!r}  ft={ft_out!r}")

    return results


@app.local_entrypoint()
def main():
    import json
    import os

    results = generate_showcase.remote(n=6)

    out_dir = os.path.join(os.path.dirname(__file__), "assets", "showcase")
    os.makedirs(out_dir, exist_ok=True)

    manifest = []
    for i, r in enumerate(results):
        img_path = os.path.join(out_dir, f"card_{i}.jpg")
        with open(img_path, "wb") as f:
            f.write(r["image_bytes"])
        manifest.append(
            {
                "image_file": f"card_{i}.jpg",
                "ground_truth": r["ground_truth"],
                "base_output": r["base_output"],
                "fine_tuned_output": r["fine_tuned_output"],
            }
        )

    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(manifest)} showcase samples to {out_dir}")
