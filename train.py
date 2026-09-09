"""
Fine-tune SmolVLM2 on Pokemon TCG Cards (structured extraction).

Ported from a Kaggle notebook (notebook0d19635ddf.ipynb) built for a 2x T4
environment. Differences from the original, and why:

- No CUDA_VISIBLE_DEVICES pinning / DataParallel-avoidance workaround. That
  was needed on Kaggle because Trainer would otherwise wrap the model across
  both visible T4s. Baseten's `compute.accelerator.count=1` in config.py
  provisions exactly one GPU, so there is nothing to pin against.
- DTYPE is bfloat16, not float16. The original used native fp16 (not
  autocast mixed precision) specifically because T4 has no good bf16
  support and mixed-precision autocast crashed SmolVLM's `inputs_merger`.
  A10G (Ampere) has real bf16 tensor-core support, so bf16 is used here -
  but the SAME strategy that avoided the crash is kept: the model is loaded
  natively in bf16 (not via autocast), and `fp16`/`bf16` stay False in
  SFTConfig below, so Trainer's own mixed-precision/GradScaler path is
  still never engaged. This has NOT been re-validated against the
  inputs_merger crash on a different dtype - if it resurfaces, reverting
  DTYPE to torch.float16 restores the exact proven-working configuration.
- Secrets come from plain env vars (HF_TOKEN, WANDB_API_KEY), not
  kaggle_secrets.UserSecretsClient - Baseten's training runtime injects
  secret values directly as env vars per config.py's
  runtime.environment_variables mapping.
- output_dir points at CHECKPOINT_DIR (set to match config.py's
  checkpointing_config.checkpoint_path) instead of /kaggle/working, so
  checkpoints land in Baseten's persisted checkpoint volume and survive a
  job restart the same way the original's Kaggle-session-resume logic did.
- The exploratory baseline-vs-finetuned comparison and the matplotlib loss
  plot were dropped from the unattended run (no one is watching a live plot
  during a headless job) - metrics still go to W&B via report_to="wandb" if
  WANDB_API_KEY is set. A lightweight before/after text comparison on one
  validation sample is kept at the end as a cheap sanity check, printed to
  the job's logs instead of rendered inline.

Everything else - the dataset formatting, the vision-encoder freeze, and
especially the label-collator + its sanity check (this is the piece that
was previously silently broken and produced loss=0.0/nan for an entire
run) - is preserved as-is, since it's proven-correct logic, not
platform-specific.
"""

import gc
import io
import json
import os
import random

import torch
from PIL import Image
from tqdm.auto import tqdm

WANDB_API_KEY = os.environ.get("WANDB_API_KEY")
HF_TOKEN = os.environ.get("HF_TOKEN")

if WANDB_API_KEY:
    import wandb

    wandb.login(key=WANDB_API_KEY)
    print("[INFO] Logged in to Weights & Biases.")
else:
    print("[WARN] No WANDB_API_KEY env var set -- W&B logging will be skipped.")

if not HF_TOKEN:
    print("[WARN] No HF_TOKEN env var set -- the Hugging Face push step will be skipped.")

from transformers import AutoModelForImageTextToText, AutoProcessor, pipeline

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16
MODEL_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
CHECKPOINT_DIR = "/tmp/training_checkpoints/smolvlm2-pokemon-cards-v1"
print(f"[INFO] Using device: {DEVICE} | dtype: {DTYPE}")

# ---------------------------------------------------------------------------
# Load dataset
# ---------------------------------------------------------------------------
from datasets import load_dataset

raw_dataset = load_dataset("TheFusion21/PokemonCards")["train"]

NUM_SAMPLES = 1500  # increase for a longer/richer run
raw_dataset = raw_dataset.shuffle(seed=42).select(range(min(NUM_SAMPLES, len(raw_dataset))))

# ---------------------------------------------------------------------------
# Download images and format into chat message format
# ---------------------------------------------------------------------------
import requests

SESSION = requests.Session()

USER_PROMPT = """This is a Pokemon TCG card. Reply with ONLY a raw JSON object:
{"name": "pokemon name", "hp": "HP digits only", "set_name": "the set name"}
"""


def download_image(url, timeout=10, retries=2):
    for attempt in range(retries):
        try:
            resp = SESSION.get(url, timeout=timeout)
            resp.raise_for_status()
            img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            return buf.getvalue()
        except Exception:
            if attempt == retries - 1:
                return None
    return None


def format_data(sample):
    img_bytes = download_image(sample["image_url"])
    if img_bytes is None:
        return None

    target = json.dumps(
        {
            "name": sample["name"],
            "hp": str(sample["hp"]),
            "set_name": sample["set_name"],
        }
    )

    # Every content dict has the same keys (type/text/image) so the schema
    # is consistent across rows, which avoids feature-inference errors.
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_bytes, "text": None},
                    {"type": "text", "text": USER_PROMPT, "image": None},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": target, "image": None}]},
        ]
    }


dataset_processed = []
skipped = 0
for sample in tqdm(raw_dataset, desc="Downloading + formatting"):
    row = format_data(sample)
    if row is None:
        skipped += 1
        continue
    dataset_processed.append(row)

print(f"[INFO] Formatted {len(dataset_processed)} samples, skipped {skipped} (failed downloads)")

# ---------------------------------------------------------------------------
# Train / validation split
# ---------------------------------------------------------------------------
from datasets import Dataset

random.seed(42)
dataset_shuffled = random.sample(dataset_processed, k=len(dataset_processed))

train_split = round(0.85 * len(dataset_shuffled))
train_dataset = Dataset.from_list(dataset_shuffled[:train_split])
val_dataset = Dataset.from_list(dataset_shuffled[train_split:])

print(f"[INFO] Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

# ---------------------------------------------------------------------------
# Load model for training
# ---------------------------------------------------------------------------
model_kwargs = dict(
    attn_implementation="eager",
    dtype=DTYPE,
    device_map={"": 0},
)

model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, **model_kwargs)
processor = AutoProcessor.from_pretrained(MODEL_ID)

# ---------------------------------------------------------------------------
# Freeze the vision encoder (train only the language model parts)
# ---------------------------------------------------------------------------
def count_parameters(m):
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return total, trainable, total - trainable


for param in model.model.vision_model.parameters():
    param.requires_grad = False

total_params, trainable_params, frozen_params = count_parameters(model)
print(f"[INFO] Total: {total_params:,} | Trainable: {trainable_params:,} | Frozen: {frozen_params:,}")

# ---------------------------------------------------------------------------
# Data collator
#
# The image placeholder token differs across processor versions/configs -
# hardcoding "<image>" can resolve to the wrong id and end up masking far
# more tokens than intended, leaving nothing for the model to learn from
# (this produced loss=0.0000 / eval_loss=nan for an entire run previously).
# get_image_token_id looks up the id the processor actually uses.
# ---------------------------------------------------------------------------
def get_image_token_id(proc):
    if getattr(proc, "image_token_id", None) is not None:
        return proc.image_token_id
    image_token = getattr(proc, "image_token", "<image>")
    if hasattr(image_token, "content"):
        image_token = image_token.content
    return proc.tokenizer.convert_tokens_to_ids(image_token)


IMAGE_TOKEN_ID = get_image_token_id(processor)
print(f"[INFO] Image placeholder token id: {IMAGE_TOKEN_ID}")


def convert_message_to_list_of_images(messages):
    image_inputs = []
    for msg in messages:
        content = msg.get("content", [])
        if not isinstance(content, list):
            content = [content]
        for element in content:
            if isinstance(element, dict) and element.get("type") == "image" and element.get("image") is not None:
                img = element["image"]
                if isinstance(img, (bytes, bytearray)):
                    img = Image.open(io.BytesIO(img))
                image_inputs.append(img.convert("RGB"))
    return image_inputs


def collate_fn(examples):
    texts, images = [], []
    for example in examples:
        images.append(convert_message_to_list_of_images(example["messages"]))
        text = processor.apply_chat_template(example["messages"], add_generation_prompt=False, tokenize=False)
        texts.append(text.strip())

    batch_input = processor(text=texts, images=images, return_tensors="pt", padding=True)

    labels = batch_input["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    labels[labels == IMAGE_TOKEN_ID] = -100
    batch_input["labels"] = labels

    return batch_input


# ---------------------------------------------------------------------------
# Sanity check: confirm the collator produces real training targets.
# Run this BEFORE kicking off a multi-hour training job. If
# valid_label_fraction is near 0, something is wrong with the label masking
# and training will silently learn nothing (as happened before this fix).
# ---------------------------------------------------------------------------
sample_batch = collate_fn([train_dataset[0], train_dataset[1]])
valid_label_fraction = (sample_batch["labels"] != -100).float().mean().item()
num_valid_tokens = int((sample_batch["labels"] != -100).sum().item())

print(f"[INFO] Batch label shape: {tuple(sample_batch['labels'].shape)}")
print(f"[INFO] Non-masked (trainable) label tokens: {num_valid_tokens}")
print(f"[INFO] Fraction of non-masked label tokens: {valid_label_fraction:.3f}")

assert num_valid_tokens > 0, "All labels are masked (-100) -- the collator is broken. Do not start training."
assert valid_label_fraction > 0.01, "Almost all labels are masked -- something is likely still wrong."
print("[INFO] Sanity check passed -- the collator has real training targets.")

# ---------------------------------------------------------------------------
# Training config
# ---------------------------------------------------------------------------
from trl import SFTConfig

MAX_STEPS = 900  # adjust based on observed step speed to hit your target run length

training_args = SFTConfig(
    output_dir=CHECKPOINT_DIR,
    max_steps=MAX_STEPS,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,  # effective batch size 16
    gradient_checkpointing=True,
    optim="adamw_torch_fused",
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,
    learning_rate=2e-4,
    fp16=False,  # model is already native bf16 -- no autocast/GradScaler needed
    bf16=False,
    max_grad_norm=1.0,
    lr_scheduler_type="constant",
    push_to_hub=False,
    report_to="wandb" if WANDB_API_KEY else "none",
    run_name="smolvlm2-pokemon-cards-v1",
    gradient_checkpointing_kwargs={"use_reentrant": False},
    dataset_text_field="",
    dataset_kwargs={"skip_prepare_dataset": True},
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    remove_unused_columns=False,
)

# ---------------------------------------------------------------------------
# Train (auto-resumes from the latest checkpoint in CHECKPOINT_DIR if one
# exists - e.g. after a job restart)
# ---------------------------------------------------------------------------
gc.collect()
torch.cuda.empty_cache()

from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=processor,
    data_collator=collate_fn,
)

last_checkpoint = None
if os.path.isdir(training_args.output_dir):
    checkpoints = [d for d in os.listdir(training_args.output_dir) if d.startswith("checkpoint-")]
    if checkpoints:
        last_checkpoint = os.path.join(
            training_args.output_dir,
            sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))[-1],
        )
        print(f"[INFO] Resuming from checkpoint: {last_checkpoint}")

trainer.train(resume_from_checkpoint=last_checkpoint)
trainer.save_model()

# ---------------------------------------------------------------------------
# Push to Hugging Face Hub
# ---------------------------------------------------------------------------
if HF_TOKEN:
    from huggingface_hub import HfApi, create_repo

    api = HfApi(token=HF_TOKEN)
    hf_username = api.whoami()["name"]
    REPO_ID = f"{hf_username}/smolvlm2-pokemon-cards-v1"

    create_repo(REPO_ID, token=HF_TOKEN, exist_ok=True, repo_type="model")
    print(f"[INFO] Repo ready: https://huggingface.co/{REPO_ID}")

    model.push_to_hub(REPO_ID, token=HF_TOKEN, commit_message="Fine-tuned SmolVLM2 on Pokemon TCG cards")
    processor.push_to_hub(REPO_ID, token=HF_TOKEN, commit_message="Add processor")

    print(f"[INFO] Push complete: https://huggingface.co/{REPO_ID}")
else:
    print("[INFO] Skipping Hugging Face push -- no HF_TOKEN env var set.")

# ---------------------------------------------------------------------------
# Lightweight before/after sanity check on one validation sample (printed to
# logs, not plotted - no one is watching a live matplotlib window during an
# unattended job)
# ---------------------------------------------------------------------------
try:
    del trainer
    gc.collect()
    torch.cuda.empty_cache()

    pt_pipe = pipeline("image-text-to-text", model=MODEL_ID, dtype=DTYPE, device=DEVICE)
    pt_pipe.model.generation_config.do_sample = False
    pt_pipe.model.generation_config.pad_token_id = processor.tokenizer.eos_token_id

    ft_pipe = pipeline("image-text-to-text", model=CHECKPOINT_DIR, dtype=DTYPE, device=DEVICE)
    ft_pipe.model.generation_config.do_sample = False
    ft_pipe.model.generation_config.pad_token_id = processor.tokenizer.eos_token_id

    random_val_sample = random.choice(val_dataset)
    sample_input = dict(random_val_sample["messages"][0])
    sample_input["content"] = [dict(c) for c in sample_input["content"]]
    sample_input["content"][0]["image"] = Image.open(io.BytesIO(sample_input["content"][0]["image"]))
    ideal_output_text = random_val_sample["messages"][-1]["content"][0]["text"]

    print(f"[INFO] Ideal output:\n{ideal_output_text}\n")
    pt_out = pt_pipe(text=[sample_input], max_new_tokens=128)
    print(f"[INFO] Base model output:\n{pt_out[0]['generated_text'][-1]['content']}\n")
    ft_out = ft_pipe(text=[sample_input], max_new_tokens=128)
    print(f"[INFO] Fine-tuned model output:\n{ft_out[0]['generated_text'][-1]['content']}\n")
except Exception as e:
    print(f"[WARN] Final before/after comparison failed (non-fatal, training already completed): {e}")

print("[INFO] Done.")
