"""
Fine-tune SmolVLM2 on Pokemon TCG Cards (structured extraction) - on Modal.

Ported from the same source as train.py (a Kaggle notebook built for 2x T4),
targeting Modal + H100 instead of Baseten + A10G. Verified against the real
installed `modal` SDK (v1.5.5), not written from memory:
  - `gpu="H100"` is a plain string in the current API (older Modal versions
    used a class-based `modal.gpu.H100()` - that's not what this version's
    `App.function` signature accepts).
  - `App.function`'s default `timeout` is only 300 seconds. Left at the
    default, a real multi-hour training run would be killed almost
    immediately. Set explicitly and generously below.
  - Heavy libraries (torch, transformers, trl, datasets, ...) are imported
    INSIDE `train()`, not at module top-level - Modal only needs the `modal`
    package itself in the environment that *defines* the app (this Windows
    machine); the actual training libraries only need to exist inside the
    remote container image, which is what `image` below builds.

Secrets: create a Modal secret named "smolvlm2-pokemon-training-secrets"
(dashboard -> Secrets -> Create new secret) with two keys: HF_TOKEN and
WANDB_API_KEY. Both are optional at runtime the same way they were in
train.py - the script skips W&B login / the HF Hub push cleanly if either
is missing - but the secret object itself must exist for `Secret.from_name`
to resolve, even if you leave a key blank.

Run with: modal run modal_train.py
(or `modal deploy modal_train.py` to keep it around instead of running once)
"""

import modal

app = modal.App("smolvlm2-pokemon-training")

image = modal.Image.debian_slim(python_version="3.11").pip_install_from_requirements(
    "requirements.txt"
)

# Persistent volume so checkpoints (and thus resume-from-checkpoint) survive
# across separate `modal run` invocations, the same role Baseten's
# checkpointing_config played in the train.py version.
checkpoint_volume = modal.Volume.from_name(
    "smolvlm2-pokemon-checkpoints", create_if_missing=True
)
CHECKPOINT_DIR = "/vol/checkpoints/smolvlm2-pokemon-cards-v1"


@app.function(
    image=image,
    gpu="H100",
    timeout=6 * 60 * 60,  # 6 hours - generous margin; H100 should finish well under this
    secrets=[modal.Secret.from_name("smolvlm2-pokemon-training-secrets")],
    volumes={"/vol/checkpoints": checkpoint_volume},
)
def train():
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

        try:
            wandb.login(key=WANDB_API_KEY, timeout=15)
            print("[INFO] Logged in to Weights & Biases.")
        except Exception as e:
            # wandb's own wandb-core subprocess can fail to report its port
            # inside a sandboxed/restricted container (seen on Modal: "Failed
            # to read port info after 30.0 seconds") - this is a wandb/
            # environment issue, not necessarily a bad key. Don't let it take
            # the whole training run down; degrade to no W&B logging instead,
            # same as the "key not provided" case below.
            print(f"[WARN] wandb.login() failed ({e}) -- continuing without W&B logging.")
            WANDB_API_KEY = None
    else:
        print("[WARN] No WANDB_API_KEY -- W&B logging will be skipped.")

    if not HF_TOKEN:
        print("[WARN] No HF_TOKEN -- the Hugging Face push step will be skipped.")

    from transformers import AutoModelForImageTextToText, AutoProcessor, pipeline

    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.bfloat16  # H100 has full bf16 tensor-core support
    MODEL_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    print(f"[INFO] Using device: {DEVICE} | dtype: {DTYPE}")

    # -----------------------------------------------------------------------
    # Load dataset
    # -----------------------------------------------------------------------
    from datasets import load_dataset

    raw_dataset = load_dataset("TheFusion21/PokemonCards")["train"]

    NUM_SAMPLES = 1500  # increase for a longer/richer run
    raw_dataset = raw_dataset.shuffle(seed=42).select(range(min(NUM_SAMPLES, len(raw_dataset))))

    # -----------------------------------------------------------------------
    # Download images and format into chat message format
    # -----------------------------------------------------------------------
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

    # -----------------------------------------------------------------------
    # Train / validation split
    # -----------------------------------------------------------------------
    from datasets import Dataset

    random.seed(42)
    dataset_shuffled = random.sample(dataset_processed, k=len(dataset_processed))

    train_split = round(0.85 * len(dataset_shuffled))
    train_dataset = Dataset.from_list(dataset_shuffled[:train_split])
    val_dataset = Dataset.from_list(dataset_shuffled[train_split:])

    print(f"[INFO] Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    # -----------------------------------------------------------------------
    # Load model for training
    # -----------------------------------------------------------------------
    model_kwargs = dict(
        attn_implementation="eager",
        dtype=DTYPE,
        device_map={"": 0},
    )

    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, **model_kwargs)
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    # -----------------------------------------------------------------------
    # Freeze the vision encoder (train only the language model parts)
    # -----------------------------------------------------------------------
    def count_parameters(m):
        total = sum(p.numel() for p in m.parameters())
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        return total, trainable, total - trainable

    for param in model.model.vision_model.parameters():
        param.requires_grad = False

    total_params, trainable_params, frozen_params = count_parameters(model)
    print(f"[INFO] Total: {total_params:,} | Trainable: {trainable_params:,} | Frozen: {frozen_params:,}")

    # -----------------------------------------------------------------------
    # Data collator - the image placeholder token differs across
    # processor versions/configs; hardcoding "<image>" can silently mask out
    # all trainable tokens (produced loss=0.0000 / eval_loss=nan previously).
    # -----------------------------------------------------------------------
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

    # -----------------------------------------------------------------------
    # Sanity check: confirm the collator produces real training targets.
    # -----------------------------------------------------------------------
    sample_batch = collate_fn([train_dataset[0], train_dataset[1]])
    valid_label_fraction = (sample_batch["labels"] != -100).float().mean().item()
    num_valid_tokens = int((sample_batch["labels"] != -100).sum().item())

    print(f"[INFO] Batch label shape: {tuple(sample_batch['labels'].shape)}")
    print(f"[INFO] Non-masked (trainable) label tokens: {num_valid_tokens}")
    print(f"[INFO] Fraction of non-masked label tokens: {valid_label_fraction:.3f}")

    assert num_valid_tokens > 0, "All labels are masked (-100) -- the collator is broken. Do not start training."
    assert valid_label_fraction > 0.01, "Almost all labels are masked -- something is likely still wrong."
    print("[INFO] Sanity check passed -- the collator has real training targets.")

    # -----------------------------------------------------------------------
    # Training config
    # -----------------------------------------------------------------------
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

    # -----------------------------------------------------------------------
    # Train (auto-resumes from the latest checkpoint on the volume if one
    # exists - e.g. after a previous `modal run` was interrupted)
    # -----------------------------------------------------------------------
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
    checkpoint_volume.commit()  # persist the volume so the checkpoint survives past this run

    # -----------------------------------------------------------------------
    # Push to Hugging Face Hub
    # -----------------------------------------------------------------------
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
        print("[INFO] Skipping Hugging Face push -- no HF_TOKEN set.")

    # -----------------------------------------------------------------------
    # Proper evaluation: structured-accuracy metrics over a batch of
    # validation samples (not just one qualitative example), base model vs
    # fine-tuned, so "did fine-tuning actually help" has a real number
    # behind it instead of an eyeballed example.
    # -----------------------------------------------------------------------
    def parse_json_output(text):
        text = text.strip()
        if text.startswith("```"):
            text = text.strip("`").strip()
            if text.lower().startswith("json"):
                text = text[4:].strip()
        try:
            return json.loads(text)
        except Exception:
            return None

    def evaluate_structured_accuracy(pipe, dataset, max_samples):
        n = min(max_samples, len(dataset))
        exact_matches = 0
        field_correct = {"name": 0, "hp": 0, "set_name": 0}
        parse_failures = 0
        for i in range(n):
            sample = dataset[i]
            sample_input = dict(sample["messages"][0])
            sample_input["content"] = [dict(c) for c in sample_input["content"]]
            sample_input["content"][0]["image"] = Image.open(
                io.BytesIO(sample_input["content"][0]["image"])
            )
            ground_truth = json.loads(sample["messages"][-1]["content"][0]["text"])

            output = pipe(text=[sample_input], max_new_tokens=128)
            generated_text = output[0]["generated_text"][-1]["content"]
            parsed = parse_json_output(generated_text)

            if parsed is None:
                parse_failures += 1
                continue

            is_exact = True
            for field in ("name", "hp", "set_name"):
                match = str(parsed.get(field, "")).strip().lower() == str(
                    ground_truth.get(field, "")
                ).strip().lower()
                if match:
                    field_correct[field] += 1
                else:
                    is_exact = False
            if is_exact:
                exact_matches += 1

        return {
            "n_samples": n,
            "parse_failures": parse_failures,
            "exact_match_accuracy": round(exact_matches / n, 4),
            "field_accuracy": {k: round(v / n, 4) for k, v in field_correct.items()},
        }

    metrics = {"final_eval_loss": None, "base_model": None, "fine_tuned_model": None}
    try:
        eval_losses = [log["eval_loss"] for log in trainer.state.log_history if "eval_loss" in log]
        if eval_losses:
            metrics["final_eval_loss"] = eval_losses[-1]
    except Exception:
        pass

    try:
        del trainer
        gc.collect()
        torch.cuda.empty_cache()

        EVAL_SAMPLES = min(100, len(val_dataset))

        pt_pipe = pipeline("image-text-to-text", model=MODEL_ID, dtype=DTYPE, device=DEVICE)
        pt_pipe.model.generation_config.do_sample = False
        pt_pipe.model.generation_config.pad_token_id = processor.tokenizer.eos_token_id
        metrics["base_model"] = evaluate_structured_accuracy(pt_pipe, val_dataset, EVAL_SAMPLES)
        print(f"[INFO] Base model metrics ({EVAL_SAMPLES} val samples): {metrics['base_model']}")
        del pt_pipe
        gc.collect()
        torch.cuda.empty_cache()

        ft_pipe = pipeline("image-text-to-text", model=CHECKPOINT_DIR, dtype=DTYPE, device=DEVICE)
        ft_pipe.model.generation_config.do_sample = False
        ft_pipe.model.generation_config.pad_token_id = processor.tokenizer.eos_token_id
        metrics["fine_tuned_model"] = evaluate_structured_accuracy(ft_pipe, val_dataset, EVAL_SAMPLES)
        print(f"[INFO] Fine-tuned model metrics ({EVAL_SAMPLES} val samples): {metrics['fine_tuned_model']}")

        # one qualitative example alongside the numbers, for a human-readable sanity check
        random_val_sample = random.choice(val_dataset)
        sample_input = dict(random_val_sample["messages"][0])
        sample_input["content"] = [dict(c) for c in sample_input["content"]]
        sample_input["content"][0]["image"] = Image.open(io.BytesIO(sample_input["content"][0]["image"]))
        ideal_output_text = random_val_sample["messages"][-1]["content"][0]["text"]
        ft_out = ft_pipe(text=[sample_input], max_new_tokens=128)
        print(f"[INFO] Example - ideal: {ideal_output_text}")
        print(f"[INFO] Example - fine-tuned output: {ft_out[0]['generated_text'][-1]['content']}")
    except Exception as e:
        print(f"[WARN] Final evaluation failed (non-fatal, training + HF push already completed): {e}")

    print(f"[INFO] Final metrics summary: {json.dumps(metrics, indent=2)}")
    print("[INFO] Done.")
    return metrics


@app.local_entrypoint()
def main():
    metrics = train.remote()
    print("Training complete. Final metrics:")
    print(metrics)
