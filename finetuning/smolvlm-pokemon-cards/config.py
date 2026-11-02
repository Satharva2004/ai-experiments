"""
Baseten Training config for smolvlm2-pokemon-training.

Verified against the real installed truss_train package (not copied from
docs without checking) - two things a fetched doc example got wrong that
this file deliberately avoids:
  1. SecretReference's actual field is `name`, not `secret_name`
     (truss_train/definitions.py: `class SecretReference(...): name: str`,
     and it's a SafeModelNoExtra model - extra="forbid" - so a wrong key
     would hard-fail validation, not silently ignore).
  2. The real CLI command is `truss train push`, not `baseten train push`
     (confirmed via `truss train --help`).

Base image pinned to torch==2.9.1+cu128's matching official image
(pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime, confirmed to exist on
Docker Hub) to match the exact torch/CUDA combination already proven
working on Baseten's H100/A10G infra for every other deployment in this
repo, rather than an arbitrary/older version.

Run with: truss train push config.py
"""

from truss.base.truss_config import AcceleratorSpec

from truss_train import (
    CacheConfig,
    CheckpointingConfig,
    Compute,
    Image,
    Runtime,
    SecretReference,
    TrainingJob,
    TrainingProject,
)

BASE_IMAGE = "pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime"
CHECKPOINT_PATH = "/tmp/training_checkpoints/smolvlm2-pokemon-cards-v1"

training_job = TrainingJob(
    image=Image(base_image=BASE_IMAGE),
    compute=Compute(
        accelerator=AcceleratorSpec(accelerator="A10G", count=1),
        cpu_count=4,
        memory="16Gi",
    ),
    runtime=Runtime(
        start_commands=[
            "pip install -r requirements.txt",
            "python train.py",
        ],
        environment_variables={
            # Deliberately NOT named "hf_access_token" - that name is
            # already used by other deployments in this repo
            # (gemma3-v12-merged-vllm's weight pulling, gemma-3-27b-it's
            # gated-model download) to authenticate their own HF pulls.
            # Reusing it here would mean this training job's token
            # overwrites/competes with whatever those deployments expect on
            # their next fresh redeploy - use a distinct, job-scoped secret
            # name instead so this project can never affect them.
            # "wandb_api_key" is optional (train.py skips W&B logging
            # cleanly if unset); "hf_access_token_pokemon_training" is
            # optional too (the final HF push step skips cleanly if unset),
            # but without it you won't get the fine-tuned weights anywhere
            # durable other than the checkpoint volume.
            "HF_TOKEN": SecretReference(name="hf_access_token_pokemon_training"),
            "WANDB_API_KEY": SecretReference(name="wandb_api_key"),
        },
        cache_config=CacheConfig(enabled=True),
        checkpointing_config=CheckpointingConfig(
            enabled=True,
            checkpoint_path=CHECKPOINT_PATH,
        ),
    ),
)

training_project = TrainingProject(
    name="smolvlm2-pokemon-cards",
    job=training_job,
)
