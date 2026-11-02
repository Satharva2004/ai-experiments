import os
import subprocess

import modal

MODEL_ID = "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16"
SERVED_NAME = "nemotron-3.5-super-vl"
GPU = "H100:4"  # model card: 4x H100, tensor parallel 4
PORT = 8000

# Official vLLM image, the version NVIDIA lists on the model card.
image = (
    modal.Image.from_registry("vllm/vllm-openai:v0.32.0")
    .entrypoint([])  # drop the image's default `vllm serve` entrypoint
    .run_commands("command -v python || ln -s $(command -v python3) /usr/local/bin/python")
    .env({"HF_HOME": "/root/.cache/huggingface"})
)

# Persistent storage so weights and compiled kernels survive restarts.
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("vllm-cache", create_if_missing=True)
volumes = {"/root/.cache/huggingface": hf_cache, "/root/.cache/vllm": vllm_cache}

# Holds HF_TOKEN and VLLM_API_KEY. Created from .env (see README).
secrets = [modal.Secret.from_name("nemotron-secrets")]

app = modal.App("nemotron-vl")


@app.function(image=image, volumes=volumes, secrets=secrets, timeout=2 * 60 * 60)
def download():
    from huggingface_hub import snapshot_download

    snapshot_download(MODEL_ID)
    hf_cache.commit()
    print("Weights cached in the hf-cache volume.")


@app.function(
    image=image,
    gpu=GPU,
    volumes=volumes,
    secrets=secrets,
    timeout=60 * 60,
    scaledown_window=15 * 60,  # shut down after 15 idle minutes (stops billing)
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=PORT, startup_timeout=45 * 60)
def serve():
    cmd = [
        "vllm", "serve", MODEL_ID,
        "--host", "0.0.0.0",
        "--port", str(PORT),
        "--served-model-name", SERVED_NAME,
        "--api-key", os.environ["VLLM_API_KEY"],  # from the Modal secret
        "--tensor-parallel-size", "4",
        "--trust-remote-code",
        "--gpu-memory-utilization", "0.85",
        "--max-num-seqs", "32",
        "--limit-mm-per-prompt", '{"video": 1, "image": 1}',
        "--video-pruning-rate", "0.5",
        "--media-io-kwargs", '{"video": {"fps": 2, "num_frames": 256}}',
        "--enable-auto-tool-choice",
        "--tool-call-parser", "qwen3_coder",
        "--reasoning-parser", "nemotron_v3",
        "--speculative_config.method", "mtp",
        "--speculative_config.num_speculative_tokens", "3",
    ]
    subprocess.Popen(cmd)
