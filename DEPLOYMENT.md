# Deploying HealthPost on Hugging Face Spaces (ZeroGPU)

This guide covers deploying HealthPost to a free Hugging Face Space with ZeroGPU,
as well as running it locally for development.

---

## Prerequisites

- A [Hugging Face account](https://huggingface.co/join)
- Access granted to the gated models:
  - [google/medgemma-4b-it](https://huggingface.co/google/medgemma-4b-it) -- click **Request access** on the model card
  - [google/medasr](https://huggingface.co/google/medasr) -- click **Request access** if gated
- A Hugging Face **access token** with `read` scope -- create one at [hf.co/settings/tokens](https://huggingface.co/settings/tokens)

---

## 1 -- Create the Space

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space).
2. Fill in the form:

   | Field | Value |
   |-------|-------|
   | **Owner** | your username or org |
   | **Space name** | `healthpost` (or any name you like) |
   | **SDK** | **Gradio** |
   | **Hardware** | **ZeroGPU** |
   | **Visibility** | Public or Private |

3. Click **Create Space**.

---

## 2 -- Add Your HF Token as a Secret

The models are gated, so the Space needs your token to download them.

1. Open your new Space and go to **Settings > Variables and secrets**.
2. Click **New secret** and add:

   | Name | Value |
   |------|-------|
   | `HF_TOKEN` | `hf_your_token_here` |

The `transformers` library reads this automatically -- no code changes needed.

---

## 3 -- Push the Code

Clone your Space repo locally and copy the project files into it:

```bash
# Clone the empty Space repo
git clone https://huggingface.co/spaces/YOUR_USERNAME/healthpost
cd healthpost

# Copy project files (from your local gemma repo)
cp -r /path/to/gemma/* .

# Commit and push
git add .
git commit -m "Initial deployment"
git push
```

Alternatively, if you already have the `gemma` repo, add the Space as a remote:

```bash
cd /path/to/gemma
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/healthpost
git push space master:main
```

### Required files at the repo root

The Space expects at minimum:

```
app.py                  # Gradio entrypoint (already configured)
requirements.txt        # Dependencies (already configured)
healthpost/             # Application package
data/                   # Static data assets (drug database, etc.)
```

---

## 4 -- How It Works on ZeroGPU

ZeroGPU provides **free GPU access** on Hugging Face Spaces. Instead of a
dedicated GPU, you get a shared GPU allocated on-demand for each function call.

### GPU allocation in the code

Functions that need GPU are decorated with `@spaces.GPU(duration=N)`, where `N`
is the maximum number of seconds the function may hold the GPU:

| Function | Duration | Purpose |
|----------|----------|---------|
| `_transcribe_audio_gpu` | 180s | Voice-to-text via MedASR |
| `_analyze_image_gpu` | 180s | Medical image analysis |
| `_diagnose_gpu` | 180s | Diagnosis + treatment plan |
| `_run_pipeline_gpu` | 180s | Full patient visit workflow |
| `_chat_respond_gpu` | 180s | Follow-up chat responses |

When the Space runs **outside** HF infrastructure (e.g. locally), a no-op
fallback is used so the decorators have no effect:

```python
try:
    import spaces
except ImportError:
    class spaces:
        @staticmethod
        def GPU(duration=60):
            return lambda fn: fn
```

### 4-bit quantization

MedGemma 4B is loaded in 4-bit precision by default (`hf_use_4bit=True` in
`Config`). This brings VRAM usage down to ~4 GB, fitting comfortably in
ZeroGPU's T4/A10G allocation. To disable quantization (e.g. if you have more
VRAM), set `hf_use_4bit=False` when constructing the `Config`.

---

## 5 -- First Launch

After pushing, the Space will:

1. **Install dependencies** from `requirements.txt` (this includes `torch`,
   `transformers`, `bitsandbytes`, etc.).
2. **Start `app.py`** -- Gradio launches on port 7860.
3. **On the first request**, the models are downloaded from the Hub and cached.
   This takes a few minutes the first time; subsequent restarts use the cache.

You can monitor progress in the Space's **Logs** tab.

### What to verify

- The Space status turns green ("Running").
- Load a demo scenario (e.g. "Malaria Case") and click **Run Complete Workflow**.
- Confirm the diagnostic report renders with diagnosis, treatment plan, and drug
  safety check.
- Test image analysis by uploading a medical photo.
- Test follow-up chat by asking a question about the diagnosis.
- Drug interaction checking is API-based (DDInter) and does not need GPU.

---

## 6 -- Running Locally

The same codebase runs locally without any HF Spaces infrastructure.

### CPU-only (no GPU)

```bash
# Install dependencies
pip install -r requirements.txt

# Set your HF token (needed for gated models)
export HF_TOKEN=hf_your_token_here      # Linux/macOS
set HF_TOKEN=hf_your_token_here         # Windows cmd
$env:HF_TOKEN="hf_your_token_here"      # PowerShell

# Run the app
python app.py
```

The app starts at `http://localhost:7860`. Inference will run on CPU (slow but
functional). The `@spaces.GPU` decorators are silently ignored.

### With a local GPU

If you have a CUDA-capable GPU, `device_map="auto"` in the backend will
automatically place the model on GPU. No config changes are needed -- just make
sure `torch` is installed with CUDA support:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Using the Ollama backend (alternative)

The `OllamaBackend` class is still available in `inference_backend.py` for local
development with an Ollama server. To switch to it, edit `create_backend()`:

```python
def create_backend(config: Config) -> InferenceBackend:
    return OllamaBackend(config)
```

Then start Ollama with a compatible model:

```bash
ollama pull hf.co/unsloth/medgemma-1.5-4b-it-GGUF:latest
```

### Creating a public link

To share the local instance temporarily via a Gradio tunnel:

```bash
python app.py --share
```

---

## 7 -- Configuration Reference

All settings live in `healthpost/config.py` as a `Config` dataclass:

| Field | Default | Description |
|-------|---------|-------------|
| `hf_model_id` | `google/medgemma-4b-it` | HF model ID for text and vision |
| `medasr_model_id` | `google/medasr` | HF model ID for speech-to-text |
| `device` | `cpu` | Torch device (`cpu` / `cuda`). ZeroGPU assigns dynamically. |
| `hf_use_4bit` | `True` | Enable 4-bit quantization (recommended for ZeroGPU) |
| `max_new_tokens` | `512` | Maximum tokens per generation call |
| `temperature` | `0.3` | Sampling temperature |
| `confidence_threshold` | `0.7` | Minimum confidence before recommending referral |
| `sample_rate` | `16000` | Audio sample rate in Hz |
| `data_dir` | `./data` | Path to static data assets |

---

## 8 -- Troubleshooting

### "Model is gated" or 401 errors

Your `HF_TOKEN` secret is missing or the token does not have access to the
model. Double-check:
- The token is set correctly in **Settings > Secrets**.
- You have accepted the model license on the model card page.

### Out of memory on ZeroGPU

Ensure `hf_use_4bit` is `True` (the default). If the error persists, the model
may be competing for VRAM with another process on the shared node -- try
restarting the Space.

### `bitsandbytes` import error

This can happen on CPU-only machines. The backend gracefully falls back to full
precision when `bitsandbytes` is unavailable. The warning
`"bitsandbytes not available; loading without quantization"` is expected in that
case.

### Slow first request

The first request downloads and loads the model weights (~4 GB at 4-bit). This
is a one-time cost per Space restart. Subsequent requests reuse the loaded model.

### `spaces` module not found (local dev)

This is normal. The fallback class in `app.py` handles it. You only need the
`spaces` package when running on HF Spaces infrastructure.

### GPU timeout errors

If a ZeroGPU function exceeds its `duration` limit, it gets killed. The current
limit of 180s (3 minutes) per function is generous for MedGemma 4B at 4-bit. If you switch to a
larger model, increase the `duration` values in the `@spaces.GPU()` decorators
in `app.py`.
