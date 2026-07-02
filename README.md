# Temperature Bench

A resumable benchmark for measuring how local language-model output degrades as
sampling temperature rises. It generates the same factual, creative, and reasoning
tasks at `T = 0.5, 1.0, 1.5, 2.0, 2.5, 3.0`, computes deterministic degeneration
metrics, and sends outputs to Gemini for blind structured scoring.

The default matrix contains 16 enabled checkpoints, 6 tasks, and 6 temperatures:
**576 local generations and up to 576 Gemini requests**. The two Gemma 3 27B
checkpoints remain defined but disabled for 40 GB GPUs. Models are loaded one at a time because a
27B checkpoint can consume most of a large GPU. Gemini judging runs concurrently in a
thread pool while the next local batches generate.

## Server setup

Prerequisites:

- Linux, an NVIDIA driver, and enough GPU memory for the chosen precision.
- Python 3.11–3.13.
- Roughly 500 GB of free model-cache space if every BF16 checkpoint is retained.
- A Gemini API key.
- A Hugging Face account/token with the Llama 3.1 and Gemma license terms accepted.

```bash
git clone YOUR_REPOSITORY_URL
cd YOUR_REPOSITORY
cp -n .env.example .env
# Put GEMINI_API_KEY and HF_TOKEN in .env

./scripts/bootstrap.sh
source .venv/bin/activate
tempbench preflight --remote
tempbench plan
```

If the default PyTorch wheel does not match the server, use the wheel index recommended
by the [PyTorch installer](https://pytorch.org/get-started/locally/):

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 ./scripts/bootstrap.sh
```

The script deliberately does not install FlashAttention: SDPA works across more GPUs
and models. After the baseline works, FlashAttention can be tested separately.
It also detects pre-CUDA-13 NVIDIA drivers and installs the official CUDA 11.8
PyTorch 2.7.1 wheels instead of an incompatible newer CUDA runtime.

`HF_HOME` does not need to be configured on managed Jupyter servers. When it is unset,
the runner compares the default home cache with `/home/jovyan/.cache/huggingface` and
uses the writable location with more free space. An explicitly configured `HF_HOME`
still takes precedence.

## Running

Start or resume the complete benchmark:

```bash
tempbench --run-name baseline run 2>&1 | tee baseline.log
```

Run a small real-GPU pilot first:

```bash
tempbench --run-name pilot \
  --models mistral-7b-v0.3-instruct \
  --prompts everest_facts,constrained_lighthouse_story run
```

Useful split-stage commands:

```bash
# Generate without making Gemini requests.
tempbench --run-name baseline generate

# Judge every generated output that has no judgment yet.
tempbench --run-name baseline judge

# Rebuild reports at any time.
tempbench --run-name baseline report
```

Every completed generation and judgment is an atomic JSON file. Repeating the same
command skips completed work. `Ctrl-C`, an API quota error, a model-access error, and a
GPU OOM therefore do not discard prior results. Failures are written under
`results/RUN_NAME/failures/`.

Reports are written to:

- `results/RUN_NAME/reports/all_results.csv`
- `results/RUN_NAME/reports/summary_by_model_temperature.csv`
- `results/RUN_NAME/reports/report.html`

## Resource controls

Edit `configs/benchmark.yaml`:

- `batch_size`: starts at 2. An OOM automatically retries the batch in halves.
- `dtype`: BF16 by default. Use FP16 only if the GPU lacks BF16 support.
- `quantization`: `none`, `4bit`, or `8bit`. Quantization reduces memory, but it is a
  real experimental confound and newer custom architectures may not support it.
- `max_new_tokens`: output cap (1,400 by default). Prompts also specify target word ranges.
- `samples_per_condition`: 1 for the initial requested run.
- `judge.concurrency`: simultaneous Gemini calls; reduce it if the API quota is low.
- `judge.model`: defaults to `gemini-3-flash-preview`.

With several GPUs, expose all GPUs and retain `device_map: auto`:

```bash
CUDA_VISIBLE_DEVICES=0,1 tempbench --run-name baseline run
```

Do not launch multiple full runner processes against the same result directory.

## Experimental design

The judge receives the prompt, task-specific reference facts, and anonymous output. It
does not receive model identity or temperature. Scores are 1–10 for coherence, internal
consistency, instruction following, fluency, overall quality, and factuality or
creativity where applicable.

Local metrics independently record word count, lexical diversity, repeated bigram and
trigram rates, and machine-checkable constraints. Sampling keeps `top_p=1` and `top_k=0`
so temperature is the manipulated decoding parameter. Conditions use paired seeds and
stable batch ordering across temperatures.

One output per condition is suitable for a pilot, not a strong statistical conclusion.
For a serious comparison, set:

```yaml
samples_per_condition: 5
```

Use a new run name after changing configuration. The runner rejects attempts to mix
different configurations in one run directory.

Limitations:

- Gemini is still a model judge. Inspect outputs and factual-claim assessments manually.
- The reference sheets improve factual scoring but are not exhaustive.
- Base and instruction-tuned checkpoints require different native prompt formatting.
  Base checkpoints receive a plain `Task/Response` completion; instruct checkpoints use
  each tokenizer's chat template. This is intentional but should be disclosed.
- A model repository can change. Each generation records the resolved Hugging Face
  commit when Transformers exposes it. Pin `revision` in `configs/models.yaml` before a
  publication-grade run.
- At very high temperatures, some outputs will hit the token cap; that behavior is part
  of the degradation being measured.

## Checkpoints and compatibility

Checkpoint IDs live in `configs/models.yaml`. Set a model's `enabled` field to select
whether it participates by default; explicitly naming a disabled model with `--models`
still runs it. The newer Qwen 3.5, Gemma 3/4, and NVIDIA
models use multimodal or custom loaders even though this benchmark supplies text only.
NVIDIA Nemotron Labs Diffusion requires `trust_remote_code`; it is enabled only for that
explicit model entry. Its benchmark adapter uses the model's AR generation mode because
temperature has the conventional token-sampling interpretation there; diffusion and
self-speculation use different controls and would constitute a separate experiment.
Transformers is constrained to major version 5 because Gemma 4
and Nemotron Labs Diffusion require the newer architecture support.

Before running, accept access terms on the official
[Llama 3.1](https://huggingface.co/meta-llama/Llama-3.1-8B),
[Gemma](https://huggingface.co/google/gemma-3-12b-it), and any other gated checkpoint
pages while signed into the account represented by `HF_TOKEN`.

## Development checks

```bash
source .venv/bin/activate
pip install -e '.[dev,gpu]'
pytest -q
ruff check .
```
