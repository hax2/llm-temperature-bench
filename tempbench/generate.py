from __future__ import annotations

import gc
import platform
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from rich.console import Console

from tempbench.config import ModelSpec, PromptSpec, RunSettings
from tempbench.metrics import local_metrics
from tempbench.storage import Job

console = Console()


class TransformersGenerator:
    """Loads one checkpoint at a time and generates batches on the visible GPU(s)."""

    def __init__(self, spec: ModelSpec, settings: RunSettings):
        self.spec = spec
        self.settings = settings
        self.model = None
        self.processor = None
        self.torch = None

    def __enter__(self) -> "TransformersGenerator":
        self.load()
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def load(self) -> None:
        import torch
        import transformers
        from transformers import AutoProcessor, AutoTokenizer

        self.torch = torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; local generation requires an NVIDIA GPU")

        token = _hf_token()
        common: dict[str, Any] = {
            "device_map": self.settings.device_map,
            "dtype": _torch_dtype(torch, self.settings.dtype),
            "low_cpu_mem_usage": self.settings.low_cpu_mem_usage,
            "trust_remote_code": self.spec.trust_remote_code or self.settings.trust_remote_code,
            "token": token,
        }
        if self.spec.revision:
            common["revision"] = self.spec.revision
        if self.settings.attn_implementation:
            common["attn_implementation"] = self.settings.attn_implementation
        if self.settings.quantization != "none":
            from transformers import BitsAndBytesConfig

            compute_dtype = _torch_dtype(torch, self.settings.dtype)
            if compute_dtype == "auto":
                compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            common["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=self.settings.quantization == "4bit",
                load_in_8bit=self.settings.quantization == "8bit",
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )

        console.print(
            f"[cyan]Loading[/cyan] {self.spec.id} ({self.spec.repo}) "
            f"with transformers {transformers.__version__}"
        )
        processor_error: Exception | None = None
        if self.spec.loader in {"multimodal", "custom"}:
            try:
                self.processor = AutoProcessor.from_pretrained(
                    self.spec.repo,
                    trust_remote_code=common["trust_remote_code"],
                    token=token,
                    revision=self.spec.revision,
                )
            except Exception as exc:
                processor_error = exc
        if self.processor is None:
            try:
                self.processor = AutoTokenizer.from_pretrained(
                    self.spec.repo,
                    trust_remote_code=common["trust_remote_code"],
                    token=token,
                    revision=self.spec.revision,
                    use_fast=True,
                )
            except Exception:
                if processor_error is not None:
                    raise RuntimeError(
                        f"Neither processor nor tokenizer loaded for {self.spec.repo}. "
                        f"Processor error: {processor_error}"
                    ) from processor_error
                raise

        self.model = self._load_model(common)
        tokenizer = self._tokenizer()
        if tokenizer is not None:
            tokenizer.padding_side = "left"
            if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
        self.model.eval()
        console.print(f"[green]Loaded[/green] {self.spec.id}")

    def _load_model(self, common: dict[str, Any]):
        from transformers import AutoModel, AutoModelForCausalLM

        candidates: list[tuple[str, Any]] = []
        if self.spec.loader == "custom":
            candidates.append(("AutoModel", AutoModel))
        elif self.spec.loader == "causal":
            candidates.append(("AutoModelForCausalLM", AutoModelForCausalLM))
        else:
            try:
                from transformers import AutoModelForImageTextToText

                candidates.append(("AutoModelForImageTextToText", AutoModelForImageTextToText))
            except ImportError:
                pass
            try:
                from transformers import AutoModelForMultimodalLM

                candidates.append(("AutoModelForMultimodalLM", AutoModelForMultimodalLM))
            except ImportError:
                pass
            candidates.append(("AutoModelForCausalLM", AutoModelForCausalLM))

        failures = []
        for name, candidate in candidates:
            try:
                return candidate.from_pretrained(self.spec.repo, **common)
            except (ValueError, TypeError, KeyError) as exc:
                failures.append(f"{name}: {type(exc).__name__}: {exc}")
        raise RuntimeError(
            f"No compatible model loader for {self.spec.repo}:\n" + "\n".join(failures)
        )

    def generate_batch(
        self,
        items: list[tuple[Job, PromptSpec]],
        on_result: Callable[[Job, PromptSpec, dict[str, Any]], None],
    ) -> None:
        if not items:
            return
        try:
            self._generate_batch_once(items, on_result)
        except self.torch.cuda.OutOfMemoryError:
            self.torch.cuda.empty_cache()
            if len(items) == 1:
                raise
            middle = len(items) // 2
            console.print(
                f"[yellow]GPU OOM[/yellow] at batch {len(items)}; retrying as "
                f"{middle} + {len(items) - middle}"
            )
            self.generate_batch(items[:middle], on_result)
            self.generate_batch(items[middle:], on_result)

    def _generate_batch_once(
        self,
        items: list[tuple[Job, PromptSpec]],
        on_result: Callable[[Job, PromptSpec, dict[str, Any]], None],
    ) -> None:
        prompts = [self._render_prompt(prompt) for _, prompt in items]
        temperature = items[0][0].temperature
        started = time.monotonic()
        self.torch.manual_seed(items[0][0].seed)
        self.torch.cuda.manual_seed_all(items[0][0].seed)

        encoded = self._encode(prompts)
        encoded = {key: value.to(self._input_device()) for key, value in encoded.items()}
        input_length = encoded["input_ids"].shape[1]
        generate_kwargs = {
            "max_new_tokens": self.settings.max_new_tokens,
            "do_sample": True,
            "temperature": temperature,
            "top_p": 1.0,
            "top_k": 0,
            "use_cache": True,
        }
        tokenizer = self._tokenizer()
        if tokenizer is not None and tokenizer.pad_token_id is not None:
            generate_kwargs["pad_token_id"] = tokenizer.pad_token_id

        with self.torch.inference_mode():
            outputs = self._call_generate(encoded, generate_kwargs)
        elapsed = time.monotonic() - started
        generated = outputs[:, input_length:] if outputs.shape[1] > input_length else outputs
        texts = self.processor.batch_decode(generated, skip_special_tokens=True)
        pad_id = tokenizer.pad_token_id if tokenizer is not None else None
        token_counts = [
            int(row.shape[0] if pad_id is None else (row != pad_id).sum().item())
            for row in generated
        ]

        for (job, prompt), text, token_count in zip(items, texts, token_counts, strict=True):
            record = {
                "schema_version": 1,
                "job": job.to_dict(),
                "model": {
                    "id": self.spec.id,
                    "repo": self.spec.repo,
                    "revision": self.spec.revision,
                    "variant": self.spec.variant,
                    "loader": self.spec.loader,
                    "resolved_revision": getattr(self.model.config, "_commit_hash", None),
                },
                "prompt": {
                    "id": prompt.id,
                    "title": prompt.title,
                    "category": prompt.category,
                    "text": prompt.prompt,
                },
                "sampling": {
                    "temperature": temperature,
                    "top_p": 1.0,
                    "top_k": 0,
                    "do_sample": True,
                    "max_new_tokens": self.settings.max_new_tokens,
                    "paired_job_seed": job.seed,
                    "actual_batch_seed": items[0][0].seed,
                    "quantization": self.settings.quantization,
                },
                "output": text.strip(),
                "local_metrics": local_metrics(text, prompt),
                "runtime": {
                    "batch_size": len(items),
                    "batch_elapsed_seconds": round(elapsed, 3),
                    "generated_token_count": token_count,
                    "hit_token_cap": token_count >= self.settings.max_new_tokens - 1,
                    "generated_at": datetime.now(UTC).isoformat(),
                    "host": platform.node(),
                    "torch_version": self.torch.__version__,
                    "cuda_version": self.torch.version.cuda,
                },
            }
            on_result(job, prompt, record)

    def _render_prompt(self, prompt: PromptSpec) -> str:
        if self.spec.variant == "base":
            return f"Task:\n{prompt.prompt.strip()}\n\nResponse:\n"
        messages = [{"role": "user", "content": prompt.prompt.strip()}]
        apply_template = getattr(self.processor, "apply_chat_template", None)
        if apply_template is None:
            tokenizer = self._tokenizer()
            apply_template = getattr(tokenizer, "apply_chat_template", None)
        if apply_template is None:
            return f"User: {prompt.prompt.strip()}\nAssistant:"
        return apply_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **self.spec.chat_template_kwargs,
        )

    def _call_generate(self, encoded: dict[str, Any], kwargs: dict[str, Any]):
        if self.spec.loader != "custom" or not hasattr(self.model, "ar_generate"):
            return self.model.generate(**encoded, **kwargs)

        # Nemotron Labs Diffusion exposes separate AR/diffusion/self-speculation
        # methods. Temperature has the standard token-sampling meaning in AR mode,
        # so that is the controlled path for this benchmark.
        import inspect

        method = self.model.ar_generate
        parameters = inspect.signature(method).parameters
        accepts_kwargs = any(
            value.kind is inspect.Parameter.VAR_KEYWORD for value in parameters.values()
        )
        if "temperature" not in parameters and not accepts_kwargs:
            raise RuntimeError(
                f"{self.spec.repo} ar_generate does not expose temperature; "
                "refusing to produce a mislabeled temperature condition"
            )
        filtered = (
            kwargs
            if accepts_kwargs
            else {key: value for key, value in kwargs.items() if key in parameters}
        )
        result = method(encoded["input_ids"], **filtered)
        return result[0] if isinstance(result, tuple) else result

    def _encode(self, prompts: list[str]) -> dict[str, Any]:
        try:
            return self.processor(
                text=prompts,
                return_tensors="pt",
                padding=True,
                truncation=False,
            )
        except TypeError:
            return self.processor(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=False,
            )

    def _tokenizer(self):
        return getattr(self.processor, "tokenizer", self.processor)

    def _input_device(self):
        try:
            return self.model.device
        except AttributeError:
            return next(self.model.parameters()).device

    def close(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        gc.collect()
        if self.torch is not None and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
            self.torch.cuda.ipc_collect()


def _torch_dtype(torch, name: str):
    if name == "auto":
        return "auto"
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def _hf_token() -> str | None:
    import os

    return os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
