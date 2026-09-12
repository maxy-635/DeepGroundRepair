from __future__ import annotations
import torch
from typing import Any, Callable
from loguru import logger
from utils.chat2gemma import Chat2GemmaLLM
from utils.chat2mistral import Chat2MistralLLM
from utils.utils import get_sampling_temperature


class LLMRuntime:
    """Implement the llmruntime component."""

    def __init__(self) -> None:
        raise RuntimeError("Use LLMRuntime.hf(...) instead.")

    @classmethod
    def hf(
        cls,
        model_id: str,
        cache_dir: str | None = None,
        data_dtype: Any = torch.bfloat16,
        temperature: float | None = None,
        top_p: float = 0.95,
        max_tokens: int = 3000,
    ) -> "LLMRuntime":
        model_key = model_id.lower()
        temperature = get_sampling_temperature(model_id) if temperature is None else temperature
        logger.bind(sampling_config=True).info(
            f"Sampling configuration: model_id={model_id}, temperature={temperature}"
        )
        if "gemma" in model_key:
            chat_class = Chat2GemmaLLM
        elif "mistral" in model_key:
            chat_class = Chat2MistralLLM
        else:
            raise ValueError(f"Unsupported HuggingFace model id: {model_id}")

        chater = chat_class()
        model, tokenizer = chater.load_model(
            model_id,
            cache_dir,
            data_dtype,
        )
        runtime = object.__new__(cls)
        runtime.model_id = model_id
        runtime._chat: Callable[[str], Any] = lambda prompt: chater.chat(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_tokens
        )
        return runtime

    def chat(self, prompt: str) -> Any:
        """Generate a model response."""
        return self._chat(prompt)
