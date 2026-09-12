import torch
from loguru import logger
from transformers import AutoTokenizer, AutoModelForCausalLM

class Chat2GemmaLLM:
    """Load a local Gemma model and generate responses."""

    def __init__(self):
        pass

    def load_model(self, model_id, cache_dir, data_dtype):

        tokenizer = AutoTokenizer.from_pretrained(
            model_id, 
            cache_dir=cache_dir,
            local_files_only=True
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=data_dtype,
            cache_dir=cache_dir,
            local_files_only=True,
            attn_implementation="eager"
        )

        return model, tokenizer

    def chat(self, model, tokenizer, prompt, temperature, top_p, max_new_tokens):

        chat = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to("cuda")

        logger.info(f"Input Prompt tokens: {inputs.shape[1]}")

        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True, 
            temperature=temperature,
            top_p=top_p
        )

        response_token_ids = outputs[0][inputs.shape[1]:]
        logger.info(f"LLM generated response tokens: {response_token_ids.shape[0]}")
        response = tokenizer.decode(response_token_ids, skip_special_tokens=True).strip()

        del inputs
        del outputs
        torch.cuda.empty_cache()

        return response
