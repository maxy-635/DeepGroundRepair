import torch
from loguru import logger
from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend, FineGrainedFP8Config


class Chat2MistralLLM:
    """Load a local Mistral model and generate responses."""

    def __init__(self):
        pass
    
    def model_id_mapping(self, model_id, cache_dir):
        if model_id == "mistralai/Ministral-3-3B-Instruct-2512":
            model_id = "models--mistralai--Ministral-3-3B-Instruct-2512/snapshots/7046a0e237b436c8fb4927061ab3773772e53741"
        if model_id == "mistralai/Ministral-3-14B-Instruct-2512":
            model_id = "models--mistralai--Ministral-3-14B-Instruct-2512/snapshots/6118730123a6291757f257ca95b41aac91bc1d9b"
        if model_id == "mistralai/Mistral-Small-3.2-24B-Instruct-2506":
            model_id = "models--mistralai--Mistral-Small-3.2-24B-Instruct-2506/snapshots/95a6d26c4bfb886c58daf9d3f7332c857cb27b43"

        model_id = cache_dir + "/" + model_id

        return model_id

    def load_model(self, model_id, cache_dir, data_dtype):
        
        model_id = self.model_id_mapping(model_id, cache_dir)

        tokenizer = MistralCommonBackend.from_pretrained(
            pretrained_model_name_or_path=model_id,
            cache_dir=cache_dir,
            local_files_only=True
        )

        model = Mistral3ForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=model_id,
            device_map="auto",
            torch_dtype=data_dtype,
            cache_dir=cache_dir,
            local_files_only=True,
            quantization_config=FineGrainedFP8Config(dequantize=True),
            attn_implementation="eager"
        )

        return model, tokenizer

    def chat(self, model, tokenizer, prompt, temperature, top_p, max_new_tokens):

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
