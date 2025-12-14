# =========================================================
#   FastDLLM Persistent Model Wrapper (final version)
# =========================================================
import sys
import os
import torch
import time
from transformers import AutoTokenizer, AutoConfig, GenerationConfig

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
fastdllm_path = os.path.join(repo_root, "FastDLLM_inferencing")
sys.path.insert(0, repo_root)
sys.path.insert(0, fastdllm_path)

from Fast_dLLM_v2_7B.modeling import Fast_dLLM_QwenForCausalLM

# Global singleton
_model_instance = None


class ModelWrapper:
    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[ModelWrapper] Loading model '{model_path}' on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

        self.model = Fast_dLLM_QwenForCausalLM.from_pretrained(
            model_path,
            config=config,
            trust_remote_code=True,
            dtype="auto",
            device_map="auto"
        )

        gen_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True)
        self.model.generation_config = gen_config

        self.total_elapsed_time = 0.0
        self.absolute_gpu_peak_memory = 0

        self.total_accepted_nll = 0.0
        self.total_accepted_tokens = 0

        print("[ModelWrapper] Model loaded successfully.\n")

    def _prepare_chat_input(self, prompt):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        return inputs

    def generate(self, prompt, params=None):
        if params is None:
            params = {}

        steps = params.get("steps", 256)
        max_new_tokens = params.get("max_new_tokens", 1024)
        small_block_size = params.get("small_block_size", 8)
        threshold = params.get("threshold", 0.95)

        # Prepare input
        inputs = self._prepare_chat_input(prompt)
        input_len = inputs["input_ids"].shape[1]

        # Generation
        start = time.time()
        with torch.inference_mode():
            gen_out = self.model.generate(
                inputs["input_ids"],
                tokenizer=self.tokenizer,
                max_new_tokens=max_new_tokens,
                small_block_size=small_block_size,
                threshold=threshold,
                steps=steps
            )
        end = time.time()

        # Update elapsed time
        self.total_elapsed_time += (end - start)

        # Update GPU peak
        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated()
            self.absolute_gpu_peak_memory = max(self.absolute_gpu_peak_memory, peak)

        # Unpack safely
        if isinstance(gen_out, tuple):
            gen_ids = gen_out[0]
        else:
            gen_ids = gen_out

        # Calculate perplexity on the generated tokens
        with torch.no_grad():
            outputs = self.model(gen_ids)
            logits = outputs.logits

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = gen_ids[..., 1:].contiguous()

            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss.view(shift_labels.size())

            # We only want to evaluate the generated tokens, not the prompt
            eval_mask = torch.ones_like(shift_labels, dtype=torch.bool)
            prompt_len = inputs["input_ids"].shape[1]
            eval_mask[:, :prompt_len-1] = False

            valid_loss = loss * eval_mask
            
            self.total_accepted_nll += valid_loss.sum().item()
            self.total_accepted_tokens += eval_mask.sum().item()


        # Remove prompt tokens
        if gen_ids.shape[1] > input_len:
            new_tokens = gen_ids[0][input_len:]
        else:
            new_tokens = gen_ids[0]

        # Decode
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        count = new_tokens.shape[0]

        return text, count



# ---------------------------------------------------------
# Public API for main.py
# ---------------------------------------------------------
def get_model_instance(model_path="Efficient-Large-Model/Fast_dLLM_7B"):
    global _model_instance
    if _model_instance is None:
        _model_instance = ModelWrapper(model_path)
    return _model_instance


def generate(prompt, params=None, model_path="Efficient-Large-Model/Fast_dLLM_7B"):
    if params is None:
        params = {}
    model = get_model_instance(model_path)
    return model.generate(prompt, params)
