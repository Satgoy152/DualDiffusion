# =========================================================
#   LLaDA Persistent Model Wrapper
# =========================================================
import sys
import os
import torch
import time
from transformers import AutoTokenizer, AutoModel

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
llada_path = os.path.join(repo_root, "LLaDA")
sys.path.insert(0, repo_root)
sys.path.insert(0, llada_path)

from LLaDA.generate import generate_per_step

# Global singleton
_model_instance = None

class ModelWrapper:
    def __init__(self, model_path="GSAI-ML/LLaDA-8B-Instruct"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[LLaDAWrapper] Loading model '{model_path}' on {self.device}...")

        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        ).to(self.device).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        self.total_elapsed_time = 0.0
        self.absolute_gpu_peak_memory = 0

        print("[LLaDAWrapper] Model loaded successfully.\n")

    def generate(self, prompt, params=None):
        if params is None:
            params = {}

        # LLaDA-specific generation parameters
        gen_length = params.get("gen_length", 64)
        n_steps = params.get("n_steps", gen_length)
        k = params.get("k", 1)
        block_length = params.get("block_length", gen_length)
        temperature = params.get("temperature", 0.0)
        cfg_scale = params.get("cfg_scale", 0.0)
        remasking = params.get("remasking", "low_confidence")
        eos_token_id = params.get("eos_token_id", 126081)  # EOS token for LLaDA

        # Format prompt
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
        else:
            formatted_prompt = prompt

        # Tokenize
        input_ids = self.tokenizer(formatted_prompt)["input_ids"]
        input_ids = torch.tensor(input_ids).to(self.device).unsqueeze(0)
        input_len = input_ids.shape[1]

        # Generate
        start_time = time.time()
        with torch.inference_mode():
            out = generate_per_step(
                self.model,
                input_ids,
                n=n_steps,
                k=k,
                gen_length=gen_length,
                block_length=block_length,
                temperature=temperature,
                cfg_scale=cfg_scale,
                remasking=remasking
            )
        end_time = time.time()

        # Track GPU/time
        self.total_elapsed_time += end_time - start_time
        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated()
            self.absolute_gpu_peak_memory = max(self.absolute_gpu_peak_memory, peak)

        # Handle tuple output from generate_per_step
        if isinstance(out, tuple):
            out_ids = out[0]
        else:
            out_ids = out

        # Slice to remove prompt tokens
        generated_token_ids = out_ids[:, input_len:]

        # Find first EOS if present
        eos_positions = (generated_token_ids[0] == eos_token_id).nonzero(as_tuple=True)[0]
        if len(eos_positions) > 0:
            num_tokens = eos_positions[0].item() + 1
            generated_token_ids = generated_token_ids[:, :num_tokens]
        else:
            num_tokens = generated_token_ids.shape[1]

        # Decode
        generated_text = self.tokenizer.batch_decode(
            generated_token_ids, skip_special_tokens=True
        )[0]

        return generated_text, num_tokens


# ---------------------------------------------------------
# Public API for main.py
# ---------------------------------------------------------
def get_model_instance(model_path="GSAI-ML/LLaDA-8B-Instruct"):
    global _model_instance
    if _model_instance is None:
        _model_instance = ModelWrapper(model_path)
    return _model_instance


def generate(prompt, params=None, model_path="GSAI-ML/LLaDA-8B-Instruct"):
    if params is None:
        params = {}
    model = get_model_instance(model_path)
    return model.generate(prompt, params)
