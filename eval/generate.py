# eval/generate.py

import sys
import os
import torch
from transformers import AutoTokenizer, AutoConfig, GenerationConfig

# ---------------------------------------------------------
# Fix import paths for FastDLLM
# ---------------------------------------------------------
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
fastdllm_path = os.path.join(repo_root, "FastDLLM_inferencing")
sys.path.insert(0, repo_root)
sys.path.insert(0, fastdllm_path)

# Now Python can find Fast_dLLM_v2_7B
from Fast_dLLM_v2_7B.modeling import Fast_dLLM_QwenForCausalLM


# =========================================================
#                   Model Wrapper
# =========================================================
class ModelWrapper:
    """Loads and manages the Fast-dLLM model."""

    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[ModelWrapper] Loading model '{model_path}' on {self.device}...")

        # -------------------------
        # Load tokenizer + config
        # -------------------------
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )

        config = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        # -------------------------
        # Load model
        # -------------------------
        self.model = Fast_dLLM_QwenForCausalLM.from_pretrained(
            model_path,
            config=config,
            trust_remote_code=True,
            dtype="auto",
            device_map="auto"
        )

        # -------------------------
        # Load generation config
        # -------------------------
        gen_config = GenerationConfig.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        self.model.generation_config = gen_config

        print("[ModelWrapper] Model loaded successfully.\n")


    # -----------------------------------------------------
    # Private helper: format the prompt using chat template
    # -----------------------------------------------------
    def _prepare_chat_input(self, prompt: str):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        return inputs


    # -----------------------------------------------------
    # Main generate() function — DO NOT rename
    # -----------------------------------------------------
    def generate(self, prompt: str, params: dict) -> str:
        """
        The evaluation suite will call this.

        Args:
            prompt: user question
            params: dict from config (e.g., {"steps": 50})
        """

        # Convert eval params → model parameters
        steps = params.get("steps", 256)
        max_new_tokens = params.get("max_new_tokens", 1024)

        print(f"[ModelWrapper] Generating: steps={steps}, max_new_tokens={max_new_tokens}")

        # Prepare chat input
        inputs = self._prepare_chat_input(prompt)

        # Generate
        with torch.inference_mode():
            gen_ids, past_key_values, past_block_key_values = self.model.generate(
                inputs["input_ids"],
                tokenizer=self.tokenizer,
                max_new_tokens=max_new_tokens,
                small_block_size=8,
                threshold=0.95,
                steps=steps
            )

        # Decode only the new tokens
        response = self.tokenizer.decode(
            gen_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

        return response


# =========================================================
#       Global instance + public generate() function
# =========================================================

# Use the model name from your notebook
model_path_from_config = "Efficient-Large-Model/Fast_dLLM_7B"
_model_wrapper = ModelWrapper(model_path_from_config)


def generate(prompt: str, params: dict = {}) -> str:
    """Public entry point. DO NOT modify signature."""
    return _model_wrapper.generate(prompt, params)


# ---------------------------------------------------------
# Test run if executed directly
# ---------------------------------------------------------
if __name__ == "__main__":
    test_prompt = "Explain gravity in simple terms."
    test_params = {"steps": 128}
    print("--- Test Run ---")
    print(f"Prompt: {test_prompt}")
    print(f"Output: {generate(test_prompt, test_params)}")
