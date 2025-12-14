# =========================================================
#       Dual Diffusion Model Wrapper with Persistent Loading
# =========================================================

import sys
import os
import torch
import time
import math
from transformers import AutoTokenizer, AutoModel, AutoConfig, GenerationConfig

# Path setup matches your environment
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
fastdllm_path = os.path.join(repo_root, "FastDLLM_inferencing")
sys.path.insert(0, repo_root)
sys.path.insert(0, fastdllm_path)

from Fast_dLLM_v2_7B.modeling import Fast_dLLM_QwenForCausalLM
from LLaDA.generate import generate_per_step
from dual_pipeline import dual_diffusion_generate

# ---------------------------------------------------------
# Global singleton
# ---------------------------------------------------------
_model_instance = None


class DualDiffusionWrapper:
    """Manages both the drafter (Fast_dLLM) and verifier (LLaDA) models."""

    def __init__(self, drafter_path, verifier_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[DualDiffusionWrapper] Loading models on {self.device}...")

        # -------------------- Verifier --------------------
        print(f"[DualDiffusionWrapper] Loading verifier: {verifier_path}")
        self.verifier = AutoModel.from_pretrained(
            verifier_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.verifier_tokenizer = AutoTokenizer.from_pretrained(
            verifier_path,
            trust_remote_code=True
        )

        # -------------------- Drafter --------------------
        print(f"[DualDiffusionWrapper] Loading drafter: {drafter_path}")
        self.drafter_tokenizer = AutoTokenizer.from_pretrained(
            drafter_path,
            trust_remote_code=True
        )
        config = AutoConfig.from_pretrained(drafter_path, trust_remote_code=True)

        self.drafter = Fast_dLLM_QwenForCausalLM.from_pretrained(
            drafter_path,
            config=config,
            trust_remote_code=True,
            dtype="auto",
            device_map="auto"
        )

        gen_config = GenerationConfig.from_pretrained(drafter_path)
        self.drafter.generation_config = gen_config

        # Mask IDs
        self.drafter_mask_id = 151665
        self.verifier_mask_id = 126336

        # Tracking for evaluation suite
        self.total_elapsed_time = 0.0
        self.absolute_gpu_peak_memory = 0

        self.total_accepted_nll = 0.0
        self.total_accepted_tokens = 0 

        print("[DualDiffusionWrapper] All models loaded.\n")

    # -----------------------------------------------------
    # Required pipeline wrapper functions
    # -----------------------------------------------------
    def fastdllm_generate_fn(self, model, tokenizer, input_ids, num_steps, **kwargs):
        """Wrapper for Fast_dLLM generation."""
        gen_ids = model.generate(
            input_ids,
            tokenizer=tokenizer,
            max_new_tokens=kwargs.get("max_new_tokens", 256),
            small_block_size=kwargs.get("small_block_size", 8),
            threshold=kwargs.get("threshold", 0.95),
            steps=num_steps
        )
        return gen_ids

    def llada_generate_fn(self, model, tokenizer, input_ids, num_steps, **kwargs):
        """Wrapper for LLaDA generation."""
        mask_id = kwargs.get("mask_id", self.verifier_mask_id)

        mask_positions = (input_ids == mask_id)
        if mask_positions.any():
            first_mask_idx = torch.where(mask_positions[0])[0][0].item()
            prompt = input_ids[:, :first_mask_idx]
        else:
            prompt = input_ids

        return generate_per_step(
            model,
            prompt,
            n=num_steps,
            k=kwargs.get("k", 1),
            gen_length=kwargs.get("gen_length", 256),
            block_length=kwargs.get("block_length", 256),
            temperature=kwargs.get("temperature", 0.0),
            remasking=kwargs.get("remasking", "low_confidence"),
            mask_id=mask_id
        )

    # -----------------------------------------------------
    # Main generate method expected by evaluation driver
    # -----------------------------------------------------
    def generate(self, prompt, params):
        # Parameters
        max_new_tokens = params.get("max_new_tokens", 256)

        # Timing + GPU measurement
        start_time = time.time()

        result = dual_diffusion_generate(
            drafter_model=self.drafter,
            drafter_tokenizer=self.drafter_tokenizer,
            verifier_model=self.verifier,
            verifier_tokenizer=self.verifier_tokenizer,
            query=prompt,
            max_new_tokens=max_new_tokens,
            num_drafter_steps=params.get("num_drafter_steps", 16),
            num_verifier_steps=params.get("num_verifier_steps", 1),
            drafter_mask_id=self.drafter_mask_id,
            verifier_mask_id=self.verifier_mask_id,
            drafter_generate_fn=self.fastdllm_generate_fn,
            verifier_generate_fn=self.llada_generate_fn,
            verification_fn=None,
            max_iterations=params.get("max_iterations", 4),
            small_block_size=params.get("small_block_size", 8),
            threshold=params.get("threshold", 0.95),
            k=params.get("k", 1),
            gen_length=params.get("gen_length", 256),
            block_length=params.get("block_length", 256),
            temperature=params.get("temperature", 0.0),
            remasking=params.get("remasking", "low_confidence")
        )

        end_time = time.time()
        self.total_elapsed_time += (end_time - start_time)

        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated()
            self.absolute_gpu_peak_memory = max(self.absolute_gpu_peak_memory, peak)

        output_text = result["output_text"]

        # =========================================================
        #   Local PPL Calculation (Independent of pipeline)
        # =========================================================
        # Note: We use the Drafter (Causal LM) to calculate perplexity 
        # because the Verifier is a masked diffusion model.
        
        # 1. Re-tokenize the final output to get tensor IDs
        full_ids = self.drafter_tokenizer(output_text, return_tensors="pt").input_ids.to(self.device)
        
        # 2. Re-tokenize prompt to find where to start evaluating
        prompt_ids = self.drafter_tokenizer(prompt, return_tensors="pt").input_ids
        prompt_len = prompt_ids.shape[1]

        # 3. Standard Causal Forward Pass
        with torch.no_grad():
            outputs = self.drafter(full_ids)
            logits = outputs.logits

            # Shift logits and labels for next-token prediction
            # Logit[i] predicts Label[i+1]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = full_ids[..., 1:].contiguous()

            # Calculate Loss (NLL)
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss.view(shift_labels.size())

            # 4. Mask out the prompt (we only evaluate the new tokens)
            eval_mask = torch.ones_like(shift_labels, dtype=torch.bool)
            
            # The indices 0 to prompt_len-2 in shift_labels correspond to predicting the prompt.
            # Index prompt_len-1 corresponds to predicting the first NEW token.
            safe_start_idx = min(prompt_len - 1, eval_mask.shape[1])
            eval_mask[:, :safe_start_idx] = False

            valid_loss = loss * eval_mask

            # 5. Update Global Stats
            self.total_accepted_nll += valid_loss.sum().item()
            self.total_accepted_tokens += eval_mask.sum().item()

        # =========================================================

        # Use verifier tokenizer for final token count if that is the standard
        num_tokens = len(self.verifier_tokenizer(output_text)["input_ids"])

        return output_text, num_tokens


# ---------------------------------------------------------
# Public API (same as FastDLLM wrapper)
# ---------------------------------------------------------
def get_model_instance(
    model_path={
        "drafter": "Efficient-Large-Model/Fast_dLLM_7B",
        "verifier": "GSAI-ML/LLaDA-8B-Instruct"
    }
):
    global _model_instance
    if _model_instance is None:
        _model_instance = DualDiffusionWrapper(
            model_path["drafter"],
            model_path["verifier"]
        )
    return _model_instance


def generate(prompt, params=None, model_path=None):
    if params is None:
        params = {}

    model_instance = get_model_instance(
        model_path or {
            "drafter": "Efficient-Large-Model/Fast_dLLM_7B",
            "verifier": "GSAI-ML/LLaDA-8B-Instruct"
        }
    )

    return model_instance.generate(prompt, params)