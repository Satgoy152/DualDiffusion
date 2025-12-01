# =========================================================
#       Dual Diffusion Model Wrapper with Persistent Loading
# =========================================================

import sys
import os
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig, GenerationConfig

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
fastdllm_path = os.path.join(repo_root, "FastDLLM_inferencing")
sys.path.insert(0, repo_root)
sys.path.insert(0, fastdllm_path)

from Fast_dLLM_v2_7B.modeling import Fast_dLLM_QwenForCausalLM
from LLaDA.generate import generate_per_step
from dual_pipeline import dual_diffusion_generate

# ---------------------------------------------------------
# Global variable for singleton model instances
# ---------------------------------------------------------
_model_instance = None


class DualDiffusionWrapper:
    """Wrapper that manages both drafter (Fast_dLLM) and verifier (LLaDA) models."""
    
    def __init__(self, drafter_path, verifier_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"[DualDiffusionWrapper] Loading models on {self.device}...")
        
        # Load LLaDA verifier
        print(f"[DualDiffusionWrapper] Loading verifier '{verifier_path}'...")
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
        
        # Load Fast_dLLM drafter
        print(f"[DualDiffusionWrapper] Loading drafter '{drafter_path}'...")
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
        
        # Model-specific mask IDs
        self.drafter_mask_id = 151665  # Fast_dLLM mask ID
        self.verifier_mask_id = 126336  # LLaDA mask ID
        
        print("[DualDiffusionWrapper] Both models loaded successfully.\n")
    
    def fastdllm_generate_fn(self, model, tokenizer, input_ids, num_steps, **kwargs):
        """Wrapper for Fast_dLLM generation to match pipeline interface."""
        small_block_size = kwargs.get('small_block_size', 8)
        threshold = kwargs.get('threshold', 0.95)
        max_new_tokens = kwargs.get('max_new_tokens', 256)
        
        gen_ids, kv_cache, block_kv_cache = model.generate(
            input_ids,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            small_block_size=small_block_size,
            threshold=threshold,
            steps=num_steps,
        )
        
        return gen_ids, kv_cache, block_kv_cache
    
    def llada_generate_fn(self, model, tokenizer, input_ids, num_steps, **kwargs):
        """Wrapper for LLaDA generation to match pipeline interface."""
        k = kwargs.get('k', 1)
        gen_length = kwargs.get('gen_length', 256)
        block_length = kwargs.get('block_length', 256)
        temperature = kwargs.get('temperature', 0.0)
        remasking = kwargs.get('remasking', 'low_confidence')
        mask_id = kwargs.get('mask_id', self.verifier_mask_id)
        
        # Extract prompt from input_ids (remove masked tokens)
        mask_positions = (input_ids == mask_id)
        if mask_positions.any():
            first_mask_idx = torch.where(mask_positions[0])[0][0].item()
            prompt = input_ids[:, :first_mask_idx]
        else:
            prompt = input_ids
        
        output = generate_per_step(
            model,
            prompt,
            n=num_steps,
            k=k,
            gen_length=gen_length,
            block_length=block_length,
            temperature=temperature,
            remasking=remasking,
            mask_id=mask_id
        )
        
        return output
    
    def generate(self, prompt: str, params: dict):
        """
        Generate text using dual diffusion pipeline.
        
        Returns:
            Tuple of (generated_text, num_tokens)
        """
        # Extract parameters with defaults
        max_new_tokens = params.get('max_new_tokens', 256)
        num_drafter_steps = params.get('num_drafter_steps', 16)
        num_verifier_steps = params.get('num_verifier_steps', 1)
        max_iterations = params.get('max_iterations', 4)
        
        # Call dual diffusion generate
        result = dual_diffusion_generate(
            # Models
            drafter_model=self.drafter,
            drafter_tokenizer=self.drafter_tokenizer,
            verifier_model=self.verifier,
            verifier_tokenizer=self.verifier_tokenizer,
            
            # Input
            query=prompt,
            max_new_tokens=max_new_tokens,
            
            # Steps
            num_drafter_steps=num_drafter_steps,
            num_verifier_steps=num_verifier_steps,
            
            # Mask IDs
            drafter_mask_id=self.drafter_mask_id,
            verifier_mask_id=self.verifier_mask_id,
            
            # Custom generate functions
            drafter_generate_fn=self.fastdllm_generate_fn,
            verifier_generate_fn=self.llada_generate_fn,
            
            # Verification
            verification_fn=None,
            
            # Iteration control
            max_iterations=max_iterations,
            
            # Model-specific kwargs
            small_block_size=params.get('small_block_size', 8),
            threshold=params.get('threshold', 0.95),
            k=params.get('k', 1),
            gen_length=params.get('gen_length', 256),
            block_length=params.get('block_length', 256),
            temperature=params.get('temperature', 0.0),
            remasking=params.get('remasking', 'low_confidence')
        )
        
        output_text = result['output_text']
        # Calculate number of tokens generated using verifier tokenizer
        num_tokens = len(self.verifier_tokenizer(output_text)['input_ids'])
        
        return output_text, num_tokens


# ---------------------------------------------------------
# Public generate() function
# ---------------------------------------------------------
def generate(
    prompt: str, 
    params: dict = {}, 
    drafter_path: str = "Efficient-Large-Model/Fast_dLLM_7B",
    verifier_path: str = "GSAI-ML/LLaDA-8B-Instruct"
):
    """
    Generate text using dual diffusion with persistent model loading.
    
    Returns:
        Tuple of (generated_text, num_tokens)
    """
    global _model_instance
    if _model_instance is None:
        _model_instance = DualDiffusionWrapper(drafter_path, verifier_path)
    return _model_instance.generate(prompt, params)


# ---------------------------------------------------------
# Optional test run
# ---------------------------------------------------------
if __name__ == "__main__":
    test_prompt = "Give me a short introduction to large language models."
    test_params = {
        "max_new_tokens": 256,
        "num_drafter_steps": 16,
        "num_verifier_steps": 1,
        "max_iterations": 4
    }
    
    text, tokens = generate(test_prompt, test_params)
    print("Generated text:")
    print(text)
    print("Number of tokens generated:", tokens)
