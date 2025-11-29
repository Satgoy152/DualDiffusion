"""
Dual Diffusion Pipeline - Main orchestration for drafter-verifier generation.

This module provides a single entry point for running dual diffusion model inference
with flexible verification strategies.
"""

import torch
from typing import Optional, Callable, Dict, Any
from inference import convert
from verification_algos import trust_verifier


def dual_diffusion_generate(
    # Models & tokenizers
    drafter_model,
    drafter_tokenizer,
    verifier_model,
    verifier_tokenizer,
    
    # Input
    query: str,
    max_new_tokens: int,
    
    # Generation parameters
    num_drafter_steps: int,
    num_verifier_steps: int = 1,
    
    # Mask token IDs
    drafter_mask_id: Optional[int] = None,
    verifier_mask_id: Optional[int] = None,
    
    # Optional custom functions
    drafter_generate_fn: Optional[Callable] = None,
    verifier_generate_fn: Optional[Callable] = None,
    verification_fn: Optional[Callable] = None,
    
    # Iteration control
    max_iterations: int = 1,
    remask_threshold: int = 0,  # Min tokens to remask before continuing iteration
    
    # Additional kwargs for generate functions
    **kwargs
) -> Dict[str, Any]:
    """
    Single function orchestrating the dual diffusion pipeline.
    
    Flow:
    1. Create initial masked query
    2. Loop (up to max_iterations):
        a. Run drafter for n steps
        b. Convert drafter output to verifier vocabulary
        c. Run verifier for m steps
        d. Convert verifier output back to drafter vocabulary
        e. Run verification algorithm
        f. If tokens need remasking and iterations remain, remask and continue
        g. Otherwise, return final output
    
    Args:
        drafter_model: Fast drafter model
        drafter_tokenizer: Tokenizer for drafter
        verifier_model: Slower, more accurate verifier model
        verifier_tokenizer: Tokenizer for verifier
        query: Input prompt string
        max_new_tokens: Number of new tokens to generate
        num_drafter_steps: Number of diffusion steps for drafter
        num_verifier_steps: Number of diffusion steps for verifier
        drafter_mask_id: Mask token ID for drafter (auto-detected if None)
        verifier_mask_id: Mask token ID for verifier (auto-detected if None)
        drafter_generate_fn: Custom function for drafter generation
        verifier_generate_fn: Custom function for verifier generation
        verification_fn: Custom verification algorithm
        max_iterations: Maximum draft-verify-remask iterations
        remask_threshold: Minimum tokens to remask to continue iterating
        **kwargs: Additional arguments passed to generate functions
    
    Returns:
        Dictionary with:
            - 'output_ids': Final output tensor
            - 'output_text': Decoded output text
            - 'stats': Debug statistics
    """
    
    # Auto-detect mask token IDs if not provided
    if drafter_mask_id is None:
        drafter_mask_id = getattr(drafter_tokenizer, 'mask_token_id', 151665)
    if verifier_mask_id is None:
        verifier_mask_id = getattr(verifier_tokenizer, 'mask_token_id', 126336)
    
    # Use default verification if none provided
    if verification_fn is None:
        verification_fn = trust_verifier
    
    # Get device from drafter model
    device = next(drafter_model.parameters()).device
    
    # Statistics tracking
    stats = {
        'iterations': 0,
        'total_drafter_steps': 0,
        'total_verifier_steps': 0,
        'tokens_remasked_per_iteration': [],
    }
    
    # 1. Create initial masked query with drafter tokenizer
    initial_input = prepare_masked_query(
        query, 
        max_new_tokens, 
        drafter_tokenizer, 
        drafter_mask_id,
        device
    )
    decoded_full = drafter_tokenizer.decode(initial_input[0], skip_special_tokens=False)
    print("Full decoded base prompt (with special tokens):")
    print(decoded_full)
    print("###############################################")
    current_state = initial_input
    input_len = initial_input[0].size(0)
    
    # 2. Main iteration loop
    for iteration in range(max_iterations):
        stats['iterations'] = iteration + 1

        new_tokens = max_new_tokens - (current_state[0].size(0) - input_len)
        
        # 2a. Drafter phase
        drafter_output, drafter_kv_cache, drafter_block_cache = run_drafter_phase(
            drafter_model,
            drafter_tokenizer,
            current_state,
            num_drafter_steps,
            new_tokens,
            drafter_mask_id,
            drafter_generate_fn,
            threshold=0.95,
            **kwargs
        )
        stats['total_drafter_steps'] += num_drafter_steps

        decoded_full = drafter_tokenizer.decode(drafter_output[0], skip_special_tokens=False)
        print(f"Full decoded for drafter step {stats['total_drafter_steps']} (with special tokens):")
        print(decoded_full)
        print("###############################################")
        
        # 2b. Convert to verifier vocabulary
        verifier_input = convert(
            drafter_mask_id,
            verifier_mask_id,
            drafter_output[:, initial_input.shape[1]:],
            drafter_tokenizer,
            verifier_tokenizer
        )

        decoded_full = verifier_tokenizer.decode(verifier_input[0], skip_special_tokens=False)
        print(f"Full decoded for conversion (with special tokens):")
        print(decoded_full)
        print("###############################################")
        
        # 2c. Verifier phase
        verifier_output = run_verifier_phase(
            verifier_model,
            verifier_tokenizer,
            verifier_input,
            num_verifier_steps,
            verifier_mask_id,
            verifier_generate_fn,
            **kwargs
        )
        stats['total_verifier_steps'] += num_verifier_steps

        decoded_full = verifier_tokenizer.decode(verifier_output[0], skip_special_tokens=False)
        print(f"Full decoded for verfier step {stats['total_verifier_steps']} (with special tokens):")
        print(decoded_full)
        break
        
        # 2d. Convert verifier output back to drafter vocabulary
        verifier_output_drafter_vocab = convert(
            verifier_mask_id,
            drafter_mask_id,
            verifier_output,
            verifier_tokenizer,
            drafter_tokenizer
        )
        
        # 2e. Verification step
        verified_output, indices_to_remask = verification_fn(
            drafter_output,
            verifier_output_drafter_vocab,
            drafter_mask_id,
            verifier_mask_id,
            **kwargs
        )
        
        num_remasked = len(indices_to_remask)
        stats['tokens_remasked_per_iteration'].append(num_remasked)
        
        # 2f. Check if we should continue iterating
        if num_remasked <= remask_threshold or iteration == max_iterations - 1:
            # Done iterating - return final output
            final_output = verified_output
            break
        
        # 2g. Remask for next iteration
        current_state = verified_output.clone()
        for idx in indices_to_remask:
            current_state[0, idx] = drafter_mask_id
    
    else:
        # If loop completed without break, use last verified output
        final_output = verified_output
    
    # 3. Decode and return
    output_text = drafter_tokenizer.decode(
        final_output[0], 
        skip_special_tokens=True
    )
    
    return {
        'output_ids': final_output,
        'output_text': output_text,
        'stats': stats
    }


def prepare_masked_query(
    query: str,
    max_new_tokens: int,
    tokenizer,
    mask_id: int,
    device: torch.device
) -> torch.Tensor:
    """
    Prepare initial input with prompt + masked tokens.
    
    Args:
        query: Input prompt
        max_new_tokens: Number of tokens to generate (will be masked)
        tokenizer: Tokenizer to use
        mask_id: Mask token ID
        device: Device to place tensor on
    
    Returns:
        Tensor of shape (1, prompt_len + max_new_tokens) with masked generation area
    """
    # apply chat tempelate
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": query},
    ]
    formatted_prompt = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )
    # Tokenize the prompt
    prompt_ids = tokenizer([text], return_tensors="pt").to(model.device)
    
    # Create full sequence: prompt + masked tokens
    # Drafter shouldn't take in a full set of masks - adds them on in the generate function
    # full_length = len(prompt_ids) + max_new_tokens
    # masked_sequence = torch.full((1, full_length), mask_id, dtype=torch.long, device=device)
    
    # Copy prompt to beginning
    # masked_sequence[0, :len(prompt_ids)] = prompt_tensor
    
    return prompt_ids["input_ids"]


def run_drafter_phase(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    num_steps: int,
    max_new_tokens,
    mask_id: int,
    generate_fn: Optional[Callable] = None,
    threshold=0.95
    **kwargs
) -> torch.Tensor:
    """
    Run the drafter model generation.
    
    Args:
        model: Drafter model
        tokenizer: Drafter tokenizer
        input_ids: Input tensor with masked tokens
        num_steps: Number of diffusion steps
        mask_id: Mask token ID
        generate_fn: Optional custom generation function
        **kwargs: Additional arguments for generation
    
    Returns:
        Generated tensor from drafter
    """
    if generate_fn is not None:
        # Use custom generation function
        return generate_fn(
            model, 
            tokenizer, 
            input_ids, 
            num_steps, 
            **kwargs)
    else:
        # Default: assume model has .generate() method
        # Adapted to fit FastDLLM custom generation function
        output = model.generate(
            input_ids,
            tokenizer=tokenizer,
            threshold=threshold,
            steps=num_steps,
            max_new_tokens=max_new_tokens,
            **kwargs
        )
        # Handle different return types (some models return tuples)
        # For Fast DLLM - this is a tuple of 3 (output_ids, kv_cache, block_kv_cache)
        return output


def run_verifier_phase(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    num_steps: int,
    mask_id: int,
    generate_fn: Optional[Callable] = None,
    **kwargs
) -> torch.Tensor:
    """
    Run the verifier model generation.
    
    Args:
        model: Verifier model
        tokenizer: Verifier tokenizer
        input_ids: Input tensor with masked tokens
        num_steps: Number of diffusion steps
        mask_id: Mask token ID
        generate_fn: Optional custom generation function
        **kwargs: Additional arguments for generation
    
    Returns:
        Generated tensor from verifier
    """
    if generate_fn is not None:
        # Use custom generation function
        return generate_fn(model, tokenizer, input_ids, num_steps, **kwargs)
    else:
        # Default: assume model has .generate() method
        # This is a placeholder - you'll need to adapt to your model's API
        output = model.generate(
            input_ids,
            tokenizer=tokenizer,
            steps=num_steps,
            **kwargs
        )
        # Handle different return types
        if isinstance(output, tuple):
            return output[0]
        return output

        