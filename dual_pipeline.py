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
    
    # Calculate prompt length for drafter (initial_input is Prompt + Masks)
    drafter_prompt_len = initial_input.size(1) - max_new_tokens
    
    decoded_full = drafter_tokenizer.decode(initial_input[0], skip_special_tokens=False)
    print("Full decoded base prompt (with special tokens):")
    print(decoded_full)
    print("###############################################")
    
    # Start with just the prompt for the first generation
    current_state = initial_input[:, :drafter_prompt_len]
    drafter_past_key_values = None
    
    # 2. Main iteration loop
    for iteration in range(max_iterations):
        stats['iterations'] = iteration + 1
        
        # 2a. Drafter phase
        # If this is a re-run (iteration > 0), we need to handle truncation based on masks
        if iteration > 0:
            # Find the first mask token in the current state (which comes from verification)
            mask_indices = (current_state == drafter_mask_id).nonzero(as_tuple=True)
            
            if len(mask_indices[0]) > 0:
                # Get the index of the first mask in the sequence
                # Assuming batch size 1
                first_mask_idx = mask_indices[1].min().item()
                
                # Truncate input_ids to the point before the first mask
                current_state = current_state[:, :first_mask_idx]
                
                # Truncate KV cache to match the new sequence length
                new_seq_len = current_state.size(1)
                drafter_past_key_values = truncate_dynamic_cache(drafter_past_key_values, new_seq_len)
                
                print(f"Iteration {iteration}: Truncated to length {new_seq_len} due to masks.")
            else:
                print(f"Iteration {iteration}: No masks found, skipping truncation.")

        # Calculate how many tokens we still need to generate
        current_len = current_state.size(1)
        target_len = drafter_prompt_len + max_new_tokens
        tokens_to_generate = target_len - current_len
        
        if tokens_to_generate <= 0:
            print("Target length reached.")
            break

        drafter_result = run_drafter_phase(
            drafter_model,
            drafter_tokenizer,
            current_state,
            num_drafter_steps,
            tokens_to_generate,
            drafter_mask_id,
            drafter_generate_fn,
            threshold=0.95,
            past_key_values=drafter_past_key_values,
        )
        stats['total_drafter_steps'] += num_drafter_steps
        
        # Update state and cache
        current_state = drafter_result[0]
        drafter_past_key_values = drafter_result[1] if len(drafter_result) > 1 else None
        drafter_logits = drafter_result[3] if len(drafter_result) > 3 else None

        decoded_full = drafter_tokenizer.decode(current_state[0], skip_special_tokens=False)
        print(f"Full decoded for drafter step {stats['total_drafter_steps']} (with special tokens):")
        print(decoded_full)
        print("###############################################")
        
        
        # 2b. Convert to verifier vocabulary
        # Extract generated part (skip prompt)
        drafter_generated = current_state[:, drafter_prompt_len:]
        
        verifier_generated_tokens = convert(
            drafter_mask_id,
            verifier_mask_id,
            drafter_generated,
            drafter_tokenizer,
            verifier_tokenizer
        )

        # Reconstruct Verifier Input (Prompt + Converted Generated)
        # Get verifier prompt
        verifier_prompt_input = prepare_masked_query(
            query, 
            0, # No extra masks, just prompt
            verifier_tokenizer, 
            verifier_mask_id, 
            device
        )
        verifier_prompt_len = verifier_prompt_input.size(1)
        
        verifier_input = torch.cat([verifier_prompt_input, verifier_generated_tokens], dim=1)

        decoded_full = verifier_tokenizer.decode(verifier_input[0], skip_special_tokens=False)
        print(f"Full decoded for conversion (with special tokens):")
        print(decoded_full)
        print("###############################################")
        
        # 2c. Verifier phase
        verifier_result = run_verifier_phase(
            verifier_model,
            verifier_tokenizer,
            verifier_input,
            num_verifier_steps,
            verifier_mask_id,
            verifier_generate_fn,
            **kwargs
        )
        stats['total_verifier_steps'] += num_verifier_steps

        verifier_output = verifier_result[0]
        verifier_logits = verifier_result[1] if len(verifier_result) > 1 else None

        decoded_full = verifier_tokenizer.decode(verifier_output[0], skip_special_tokens=False)
        print(f"Full decoded for verfier step {stats['total_verifier_steps']} (with special tokens):")
        print(decoded_full)
        
        # 2d. Convert verifier output back to drafter vocabulary
        # Extract generated part
        verifier_generated = verifier_output[:, verifier_prompt_len:]
        
        verifier_output_drafter_vocab_generated = convert(
            verifier_mask_id,
            drafter_mask_id,
            verifier_generated,
            verifier_tokenizer,
            drafter_tokenizer
        )
        
        # Reconstruct full sequence for verification (Drafter Prompt + Converted Back Generated)
        # Use the original drafter prompt part
        drafter_prompt_part = initial_input[:, :drafter_prompt_len]
        verifier_output_drafter_vocab = torch.cat([drafter_prompt_part, verifier_output_drafter_vocab_generated], dim=1)
        
        # 2e. Verification step
        # Handle length mismatch by truncating to minimum length
        drafter_output = current_state
        min_len = min(drafter_output.size(1), verifier_output_drafter_vocab.size(1))
        if drafter_output.size(1) != verifier_output_drafter_vocab.size(1):
            print(f"Warning: Length mismatch. Drafter: {drafter_output.size(1)}, Verifier(conv): {verifier_output_drafter_vocab.size(1)}. Truncating to {min_len}.")
        
        drafter_output_ver = drafter_output[:, :min_len]
        verifier_output_ver = verifier_output_drafter_vocab[:, :min_len]
        
        verified_output, indices_to_remask = verification_fn(
            drafter_output_ver,
            verifier_output_ver,
            drafter_mask_id,
            verifier_mask_id,
            drafter_logits=drafter_logits,
            verifier_logits=verifier_logits,
            raw_verifier_output=verifier_output,
            **kwargs
        )
        
        num_remasked = len(indices_to_remask)
        stats['tokens_remasked_per_iteration'].append(num_remasked)
        
        # 2f. Check if we should continue iterating
        if iteration == max_iterations - 1:
            # Done iterating - return final output
            final_output = verified_output
            break
        
        # 2g. Remask for next iteration
        current_state = verified_output.clone()
        for idx in indices_to_remask:
            current_state[0, idx] = drafter_mask_id
            
        # Note: current_state might be shorter than initial_input if truncated.
        # If we need to maintain length, we might need to pad?
        # For now, we proceed with the truncated/verified state.
    
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
    prompt_ids = tokenizer(formatted_prompt, return_tensors="pt")["input_ids"].to(device)
    
    # Create masked extension
    mask_tensor = torch.full((1, max_new_tokens), mask_id, dtype=torch.long, device=device)
    
    # Concatenate
    full_input = torch.cat([prompt_ids, mask_tensor], dim=1)
    
    return full_input


def run_drafter_phase(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    num_steps: int,
    max_new_tokens,
    mask_id: int,
    generate_fn: Optional[Callable] = None,
    threshold=0.95,
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
    
def truncate_dynamic_cache(dynamic_cache, new_len):
    """
    Truncates a HuggingFace DynamicCache to match a shortened sequence length.
    """
    for i in range(len(dynamic_cache)):
        if dynamic_cache.layers[i].keys is not None:
            dynamic_cache.layers[i].keys = dynamic_cache.layers[i].keys[:, :, :new_len, :].contiguous()
        if dynamic_cache.layers[i].values is not None:
            dynamic_cache.layers[i].values = dynamic_cache.layers[i].values[:, :, :new_len, :].contiguous()
    return dynamic_cache

        