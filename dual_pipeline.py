"""
Dual Diffusion Pipeline - Main orchestration for drafter-verifier generation.

This module provides a single entry point for running dual diffusion model inference
with flexible verification strategies.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Callable, Dict, Any
from inference import convert
from verification_algos import trust_verifier, confidence_threshold_verification


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
    remask_threshold: int = 0,
    
    # Additional kwargs
    **kwargs
) -> Dict[str, Any]:
    
    # Auto-detect mask token IDs
    if drafter_mask_id is None:
        drafter_mask_id = getattr(drafter_tokenizer, 'mask_token_id', 151665)
    if verifier_mask_id is None:
        verifier_mask_id = getattr(verifier_tokenizer, 'mask_token_id', 126336)
    
    if verification_fn is None:
        verification_fn = trust_verifier
    
    device = next(drafter_model.parameters()).device
    
    # Statistics tracking
    stats = {
        'iterations': 0,
        'total_drafter_steps': 0,
        'total_verifier_steps': 0,
        'tokens_remasked_per_iteration': [],
        'accepted_nll_sum': 0.0,
        'accepted_token_count': 0,
        'calls' : 0
    }
    
    # Perplexity accumulators
    total_accepted_nll = 0.0
    total_accepted_count = 0
    
    # 1. Create initial masked query
    initial_input = prepare_masked_query(query, max_new_tokens, drafter_tokenizer, drafter_mask_id, device)
    drafter_prompt_len = initial_input.size(1) - max_new_tokens
    
    # decoded_full = drafter_tokenizer.decode(initial_input[0], skip_special_tokens=False)
    # print("Full decoded base prompt (with special tokens):")
    # print(decoded_full)
    # print("###############################################")
    
    current_state = initial_input[:, :drafter_prompt_len]
    drafter_past_key_values = None
    
    # 2. Main iteration loop
    for iteration in range(max_iterations):
        stats['iterations'] = iteration + 1
        
        # 2a. Drafter phase (Truncation)
        if iteration > 0:
            mask_indices = (current_state == drafter_mask_id).nonzero(as_tuple=True)
            if len(mask_indices[0]) > 0:
                first_mask_idx = mask_indices[1].min().item()
                current_state = current_state[:, :first_mask_idx]
                new_seq_len = current_state.size(1)
                drafter_past_key_values = truncate_dynamic_cache(drafter_past_key_values, new_seq_len)
                print(f"Iteration {iteration}: Truncated to length {new_seq_len} due to masks.")
            else:
                print(f"Iteration {iteration}: No masks found, skipping truncation.")

        current_len = current_state.size(1)
        target_len = drafter_prompt_len + max_new_tokens
        tokens_to_generate = target_len - current_len
        
        if tokens_to_generate <= 0:
            print("Target length reached.")
            break

        # --- PERPLEXITY TRACKING HOOK ---
        current_draft_nll_data = []

        def capture_nll_hook(step, x, logits):
            stats['calls'] += 1
            if logits is None:
                return x
            
            with torch.no_grad():
                probs = F.softmax(logits, dim=-1)
                log_probs = torch.log(probs + 1e-9)
                
                # Gather log probs for the tokens currently in the sequence
                # Note: logits shape is typically [batch, seq_len, vocab]
                chosen_log_probs = log_probs.gather(-1, x.unsqueeze(-1)).squeeze(-1)
                is_masked = (x == drafter_mask_id)
                
                current_draft_nll_data.append({
                    'nll': -chosen_log_probs.detach(), # Store as positive NLL
                    'is_masked': is_masked.detach()
                })
            return x
        # --------------------------------

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
            generation_tokens_hook_func=capture_nll_hook  # <--- PASSING HOOK HERE
        )
        stats['total_drafter_steps'] += num_drafter_steps
        
        # [Debug] Check if hook captured data
        if not current_draft_nll_data:
            print(f"WARNING: Hook captured no data in iteration {iteration}. Ensure model calls generation_tokens_hook_func.")

        # Handle drafter result types
        if isinstance(drafter_result, tuple):
            current_state = drafter_result[0]
            drafter_past_key_values = drafter_result[1] if len(drafter_result) > 1 else None
            # If your model returns logits in the tuple, we can use them as fallback, 
            # but usually the hook handles it.
            drafter_logits = drafter_result[3] if len(drafter_result) > 3 else None
        else:
            current_state = drafter_result
            drafter_logits = None
            
        # If drafter_logits came back as a list (history), flatten it for verification steps
        if drafter_logits is not None:
            if isinstance(drafter_logits, list):
                # Concatenate the history of logits if available
                if len(drafter_logits) > 0:
                     drafter_logits = torch.cat(drafter_logits, dim=1)
                else:
                     drafter_logits = None
        
        # decoded_full = drafter_tokenizer.decode(current_state[0], skip_special_tokens=False)
        # print(f"Full decoded for drafter step {stats['total_drafter_steps']}:")
        # print(decoded_full)
        # print("###############################################")
        
        # 2b. Convert to verifier vocab
        drafter_generated = current_state[:, drafter_prompt_len:]
        verifier_generated_tokens = convert(
            drafter_mask_id, verifier_mask_id, drafter_generated, 
            drafter_tokenizer, verifier_tokenizer
        )
        
        verifier_prompt_input = prepare_masked_query(query, 0, verifier_tokenizer, verifier_mask_id, device)
        verifier_prompt_len = verifier_prompt_input.size(1)
        verifier_input = torch.cat([verifier_prompt_input, verifier_generated_tokens], dim=1)
        
        # decoded_full = verifier_tokenizer.decode(verifier_input[0], skip_special_tokens=False)
        # print(f"Full decoded for conversion:")
        # print(decoded_full)
        # print("###############################################")
        
        # 2c. Verifier phase
        verifier_result = run_verifier_phase(
            verifier_model, verifier_tokenizer, verifier_input, num_verifier_steps,
            verifier_mask_id, verifier_generate_fn, **kwargs
        )
        stats['total_verifier_steps'] += num_verifier_steps
        
        # Handle verifier outputs
        if isinstance(verifier_result, tuple):
            verifier_output = verifier_result[0]
            verifier_logits = verifier_result[1] if len(verifier_result) > 1 else None
        else:
            verifier_output = verifier_result
            verifier_logits = None
        
        # Align logits if both exist
        if verifier_logits is not None and drafter_logits is not None:
            slice_len = min(verifier_logits.size(1), drafter_logits.size(1))
            verifier_logits = verifier_logits[:, :slice_len, :]

        # decoded_full = verifier_tokenizer.decode(verifier_output[0], skip_special_tokens=False)
        # print(f"Full decoded for verifier step {stats['total_verifier_steps']}:")
        # print(decoded_full)
        
        # 2d. Convert back
        verifier_generated = verifier_output[:, verifier_prompt_len:]
        verifier_output_drafter_vocab_generated = convert(
            verifier_mask_id, drafter_mask_id, verifier_generated, 
            verifier_tokenizer, drafter_tokenizer
        )
        
        drafter_prompt_part = initial_input[:, :drafter_prompt_len]
        verifier_output_drafter_vocab = torch.cat([drafter_prompt_part, verifier_output_drafter_vocab_generated], dim=1)
        
        # 2e. Verification
        drafter_output = current_state
        min_len = min(drafter_output.size(1), verifier_output_drafter_vocab.size(1))
        
        if drafter_output.size(1) != verifier_output_drafter_vocab.size(1):
            print(f"Warning: Length mismatch. Drafter: {drafter_output.size(1)}, Verifier(conv): {verifier_output_drafter_vocab.size(1)}. Truncating to {min_len}.")
        
        drafter_output_ver = drafter_output[:, :min_len]
        verifier_output_ver = verifier_output_drafter_vocab[:, :min_len]

        verified_output, indices_to_remask = verification_fn(
            drafter_output_ver, verifier_output_ver, drafter_mask_id, verifier_mask_id,
            drafter_logits=drafter_logits, verifier_logits=verifier_logits,
            raw_verifier_output=verifier_output, **kwargs
        )
        
        num_remasked = len(indices_to_remask)
        stats['tokens_remasked_per_iteration'].append(num_remasked)
        
        # --- PERPLEXITY CALCULATION (FILTERING) ---
        # We assume current_draft_nll_data contains NLLs for each step of generation.
        # We need to filter out tokens that were eventually rejected.
        
        seq_len = current_state.size(1)
        rejected_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        
        # Mark rejected indices
        if indices_to_remask:
            valid_remask_indices = [i for i in indices_to_remask if i < seq_len]
            rejected_mask[valid_remask_indices] = True

        # Sum up valid NLLs from hook data
        for step_data in current_draft_nll_data:
            nll_tensor = step_data['nll'][0].to(device)       # [seq_len]
            is_masked_tensor = step_data['is_masked'][0].to(device) # [seq_len]
            
            # Handle potential length mismatches (e.g. prompt length changes or truncation)
            step_len = nll_tensor.size(0)
            current_rejected_mask = rejected_mask[:step_len]
            
            # Valid = Not Masked AND Not Rejected
            valid_tokens = (~is_masked_tensor[:step_len]) & (~current_rejected_mask)
            
            total_accepted_nll += nll_tensor[valid_tokens].sum().item()
            total_accepted_count += valid_tokens.sum().item()
        # ------------------------------------------

        # 2f. Loop Control
        if iteration == max_iterations - 1:
            final_output = verified_output
            break
        
        # 2g. Remask
        current_state = verified_output.clone()
        valid_indices = {idx for idx in indices_to_remask if idx < current_state.size(1)}
        for idx in valid_indices:
            current_state[0, idx] = drafter_mask_id
    
    else:
        final_output = verified_output
    
    # Store final aggregated stats
    stats['accepted_nll_sum'] = total_accepted_nll
    stats['accepted_token_count'] = total_accepted_count
    # print(f"DEBUG: Hook Calls: {stats['calls']}")
    
    output_text = drafter_tokenizer.decode(final_output[0], skip_special_tokens=True)
    
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
    """Prepare initial input with prompt + masked tokens."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": query},
    ]
    formatted_prompt = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )
    prompt_ids = tokenizer(formatted_prompt, return_tensors="pt")["input_ids"].to(device)
    mask_tensor = torch.full((1, max_new_tokens), mask_id, dtype=torch.long, device=device)
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
    """Run the drafter model generation."""
    if generate_fn is not None:
        return generate_fn(
            model, 
            tokenizer, 
            input_ids, 
            num_steps, 
            **kwargs)
    else:
        output = model.generate(
            input_ids,
            tokenizer=tokenizer,
            threshold=threshold,
            steps=num_steps,
            max_new_tokens=max_new_tokens,
            **kwargs
        )
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
    """Run the verifier model generation."""
    if generate_fn is not None:
        return generate_fn(model, tokenizer, input_ids, num_steps, **kwargs)
    else:
        output = model.generate(
            input_ids,
            tokenizer=tokenizer,
            steps=num_steps,
            **kwargs
        )
        if isinstance(output, tuple):
            return output[0]
        return output
    
def truncate_dynamic_cache(dynamic_cache, new_len):
    """Truncates a HuggingFace DynamicCache."""
    if not dynamic_cache:
        return None
    for i in range(len(dynamic_cache)):
        if dynamic_cache.layers[i].keys is not None:
            dynamic_cache.layers[i].keys = dynamic_cache.layers[i].keys[:, :, :new_len, :].contiguous()
        if dynamic_cache.layers[i].values is not None:
            dynamic_cache.layers[i].values = dynamic_cache.layers[i].values[:, :, :new_len, :].contiguous()
    return dynamic_cache