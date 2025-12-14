"""
Dual Diffusion Pipeline - Main orchestration for drafter-verifier generation.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Callable, Dict, Any
from inference import convert
from verification_algos import trust_verifier, confidence_threshold_verification

def dual_diffusion_generate(
    drafter_model, drafter_tokenizer, verifier_model, verifier_tokenizer,
    query: str, max_new_tokens: int, num_drafter_steps: int, num_verifier_steps: int = 1,
    drafter_mask_id: Optional[int] = None, verifier_mask_id: Optional[int] = None,
    drafter_generate_fn: Optional[Callable] = None, verifier_generate_fn: Optional[Callable] = None,
    verification_fn: Optional[Callable] = None,
    max_iterations: int = 1, remask_threshold: int = 0,
    **kwargs
) -> Dict[str, Any]:
    
    # Defaults
    if drafter_mask_id is None: drafter_mask_id = getattr(drafter_tokenizer, 'mask_token_id', 151665)
    if verifier_mask_id is None: verifier_mask_id = getattr(verifier_tokenizer, 'mask_token_id', 126336)
    if verification_fn is None: verification_fn = trust_verifier
    device = next(drafter_model.parameters()).device
    
    # Stats
    stats = {
        'iterations': 0, 'total_drafter_steps': 0, 'total_verifier_steps': 0,
        'tokens_remasked_per_iteration': [], 'accepted_nll_sum': 0.0, 'accepted_token_count': 0
    }
    total_accepted_nll = 0.0
    total_accepted_count = 0
    
    # 1. Initialize
    initial_input = prepare_masked_query(query, max_new_tokens, drafter_tokenizer, drafter_mask_id, device)
    # prompt_len is the length of the user query + system prompt. We don't score these.
    drafter_prompt_len = initial_input.size(1) - max_new_tokens
    
    current_state = initial_input[:, :drafter_prompt_len]
    drafter_past_key_values = None
    
    # 2. Main Loop
    for iteration in range(max_iterations):
        stats['iterations'] = iteration + 1
        
        # Truncation logic
        if iteration > 0:
            mask_indices = (current_state == drafter_mask_id).nonzero(as_tuple=True)
            if len(mask_indices[0]) > 0:
                first_mask_idx = mask_indices[1].min().item()
                current_state = current_state[:, :first_mask_idx]
                drafter_past_key_values = truncate_dynamic_cache(drafter_past_key_values, current_state.size(1))

        tokens_to_generate = (drafter_prompt_len + max_new_tokens) - current_state.size(1)
        if tokens_to_generate <= 0: break

        # Run Drafter
        drafter_result = run_drafter_phase(
            drafter_model, drafter_tokenizer, current_state, num_drafter_steps,
            tokens_to_generate, drafter_mask_id, drafter_generate_fn,
            threshold=0.95, past_key_values=drafter_past_key_values
        )
        stats['total_drafter_steps'] += num_drafter_steps
        
        # Unpack Result
        if isinstance(drafter_result, tuple):
            current_state = drafter_result[0]
            drafter_past_key_values = drafter_result[1] if len(drafter_result) > 1 else None
            # We ignore returned logits because we will re-compute them cleanly later
            drafter_logits = None 
        else:
            current_state = drafter_result
            drafter_logits = None

        # Convert & Verify
        drafter_generated = current_state[:, drafter_prompt_len:]
        verifier_generated_tokens = convert(drafter_mask_id, verifier_mask_id, drafter_generated, drafter_tokenizer, verifier_tokenizer)
        verifier_prompt_input = prepare_masked_query(query, 0, verifier_tokenizer, verifier_mask_id, device)
        verifier_input = torch.cat([verifier_prompt_input, verifier_generated_tokens], dim=1)
        
        verifier_result = run_verifier_phase(
            verifier_model, verifier_tokenizer, verifier_input, num_verifier_steps,
            verifier_mask_id, verifier_generate_fn, **kwargs
        )
        stats['total_verifier_steps'] += num_verifier_steps
        
        verifier_output = verifier_result[0] if isinstance(verifier_result, tuple) else verifier_result
        verifier_logits = verifier_result[1] if isinstance(verifier_result, tuple) and len(verifier_result) > 1 else None

        # Reconstruct & Verify
        verifier_generated = verifier_output[:, verifier_prompt_input.size(1):]
        verifier_output_drafter_vocab = convert(verifier_mask_id, drafter_mask_id, verifier_generated, verifier_tokenizer, drafter_tokenizer)
        
        min_len = min(current_state.size(1), verifier_output_drafter_vocab.size(1) + drafter_prompt_len)
        drafter_output_ver = current_state[:, :min_len]
        verifier_output_ver = torch.cat([initial_input[:, :drafter_prompt_len], verifier_output_drafter_vocab], dim=1)[:, :min_len]

        verified_output, indices_to_remask = verification_fn(
            drafter_output_ver, verifier_output_ver, drafter_mask_id, verifier_mask_id,
            drafter_logits=drafter_logits, verifier_logits=verifier_logits,
            raw_verifier_output=verifier_output, **kwargs
        )
        
        stats['tokens_remasked_per_iteration'].append(len(indices_to_remask))

        # ==============================================================================
        # ROBUST PERPLEXITY CALCULATION (CAUSAL FORWARD PASS)
        # ==============================================================================
        # We calculate the perplexity of the accepted tokens using the Drafter itself.
        # This is a standard "Forward Pass" method, robust to hooks or decoding strategies.
        
        # 1. Run forward pass on the accepted sequence
        with torch.no_grad():
            # verified_output contains [Prompt + Generated_Tokens]
            outputs = drafter_model(verified_output)
            logits = outputs.logits
            
            # 2. Shift logits and labels for Causal LM loss (next token prediction)
            # Logits at [:-1] predict labels at [1:]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = verified_output[..., 1:].contiguous()
            
            # 3. Calculate element-wise Cross Entropy (NLL)
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            # flatten to [batch*seq_len, vocab] and [batch*seq_len]
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss.view(shift_labels.size()) # Reshape back to [batch, seq_len]

            # 4. Create Evaluation Mask
            # We want to count a token IF:
            #   a. It is NOT part of the prompt (we only evaluate generation)
            #   b. It is NOT a token we just rejected/masked
            
            eval_mask = torch.ones_like(shift_labels, dtype=torch.bool)
            
            # a. Mask out prompt
            # Note: shift_labels indices are shifted by 1 relative to full sequence
            # So index 0 in shift_labels corresponds to index 1 in full sequence
            # If prompt length is P, we want to ignore indices 0 to P-2 in shift_labels?
            # Easiest way: absolute indices.
            
            seq_len = verified_output.size(1)
            # Indices [0 ... seq_len-1]
            # prompt ends at drafter_prompt_len. 
            # We want to evaluate tokens starting from drafter_prompt_len.
            # In shift_labels, index i corresponds to verified_output[i+1].
            # So we want i+1 >= drafter_prompt_len  =>  i >= drafter_prompt_len - 1
            
            prompt_mask = torch.arange(seq_len - 1, device=device) >= (drafter_prompt_len - 1)
            eval_mask = eval_mask & prompt_mask
            
            # b. Mask out rejected tokens
            if indices_to_remask:
                # indices_to_remask are absolute indices in verified_output
                for idx in indices_to_remask:
                    # We need to map absolute idx to shift_labels idx.
                    # shift_labels[i] is verified_output[i+1]
                    # So if verified_output[idx] is rejected, we want to mask shift_labels[idx-1]
                    shifted_idx = idx - 1
                    if 0 <= shifted_idx < eval_mask.size(1):
                        eval_mask[0, shifted_idx] = False

            # 5. Sum and Count
            valid_loss = loss * eval_mask
            
            total_accepted_nll += valid_loss.sum().item()
            total_accepted_count += eval_mask.sum().item()
        # ==============================================================================

        # Loop Control
        if iteration == max_iterations - 1:
            final_output = verified_output
            break
        
        current_state = verified_output.clone()
        for idx in indices_to_remask:
            if idx < current_state.size(1): current_state[0, idx] = drafter_mask_id
    else:
        final_output = verified_output

    stats['accepted_nll_sum'] = total_accepted_nll
    stats['accepted_token_count'] = total_accepted_count
    
    output_text = drafter_tokenizer.decode(final_output[0], skip_special_tokens=True)
    
    return {
        'output_ids': final_output,
        'output_text': output_text,
        'stats': stats
    }

# Helpers
def prepare_masked_query(query, max_new_tokens, tokenizer, mask_id, device):
    msgs = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": query}]
    prompt = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    p_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    masks = torch.full((1, max_new_tokens), mask_id, dtype=torch.long, device=device)
    return torch.cat([p_ids, masks], dim=1)

def run_drafter_phase(model, tokenizer, input_ids, num_steps, max_new_tokens, mask_id, generate_fn, **kwargs):
    # Strip hook from kwargs to avoid errors if model doesn't support it
    kwargs.pop('generation_tokens_hook_func', None)
    
    if generate_fn: 
        return generate_fn(model, tokenizer, input_ids, num_steps, **kwargs)
    return model.generate(input_ids, tokenizer=tokenizer, steps=num_steps, max_new_tokens=max_new_tokens, **kwargs)

def run_verifier_phase(model, tokenizer, input_ids, num_steps, mask_id, generate_fn, **kwargs):
    if generate_fn: return generate_fn(model, tokenizer, input_ids, num_steps, **kwargs)
    output = model.generate(input_ids, tokenizer=tokenizer, steps=num_steps, **kwargs)
    if isinstance(output, tuple): return output[0]
    return output
    
def truncate_dynamic_cache(dynamic_cache, new_len):
    if not dynamic_cache: return None
    for i in range(len(dynamic_cache)):
        if dynamic_cache.layers[i].keys is not None:
            dynamic_cache.layers[i].keys = dynamic_cache.layers[i].keys[:, :, :new_len, :].contiguous()
        if dynamic_cache.layers[i].values is not None:
            dynamic_cache.layers[i].values = dynamic_cache.layers[i].values[:, :, :new_len, :].contiguous()
    return dynamic_cache