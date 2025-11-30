def LLaDA_inf(prompt_tensor, context_tensor, steps, context_size):

    return output_tensor


def convert(maskid1, maskid2, context_tensor, tokenizer1, tokenizer2):
    """
    Convert context_tensor from tokenizer1's vocabulary to tokenizer2's vocabulary.
    Handles mask tokens by splitting text, decoding segments, and re-encoding.

    Args:
        maskid1: Mask token ID in tokenizer1
        maskid2: Mask token ID in tokenizer2
        context_tensor: Input tensor with token IDs from tokenizer1
        tokenizer1: Source tokenizer
        tokenizer2: Target tokenizer

    Returns:
        Tensor with token IDs from tokenizer2's vocabulary
    """
    import torch

    # Ensure input is 1D or handle batch dimension (assuming batch size 1 for now as per pipeline)
    if context_tensor.dim() == 2:
        flat_ids = context_tensor[0]
    else:
        flat_ids = context_tensor

    # Find mask indices
    mask_indices = (flat_ids == maskid1).nonzero(as_tuple=True)[0]
    
    new_tokens = []
    last_idx = 0
    
    # Iterate through mask positions to split text
    for mask_idx in mask_indices:
        # Decode text before this mask
        if mask_idx > last_idx:
            segment_ids = flat_ids[last_idx:mask_idx]
            # Decode segment
            text_segment = tokenizer1.decode(segment_ids, skip_special_tokens=True)
            if text_segment:
                # Encode segment
                encoded_segment = tokenizer2.encode(text_segment, add_special_tokens=False)
                new_tokens.extend(encoded_segment)
        
        # Add the new mask token
        new_tokens.append(maskid2)
        last_idx = mask_idx + 1
        
    # Handle remaining text after last mask
    if last_idx < len(flat_ids):
        segment_ids = flat_ids[last_idx:]
        text_segment = tokenizer1.decode(segment_ids, skip_special_tokens=True)
        if text_segment:
            encoded_segment = tokenizer2.encode(text_segment, add_special_tokens=False)
            new_tokens.extend(encoded_segment)
            
    # Convert back to tensor
    output_tensor = torch.tensor([new_tokens], device=context_tensor.device)
    
    # Debug prints
    print(f"Conversion: {len(flat_ids)} tokens -> {len(new_tokens)} tokens")
    
    return output_tensor


def fastdllm_inf(prompt_tensor, context_tensor, steps, context_size):

    return output_tensor