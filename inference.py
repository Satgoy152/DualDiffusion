def LLaDA_inf(prompt_tensor, context_tensor, steps, context_size):

    return output_tensor


def convert(maskid1, maskid2, context_tensor, tokenizer1, tokenizer2):
    """
    Convert context_tensor from tokenizer1's vocabulary to tokenizer2's vocabulary.

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

    # Decode entire sequence (masks → special placeholder)
    mask_positions = (context_tensor == maskid1)
    
    # Replace masks with pad token for decoding
    temp_tensor = context_tensor.clone()
    # temp_tensor[mask_positions] = tokenizer1.pad_token_id
    
    # Decode to text using batch_decode - MUST CONVERT TO LIS
    text = tokenizer1.batch_decode(temp_tensor.tolist(), skip_special_tokens=False)
    print("convert tokenizer 1 to text", text)
    # Re-encode with tokenizer2
    new_ids = tokenizer2(text)['input_ids']
    
    # Handle length mismatch - proportional masking
    # original_mask_ratio = mask_positions.float().mean()
    # num_masks = int(len(new_ids) * original_mask_ratio)
    
    output_tensor = torch.tensor(new_ids, dtype=context_tensor.dtype, 
                                  device=context_tensor.device)
    output_tensor[mask_positions] = maskid2
    # Mask tokens uniformly
    # if num_masks > 0:
    #     mask_indices = torch.linspace(0, len(new_ids)-1, num_masks).long()
    #     output_tensor[mask_indices] = maskid2
    
    return output_tensor


def fastdllm_inf(prompt_tensor, context_tensor, steps, context_size):

    return output_tensor