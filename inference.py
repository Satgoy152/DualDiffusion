

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

    # Create mask for non-masked tokens
    non_masked = (context_tensor != maskid1)

    # Get the non-masked token IDs
    non_masked_ids = context_tensor[non_masked].tolist()

    # Detokenize non-masked tokens to strings using tokenizer1 (batch_decode)
    text_tokens = tokenizer1.batch_decode([[token_id] for token_id in non_masked_ids], skip_special_tokens=False)

    # Retokenize using tokenizer2
    retokenized_ids = [tokenizer2(text)['input_ids'][0] if tokenizer2(text)['input_ids'] else maskid2 for text in text_tokens]

    # Create output tensor filled with maskid2
    output_tensor = torch.full_like(context_tensor, maskid2)

    # Fill in the retokenized IDs at non-masked positions
    output_tensor[non_masked] = torch.tensor(retokenized_ids, dtype=context_tensor.dtype, device=context_tensor.device)

    return output_tensor


def fastdllm_inf(prompt_tensor, context_tensor, steps, context_size):

    return output_tensor