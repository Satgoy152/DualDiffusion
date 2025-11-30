import torch
import numpy as np
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel


def set_seed(seed):
    """Sets the random seed for reproducibility across different components."""
    # NumPy random number generator
    np.random.seed(seed)
    
    # PyTorch CPU random number generator
    torch.manual_seed(seed)
    
    # PyTorch CUDA random number generator (for all GPUs)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Ensure deterministic algorithms for cuDNN (if used)
    # This might slightly reduce performance but ensures reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


@ torch.no_grad()
def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device) # create a sequence of masked tokens size prompt len + gen len
    x[:, :prompt.shape[1]] = prompt.clone() # copy prompt to beginning

    set_seed(42)

    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)
    

    prompt_index = (x != mask_id) # array of which indexes are from prompt
    prompt_embeds = model.model.transformer.wte(x[:, :prompt.shape[1]]) # prompt embedded

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length # each block is of block length size

    assert steps % num_blocks == 0
    steps = steps // num_blocks # steps per block

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id) # get current block
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps) # number of tokens to unmask
        print(num_transfer_tokens)
        for i in range(steps):
            mask_index = (x == mask_id) # all currently masked tokens
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                out = model(x, output_hidden_states=True)
                # logits = out.logits
                hidden_states = out.hidden_states

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l


            probs = F.softmax(logits[0], dim=-1)  # Shape: (seq_len, vocab_size)
            top10_probs, top10_indices = torch.topk(probs, k=10, dim=-1)  # Both: (seq_len, 10)

            # # Create Nx10x2 tensor: [token_id, probability]
            # N = probs.shape[0]
            # top10_table = torch.zeros(N, 10, 2)
            # top10_table[:, :, 0] = top10_indices.float()  # Token IDs
            # top10_table[:, :, 1] = top10_probs  # Probabilities
            
            # # Print table for each position
            # print(f"\nTop-10 tokens per position at step {i}:")
            # pos = 133
            # # Only show masked positions to reduce clutter
            # print(f"\nPosition {pos}:")
            # print(f"  {'Rank':<6} {'Token':<30} {'Probability':<12}")
            # print(f"  {'-'*48}")
            # for rank in range(10):
            #     token_id = int(top10_table[pos, rank, 0].item())
            #     prob = top10_table[pos, rank, 1].item()
            #     token_str = tokenizer.decode([token_id]).replace('\n', '\\n')
            #     selected = "◄ SELECTED" if token_id == x0[0, pos].item() else ""
            #     print(f"  {rank+1:<6} {token_str:<30} {prob:<12.6f} {selected}")
            # pos = pos + 1
            # # Only show masked positions to reduce clutter
            # print(f"\nPosition {pos}:")
            # print(f"  {'Rank':<6} {'Token':<30} {'Probability':<12}")
            # print(f"  {'-'*48}")
            # for rank in range(10):
            #     token_id = int(top10_table[pos, rank, 0].item())
            #     prob = top10_table[pos, rank, 1].item()
            #     token_str = tokenizer.decode([token_id]).replace('\n', '\\n')
            #     selected = "◄ SELECTED" if token_id == x0[0, pos].item() else ""
            #     print(f"  {rank+1:<6} {token_str:<30} {prob:<12.6f} {selected}")
            # pos = pos + 1
            # # Only show masked positions to reduce clutter
            # print(f"\nPosition {pos}:")
            # print(f"  {'Rank':<6} {'Token':<30} {'Probability':<12}")
            # print(f"  {'-'*48}")
            # for rank in range(10):
            #     token_id = int(top10_table[pos, rank, 0].item())
            #     prob = top10_table[pos, rank, 1].item()
            #     token_str = tokenizer.decode([token_id]).replace('\n', '\\n')
            #     selected = "◄ SELECTED" if token_id == x0[0, pos].item() else ""
            #     print(f"  {rank+1:<6} {token_str:<30} {prob:<12.6f} {selected}")

            # print(f"{'='*80}\n")

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index] # re mask
            # print("X: ", x)
            # print("Len x:", len(x[0]))
            # print("not masked:", (x != mask_id))
            # x[mask_index] = x0[mask_index]
            # print("X: ", x)
            # print("Len x:", len(x[0]))
            # print("not masked:", (x != mask_id))

            predicted_tokens = [tokenizer.decode([token_id.item()]) for token_id in x[0]]
            # print(f"\n{'='*80}")
            # print(f"STEP {i} - Block {num_block}")
            # print(f"{'='*80}")
            # print("Predicted tokens (before remasking):")
            # print(predicted_tokens[prompt.shape[1]:])

    return x


@ torch.no_grad()
def generate_per_step(model, prompt, n, k, gen_length=128, block_length=128, temperature=0.,
                      cfg_scale=0., remasking='low_confidence', mask_id=126336):
    '''
    Generate tokens by running for exactly n steps and unmasking k tokens per step.
    Some tokens may remain masked in the final output.

    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        n: Number of steps to run.
        k: Number of tokens to unmask per step.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device) # create a sequence of masked tokens size prompt len + gen len
    x[:, :prompt.shape[1]] = prompt.clone() # copy prompt to beginning

    set_seed(42)

    # tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)


    prompt_index = (x != mask_id) # array of which indexes are from prompt
    prompt_embeds = model.model.transformer.wte(x[:, :prompt.shape[1]]) # prompt embedded

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length # each block is of block length size

    assert n % num_blocks == 0
    steps_per_block = n // num_blocks # steps per block

    all_logits = []

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id) # get current block

        for i in range(steps_per_block):
            mask_index = (x == mask_id) # all currently masked tokens
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                out = model(x, output_hidden_states=True)
                logits = out.logits
                hidden_states = out.hidden_states

            # Store logits for the generated part
            # logits shape: (batch, seq_len, vocab)
            # We want (generation_length, vocab) for this step (assuming batch=1)
            gen_logits = logits[:, prompt.shape[1]:, :]
            all_logits.append(gen_logits)

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                # Unmask exactly k tokens per step
                num_masked = (confidence[j] != -np.inf).sum().item()
                num_to_unmask = min(k, num_masked)  # Don't try to unmask more than available
                if num_to_unmask > 0:
                    _, select_index = torch.topk(confidence[j], k=num_to_unmask)
                    transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index] # unmask selected tokens

            # predicted_tokens = [tokenizer.decode([token_id.item()]) for token_id in x[0]]
            # print(predicted_tokens[prompt.shape[1]:])

    stacked_logits = torch.cat(all_logits, dim=0) # (steps, gen_len, vocab)
    return x, stacked_logits


def main():
    device = 'cuda'

    model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    m = [{"role": "user", "content": prompt}, ]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

    input_ids = tokenizer(prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

    out = generate(model, input_ids, steps=128, gen_length=128, block_length=32, temperature=0., cfg_scale=0., remasking='low_confidence')
    print(tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0])


if __name__ == '__main__':
    main()
