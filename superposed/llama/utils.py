import torch

def log_prob_to_prob(log_probs, temp=1):
    """
    Convert log probabilities to probability distribution and normalize.
    Args:
        log_probs (torch.Tensor): Log probs (n_prompts, n_drafts, vocab_size)
    Returns:
        Probability distribution (n_prompts, n_drafts, vocab_size)
    """
    # stability constant
    log_probs = log_probs + torch.max(log_probs, dim=-1, keepdim=True)[0]
    probs = torch.softmax(log_probs / temp, dim=-1)
    return probs

def decode(tokenizer, encoding):
    """
    Decode a list of tokens to a string
    Args:
        tokenizer (Any): Tokenizer
        encoding (torch.Tensor): Encoding
    Returns:
        decoding (str)
    """
    pad_locs = (encoding == -1).nonzero()
    if len(pad_locs > 0):
        encoding = encoding[:pad_locs[0].item()]
    return tokenizer.decode(encoding.to(torch.int32).tolist())

def print_gen(gens, logprobs, tokenizer, n_drafts, prompt_len, output_file):
    """
    Print out generations for debugging.
    Args:
        gens (n_prompts * n_drafts, seq_len): Generations to print
        logprobs (n_prompts * n_drafts): Log probs of each generation
        tokenizer (any): Tokenizer
        n_drafts (int): Number of drafts per prompt
        prompt_len (int): Number of tokens in prompt
    """
    n_prompts, n_drafts, seq_len = gens.shape
    gens = gens.reshape(-1, seq_len)
    logprobs = logprobs.flatten()
    count = 0
    for i in range(len(gens)):
        d = decode(tokenizer, gens[i])
        # first draft of this prompt
        if i % n_drafts == 0:
            count = 0
            print("---------------", file=output_file)
            prompt = decode(tokenizer, gens[i][:prompt_len])
            print(f"prompt: {prompt}", file=output_file)
        print(f"logprob: {logprobs[i]} {count}: {d}", file=output_file)
        count += 1
        
def print_probs(next_probs, tokenizer, output_file):
    """
    Print out next token options and probabilities for debugging
    Args:
        next_probs (torch.Tensor): Next token probabilities (n_prompts, n_drafts, vocab_size)
        tokenizer (any): Tokenizer
    """
    print("\tReminder: At most first n_drafts from seq can be selected.", file=output_file)
    n_prompts, n_drafts, vocab_size = next_probs.shape
    for p_idx in range(n_prompts):
        print(f"\tPrompt {p_idx}:", file=output_file)
        for d_idx in range(n_drafts):
            next_token_probs, next_token_idx = next_probs[p_idx, d_idx].topk(n_drafts+2, dim=-1)
            print(f"\t\tTokens: {[tokenizer.decode([i.item()]) for i in next_token_idx]}", file=output_file)
            print(f"\t\tLog Probs: {torch.log(next_token_probs)}", file=output_file)
            print(f"\t\tProbs: {next_token_probs}", file=output_file)