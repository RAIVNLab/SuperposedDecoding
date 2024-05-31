from typing import Any, List

import torch
from tqdm import tqdm

def prepare_encodings(batch, tokenizer, length):
    """
    Return tokenized version of `batch`.
    Args:
        batch (List[str]): Batch of text to encode
        tokenizer: Llama tokenizer
        length (int): Prompt length
    Returns:
        Tokenized strings
    """
    tokens = tokenizer.encode(batch, True, False)
    new_encodings = []
    for i, encoded_text in enumerate(tokens):
        new_encodings.append(encoded_text[:length])
    return new_encodings

def evaluate_mixed_losses(data: List[List[str]],
                          model: Any,
                          tokenizer: Any,
                          smoothing: str,
                          prompt_len: int,
                          max_gen_len: int,
                          alpha: float,
                          temp: float,
                          n_drafts: int,
                          n_token_sample: int,
                          bsz=16,
                          i_weights = None,
                          i_length = None,
                          ngrams = None,
                          get_time = False,
                          penalty = 200,
                          marker = True):
    """
    Generate `n_drafts` from `data` using the first `prompt_len` tokens as the
    prefix, generating until `max_gen_len` is reached. Returns the drafts
    with the highest probability.
    Args:
        data (List[List[String]]): Input data
        model: Llama model
        tokenizer: Llama tokenizer
        prompt_len (int): Number of tokens in prefix
        max_gen_len (int): Maximum numbers of tokens to generate
        alpha (float): Alpha value
        temp (float): Temperature
        n_drafts (int): Number of drafts
        bsz (int): Batch size (default = 16)
        i_weights (List[float]): Ngram interpolation weights
        i_length (List[int]): Ngram models to interpolate (1 for bigram, 2 for trigram, etc.)
        ngrams (Tuple): Ngram models 
        get_time (bool): Return information on time spent doing Ngram lookup
        penalty (float): Penalty on uninterpolated drafts
        marker (bool): Progress bar toggle
        
    Return:
        sequences (torch.Tensor): Generated sequences (n_prompts, n_drafts, prompt_len+max_gen_len)
        ppl (torch.Tensor): Generation perplexity (n_prompts, n_drafts)
        time (datetime object, only if `get_time` is True): Total time spent doing ngram lookup 
    """
    it = range(0, len(data), bsz)
    if marker:
        it = tqdm(it)
    sequences = torch.zeros(len(data), n_drafts, prompt_len+max_gen_len, dtype=torch.long)
    ppl = torch.zeros(len(data), n_drafts)
    ovr_time = None
    for b_start in it:
        b_end = b_start + bsz
        # Preprocessing
        batch = data[b_start : b_end]
        truncated_tokens = prepare_encodings(batch, tokenizer, prompt_len)
        
        # Inference
        k = model.sup_generate(prompt_tokens=truncated_tokens, 
                                            smoothing=smoothing,
                                            max_gen_len=max_gen_len, 
                                            n_token_sample=n_token_sample,
                                            alpha=alpha, 
                                            temp=temp,
                                            n_drafts=n_drafts,
                                            i_weights=i_weights,
                                            i_length=i_length,
                                            ngrams=ngrams,
                                            get_time=get_time,
                                            penalty=penalty)
        # Update returns
        if not get_time:
            (alive_seq, alive_ppl), (fin_seq, fin_ppl) = k
        else:
            (alive_seq, alive_ppl), (fin_seq, fin_ppl), ngram_time = k
            ovr_time = ngram_time if ovr_time is None else ovr_time + ngram_time
        # seq: n_prompts, n_drafts, prompt_len+max_gen_len
        # ppl: n_prompts, n_drafts
        combined_ppl = torch.cat([alive_ppl, fin_ppl], dim=1) # n_prompts, 2*n_drafts
        combined_seq = torch.cat([alive_seq, fin_seq], dim=1) # n_prompts, 2*n_drafts, prompt_len+max_gen_len
        top_ppl, top_idx = torch.topk(combined_ppl, n_drafts, dim=-1, largest=False)
        top_seq = torch.take_along_dim(combined_seq, top_idx.unsqueeze(dim=2), dim=1) # n_prompts, n_drafts, prompt_len+max_gen_len
        ppl[b_start : b_end, :] = top_ppl
        sequences[b_start : b_end, :, :] = top_seq
    if not get_time:    
        return sequences, ppl
    else:
        return sequences, ppl, ovr_time

def evaluate_nucleus_losses(data,
                            model,
                            tokenizer,
                            prompt_len,
                            max_gen_len,
                            temp,
                            bsz=16,
                            marker=True):
    """
    Generate using nucleus sampling and return results.
    Args:
        data (List[List[String]]): Input data
        model: Model
        tokenizer: Llama tokenizer
        prompt_len (int): Number of tokens to use as prefix
        max_gen_len (int): Maximum number of tokens to generate
        temp (float): Temperature
        bsz (int): Batch size (default = 16)
        marker (bool): Progress bar toggle
    Return:
        sequences (torch.Tensor): Generated sequences (n_prompts, prompt_len+max_gen_len)
        ppl (torch.Tensor): Generation perplexity (n_prompts)
    """
    it = range(0, len(data), bsz)
    if marker:
        it = tqdm(it)
    sequences = torch.zeros(len(data), prompt_len+max_gen_len, dtype=torch.long)
    ppl = torch.zeros(len(data), dtype=torch.float32)
    for b_start in it:
        b_end = b_start + bsz
        # Preprocess
        batch = data[b_start : b_end]
        truncated_tokens = prepare_encodings(batch, tokenizer, prompt_len)
        
        # Inference
        curr_seq, curr_ppl = model.generate(prompt_tokens=truncated_tokens,
                                  max_gen_len=max_gen_len,
                                  temperature=temp,
                                  top_p=0.9,
                                  logprobs=True)
        sequences[b_start : b_end, :] = curr_seq
        ppl[b_start : b_end] = curr_ppl
    return sequences, ppl