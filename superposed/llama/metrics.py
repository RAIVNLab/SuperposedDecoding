import torch
import nltk
from nltk.translate.bleu_score import SmoothingFunction
from tqdm import tqdm

def calculate_perplexity(model, tokens, prompt_len, bsz=1, marker=False):
    """
    Calculate perplexity of given tokens using provided model, ignoring padding tokens. 
    Args:
        model: Llama model
        tokens (List[List[int]] or torch.Tensor): Input tokens (n_prompt * n_draft, seqlen)
        prompt_len (int): Prefix length
        bsz (int): Batch size
        marker (bool): Whether to show progress bar
    Returns:
        Perplexity across all generations (n_prompt * n_drafts)
    """
    it = range(0, len(tokens), bsz)
    if marker:
        it = tqdm(it)
    start = 0
    ppl = torch.zeros(len(tokens))
    for start in it:
        end = start + bsz
        data = tokens[start : end]
        if not isinstance(data, list):
            data = data.tolist()
        # Remove any padding tokens (-1) in generations
        for d_idx in range(len(data)):
            cur = data[d_idx]
            if -1 in cur:
                data[d_idx] = cur[:cur.index(-1)]
        # Calculate cross entropy loss on tokens
        ce_loss = model.generate(data, max_gen_len=0, temperature=-1, top_p=-1, grade=True)
        # Cut off everything past `prompt_len`
        ce_loss = ce_loss[:, prompt_len-1:]  # Subtract 1 because the first token (start token) is removed
        # Calculate perplexity 
        lengths = (ce_loss != 0).sum(dim=-1)
        mean = ce_loss.sum(dim=-1) / lengths
        ppl[start : end] = torch.exp(-1 * mean)
    return ppl
    
def calculate_diversity(generations, k=4):
    """
    Calculate diversity of generations using SELF-BLEU.
    Args:
        generations (List[List[List[int]]]): Tokenized input
        k (int, Optional): Number of n-grams to use for bleu
    Returns:
        Average diversity across all generations (float)
    """
    nltk.download('punkt')  # Can be deleted once downloaded
    smooth = SmoothingFunction()
    bleus = []
    
    for drafts in generations:
        tokenized_drafts = []
        # Stringify tokens
        for d in drafts:
            if -1 in d:
                d = d[:d.index(-1)]
            tokenized_drafts.append([str(n) for n in d])
        # Calculate SELF-BLEU
        minlength = min([len(g) for g in tokenized_drafts])
        minlength = min(minlength, k)
        weights = tuple((1. / minlength for _ in range(minlength)))
        for i in range(len(drafts)):
            # Create source and reference (all other drafts)
            src = tokenized_drafts[i]
            ref = tokenized_drafts[:i] + tokenized_drafts[i+1:]
            tmp = nltk.translate.bleu_score.sentence_bleu(references=ref, 
                                                          hypothesis=src, 
                                                          weights=weights,
                                                          smoothing_function=smooth.method1)
            bleus.append(tmp)
    bleus = torch.Tensor(bleus)
    return torch.mean(bleus)


def calculate_ngram_repetition(sequences):
    """
    Calculate uniqueness scores of `sequences`.
    Args:
        sequences (List[List[int]]): Generated sequences
    Returns:
        (unigram_uniqueness, bigram_uniqueness, trigram_uniqueness)
    """
    u_total = 0
    b_total = 0
    t_total = 0
    # Iterate through all sequences indiscriminately
    for gen in sequences:
        if -1 in gen:
            gen = gen[:gen.index(-1)]
        unigrams, bigrams, trigrams = [], [], []
        o = [str(i) for i in gen]
        # Create lists of n-grams for the generation
        for i in range(len(o)):
            unigrams.append(o[i])
        for i in range(len(o) - 1):
            bigrams.append(o[i] + '_' + o[i + 1])
        for i in range(len(o) - 2):
            trigrams.append(o[i] + '_' + o[i + 1] + '_' + o[i + 2])
        # Calculate uniqueness of the generation
        u, b, t = len(set(unigrams)) / len(unigrams), len(set(bigrams)) / len(bigrams), len(set(trigrams)) / len(trigrams)
        u_total += u
        b_total += b
        t_total += t
    return u_total / len(sequences), b_total / len(sequences), t_total / len(sequences)
