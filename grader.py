import os
import pickle
from heapq import heappush

import torch
from loguru import logger
from tqdm import tqdm

from eval import *
from superposed.llama.metrics import *
from superposed.llama.generation import Llama

if __name__ == "__main__":
    # UPDATE THESE PARAMETERS #
    result_file = "./owt/p15_d3_ngram4_llama7B_owt.pkl"  # File containing generations
    mode = "eval"  # Evaluation mode (set to "eval" or "tune")
    prompt_len = 15  # Length of prefixes
    logger.info("File: " + result_file)
    logger.info("Mode: " + mode)
    logger.info("Prompt Length: " + str(prompt_len))

    # Load model
    reg_model = Llama.build(ckpt_dir="./70B/", 
                        tokenizer_path='./7B/tokenizer.model', 
                        max_seq_len=100, 
                        max_batch_size=64,
                        device=None,
                        model_parallel_size=8)

    # Load result file
    filename, ext = os.path.splitext(result_file)
    if ext == ".pt":
        r = torch.load(result_file)
    elif ext == ".pkl":
        with open(result_file, "rb") as f:
            r = pickle.load(f)

    # Main loop
    loop = tqdm(total=len(r), position=0, leave=True)
    if mode == "tune":
        # Evaluate results for a file containing a dictionary of {parameters : (n_prompts, n_drafts, seq_len)}
        heap = []
        # Loop over every hyperparameter combination
        for param in r:
            seqs = r[param]
            n_prompts, n_drafts, gen_len = seqs.shape
            # Calculate and average perplexity across drafts
            output_ppl = calculate_perplexity(reg_model, 
                                              seqs.reshape(n_prompts * n_drafts, -1), 
                                              prompt_len=prompt_len, 
                                              bsz=64, 
                                              marker=False)
            output_ppl = torch.mean(output_ppl)
            # Store perplexity values and the corresponding hyperparameters
            heappush(heap, (output_ppl.item(), param))
            # Update loop info
            loop.set_description(f"Average Perplexity: {output_ppl.item():.4f}")
            loop.update(1)
        # Save results
        logger.info("Saving tuning results...")
        with open(f"{filename}_llama_tune.pkl", "wb") as f:
            pickle.dump(heap, f)
    elif mode == "eval":
        # Evaluate results for single file containing a tensor (n_prompts, n_drafts, seq_len)
        with torch.no_grad():
            n_prompts, n_drafts, gen_len = r.shape
            # Calculate perplexity over all generations
            output_ppl = calculate_perplexity(reg_model, 
                                              r.reshape(n_prompts * n_drafts, -1), 
                                              prompt_len=prompt_len, 
                                              bsz=64, 
                                              marker=True)
            output_ppl = output_ppl.reshape(n_prompts, n_drafts)
            # Calculate average perplexity by draft
            draft_avg = torch.mean(output_ppl, dim=0)    
            draft_std = torch.std(output_ppl, dim=0)
            # Calculate average perplexity of best generations
            best_ppl = output_ppl.min(dim=-1)
            best_avg = torch.mean(best_ppl[0])
            best_std = torch.std(best_ppl[0])
            logger.info(f"Draft Avg: {draft_avg} Draft Std: {draft_std}")
            logger.info(f"Best Avg: {best_avg} Best Std: {best_std}")
            # Save perplexities of each generation
            torch.save(output_ppl, f"{filename}_ppl.pt") 
    else:
        logger.error("Use `eval` or `tune` as evaluation modes")
