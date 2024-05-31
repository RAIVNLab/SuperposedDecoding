## FILE TO GRADE DIVERSITY OF OUTPUTS USING SELF-BLEU
import argparse
import os
import pickle

from loguru import logger

from superposed.llama.metrics import *

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", type=str, help="Result file")
    parser.add_argument("prompt_len", help="Length of prefixes")
    parser.add_argument("eval_type", help="Whether to run in `eval` or `tune` mode")
    args = parser.parse_args()
    file_path = args.file_path
    prompt_len = int(args.prompt_len)
    eval_type = args.eval_type
    file_name, file_type = os.path.splitext(file_path)
    logger.info("File: " + file_path)
    logger.info("Mode: " + eval_type)
    logger.info("Prompt Length: " + str(prompt_len))
    
    # Load result file
    with open(file_path, "rb") as f:
        r = pickle.load(f)
    
    if eval_type == "eval":
        # Evaluate results for single file containing a tensor (n_prompts, n_drafts, seq_len)
        n_prompts, n_drafts, seq_len = r.shape
        if n_drafts > 1:
            gens = r[:, :, prompt_len:] # cut prompts
            diversity = calculate_diversity(gens.tolist())
            print(f"Diversity: {diversity}")
        for i in range(n_drafts):    
            u, b, t = calculate_ngram_repetition(r[:, i, :].reshape(n_prompts, -1)[:, prompt_len:].tolist())
            print(f"Unigram Repeat: {u}  Bigram Repeat: {b}  Trigram Repeat: {t}  Avg: {(u + b + t) / 3}")
    elif eval_type == "ablation":
        # Evaluate results for a file containing a dictionary of {parameters : (n_prompts, n_drafts, seq_len)}
        results = {}
        for param in r:
            tr = r[param]
            n_prompts, n_drafts, seq_len = tr.shape
            tr = tr[:, :, prompt_len:] # cut prompts
            diversity = calculate_diversity(tr.tolist())
            logger.info(f"Parameter: {param} Diversity: {diversity}")
            results[param] = diversity
        logger.info("Saving...")
        with open(file_name + "_div." + file_type, "wb") as f:
            pickle.dump(results, f) 
    else:
        logger.error("Invalid evaluation type")
