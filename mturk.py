import csv
import os
import pickle
import random
random.seed(42)
import torch

from tqdm import tqdm
from superposed.llama.tokenizer import Tokenizer

def decode(tokenizer, encoding):
    """
    Args:
        tokenizer (Any): Tokenizer
        encoding (torch.Tensor): Encoding
    Returns:
        decoding (str)
    """
    eos_locs = (encoding == tokenizer.eos_id).nonzero()
    if len(eos_locs > 0):
        encoding = encoding[:eos_locs[0]]
    return tokenizer.decode(encoding.to(torch.int32).tolist())

# Open result files
prompt_len = 15
tokenizer = Tokenizer('./7B/tokenizer.model')

# Create CSV
def create_general_csv(spd_file, output_file, n_nucleus=1):
    """
    Create a CSV file containing superposed generations from `spd_file`
    and `n_nucleus` generations from a nucleus sampling file, randomly
    arranging the generations on rows and storing the output in `output_file`.
    """
    # Load files
    path, ext = os.path.splitext(spd_file)
    if ext == ".pkl":
        with open(spd_file, "rb") as f:
            mixed_results = pickle.load(f)
    else:
        mixed_results = torch.load(spd_file)
    # Open file containing nucleus sampling generations
    with open("./owt/p15_d3_nucleus_owt.pkl", "rb") as f:
        nucleus_results = pickle.load(f)
    print(f"Mixed Shape: {mixed_results.shape}")
    print(f"Nucleus Shape: {nucleus_results.shape}")
    if len(mixed_results.shape) == 2:
        mixed_results = mixed_results.unsqueeze(1)
    n_prompts, n_drafts, _ = mixed_results.shape
    # Create header
    fields = ["prompt"]
    num_gens = n_drafts + n_nucleus
    for i in range(1, num_gens+1):
        fields.append(f"gen_{i}")
    for i in range(1, num_gens+1):
        fields.append(f"source_{i}")
    # Write file
    with open(output_file, "w") as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(fields)
        idxs = range(5000)
        for i in tqdm(idxs):
            # Unordered sequences first
            mixed_seqs = mixed_results[i, :, :] # (n_drafts, seq_len)
            nucleus_seq = nucleus_results[i, :, :] # (seq_len)
            prompt = decode(tokenizer, mixed_seqs[0, :prompt_len])
            order = []
            # Add nucleus generations
            for j in range(n_nucleus):
                order.append((decode(tokenizer, nucleus_seq[j, prompt_len:]), f"nucleus_{j+1}"))
            # Add spd generations
            for j in range(n_drafts):
                order.append((decode(tokenizer, mixed_seqs[j, prompt_len:]), f"mixed_{j+1}"))
            # Shuffle
            random.shuffle(order)
            # Create row
            row = [prompt.replace("\n", "\\n")]
            # Flags for duplicate and non-ASCII generations
            duplication = set()
            valid = True
            for seq_tuple in order:
                temp = seq_tuple[0].replace("\n", "\\n")
                row.append(temp)
                duplication.add(temp)
                valid = valid and temp.isascii()
            for seq_tuple in order:
                row.append(seq_tuple[1])
            # Add row
            if len(duplication) == num_gens and valid:
                csvwriter.writerow(row)

def filter(to_filter_file, filter_file):
    """
    Internal method to only keep rows in `to_filter_file` if the
    prefix appears in `filter_file`.
    """
    with open(filter_file, "r") as ff:
        with open(to_filter_file, "r") as tff:
            with open (f"filtered_{to_filter_file}", "w") as nf:
                csvwriter = csv.writer(nf)
                csvreader_one = csv.reader(ff)
                csvreader_two = csv.reader(tff)
                for ir in csvreader_one:
                    for jr in csvreader_two:
                        if jr[0] == ir[0]:
                            csvwriter.writerow(jr)
                            break

# Run methods
create_general_csv("owt/0.54_p15_d3_llama7B_owt_best.pt", "mturk_1v1.csv", 1)
create_general_csv("owt/0.54_p15_d3_llama7B_owt.pkl", "mturk_3v2.csv", 2)
