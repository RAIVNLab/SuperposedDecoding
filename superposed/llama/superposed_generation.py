# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)

from superposed.llama.model import ModelArgs
from superposed.llama.superposed_model import SuperposedTransformer
from superposed.llama.tokenizer import Tokenizer
from superposed.llama.superpose import Superpose
from superposed.llama.utils import *
from superposed.ngrams.ngram_models import make_models

class SuperposedLlama:
    @staticmethod
    def build(
        ckpt_dir: str,
        tokenizer_path: str,
        max_seq_len: int,
        max_batch_size: int,
        device = None,
        model_parallel_size: Optional[int] = None,
        seed: int = 1,
    ):
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group("nccl")
        if not model_parallel_is_initialized():
            if model_parallel_size is None:
                model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
            initialize_model_parallel(model_parallel_size)

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if device == None:
            torch.cuda.set_device(local_rank)
            device = torch.cuda.current_device()
        torch.manual_seed(seed)

        if local_rank > 0:
            sys.stdout = open(os.devnull, "w")

        start_time = time.time()
        checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
        assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}"
        assert model_parallel_size == len(
            checkpoints
        ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {model_parallel_size}"
        ckpt_path = checkpoints[get_model_parallel_rank()]
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        with open(Path(ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            **params,
        )
        tokenizer = Tokenizer(model_path=tokenizer_path)
        model_args.vocab_size = tokenizer.n_words
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
        # Set up superposed decoding
        model = SuperposedTransformer(model_args)
        model.load_state_dict(checkpoint, strict=False)
        print(f"Loaded in {time.time() - start_time:.2f} seconds")
        return SuperposedLlama(model, tokenizer, device)

    def __init__(self, model: SuperposedTransformer, tokenizer: Tokenizer, device):
        print(device)
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device
        
    @torch.inference_mode()
    def sup_generate(
        self,
        prompt_tokens: List[List[int]],
        smoothing,
        max_gen_len: int,
        n_token_sample: int,
        alpha: int, # weight on bigram probs
        temp: int,
        n_drafts: int = 1, # number of beams
        verbose: bool = False,
        i_weights = None,
        i_length = None,
        ngrams = None,
        get_time: bool = False,
        penalty = 200
    ):
        """
        Run multi-sequence generation using superposed embeddings.
        Args:
            prompt_tokens (List[List[int]]): Initial tokenized prompts
            max_gen_len (int): Maximum numbers of tokens to generate
            alpha (float): Alpha value
            temp (float): Temperature
            n_drafts (int): Number of drafts
            verbose (bool): Whether to save intermediate embeddings for analysis
            bsz (int): Batch size (default = 16)
            i_weights (List[float]): Ngram interpolation weights
            i_length (List[int]): Ngram models to interpolate (1 for bigram, 2 for trigram, etc.)
            ngrams (Tuple): Ngram models 
            get_time (bool): Return information on time spent doing Ngram lookup
            penalty (float): Penalty on uninterpolated drafts
        Returns:
            (alive_seq, alive_ppl), (fin_seq, fin_ppl): Tuple of (n_prompts, n_drafts, seqlen),
            (n_prompts, n_drafts) for sequences still generating and sequences that have finished.
        """
        # Check batch size and prompt lengths
        params = self.model.params
        bsz = len(prompt_tokens)
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        prompt_len = min_prompt_len
        assert max_prompt_len <= params.max_seq_len
        total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)
        pad_id = self.tokenizer.pad_id
        
        # Initialize token tensor and pad where necessary
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device=self.device)
        for k, t in enumerate(prompt_tokens):
            tokens[k, :len(t)] = torch.tensor(t, dtype=torch.long, device=self.device)
        
        # If no generation is possible
        if min_prompt_len == total_len:
            raise RuntimeError("no generation possible")

        # Initialize decoding object
        initial_tokens = tokens.unsqueeze(1).repeat(1, n_drafts, 1)
        superpose = Superpose(initial_tokens, 
                           tokenizer=self.tokenizer,
                           vocab_size=params.vocab_size,
                           smoothing=smoothing,
                           alpha=alpha,
                           i_weights=i_weights,
                           i_length=i_length,
                           ngrams=ngrams,
                           get_time=get_time,
                           penalty=penalty)
        unseen_first = torch.ones(bsz)
        # Superposition matrix
        token_weights = torch.zeros(bsz, self.model.vocab_size)
        if verbose:
            state_list = []
        prev_pos = 0
        # Begin inference
        for cur_pos in range(min_prompt_len, total_len):
            input_text_mask = tokens != pad_id
            # Take model step
            if cur_pos == min_prompt_len:
                token_weights = None
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], 
                                        start_pos=prev_pos, 
                                        token_weights=token_weights, 
                                        verbose=verbose)
            if verbose:
                logits, states = logits
            # Softmax
            if temp > 0:
                probs = torch.softmax(logits[:, -1] / temp, dim=-1)
            else:
                raise RuntimeError("Temperature must be greater than 0 while mixing")
            if verbose:
                states["end_probs"] = probs
                state_list.append(states)
            # Flag prompts on first generation
            is_first = torch.mul(tokens[:, cur_pos] == pad_id, unseen_first)
            unseen_first[is_first.nonzero(as_tuple=True)[0]] = 0
            # Flag prompts not yet generating
            still_prompt = input_text_mask[:, cur_pos]
            # Superposition pass
            token_weights = superpose(probs, still_prompt, is_first, cur_pos, n_token_sample)
            # Do not superpose for prompts not yet generating
            keep_idx = input_text_mask[:, cur_pos].ravel().nonzero()
            keep_token_weights = torch.zeros_like(token_weights)
            keep_token_weights[keep_idx, tokens[keep_idx, cur_pos]] = 1
            token_weights = torch.where(input_text_mask[:, cur_pos].unsqueeze(1).expand(-1, self.model.vocab_size), 
                                        keep_token_weights, token_weights)
            prev_pos = cur_pos
        results = superpose.return_results(prompt_len)
        if verbose:
            torch.save(state_list, "../embeddings.pt")
            return results
        else:
            return results