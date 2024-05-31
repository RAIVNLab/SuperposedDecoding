# Implementation loosely based on https://github.com/tensorflow/tensor2tensor/blob/bafdc1b67730430d38d6ab802cbd51f9d053ba2e/tensor2tensor/utils/beam_search.py#L554
import requests
import time
from datetime import datetime, timedelta
from typing import Optional, Literal

import torch
import torch.nn as nn
from transformers import LlamaTokenizer

from superposed.llama.utils import *
from superposed.ngrams.ngram_models import NGram

INF = 1. * 1e7

# Test by scaling # beams & verify work
class Superpose(nn.Module): 
    def __init__(self, 
                 initial_tokens,
                 tokenizer,
                 vocab_size,
                 smoothing=Optional[Literal["geom", "all"]],
                 alpha = None,
                 verbose = False,
                 i_weights = None,
                 i_length = None,
                 ngrams = None,
                 sample_beams = False,
                 sample_tokens = False,
                 get_time = False,
                 penalty = 200): # default no effect
        """
        Initialize a beam search class.
        
        Args:
            initial_tokens (torch.Tensor): Initial tokens
            n_prompts (int): Number of prompts
            tokenizer (Tokenizer): Llama tokenizer
            vocab_size (int): Total vocab size
            smoothing (str): Smoothing method ("geom" for default, "all" for only ngram, None for no ngram)
            ngram_length (int): N gram length to consider
            alpha (float): Alpha parameter
            debug (bool): Whether to print information
        """
        super().__init__()
        # primary parameters
        self.n_prompts, self.n_drafts, _ = initial_tokens.shape
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.alive_seq = initial_tokens
        self.fin_seq = initial_tokens
        self.smoothing = smoothing
        self.alive_log_probs = torch.zeros(self.n_prompts, self.n_drafts)
        self.fin_log_probs = torch.full((self.n_prompts, self.n_drafts), float("-inf"))
        self.alpha = alpha
        self.verbose = verbose
        self.penalty = penalty
        # devices
        self.cpu = torch.device('cpu')
        self.gpu = torch.device('cuda')
        # Interpolation length and weights
        self.interpolation_weights = i_weights
        self.i_length = i_length
        # N-grams
        self.bigram = ngrams[0] if len(ngrams) >= 1 else None
        self.trigram = ngrams[1] if len(ngrams) >= 2 else None
        self.fourgram = ngrams[2] if len(ngrams) >= 3 else None
        self.fivegram = ngrams[3] if len(ngrams) >= 4 else None
        self.sixgram = ngrams[4] if len(ngrams) >= 5 else None
        self.sevengram = ngrams[5] if len(ngrams) >= 6 else None
        # Timing
        self.get_time = get_time
        self.lookup_time = None

    def forward(self, probs, still_prompt, is_first, cur_pos, n_token_sample):
        """
        Apply beam decoding to update generations.
        
        Args:
            probs (torch.Tensor): Next token probability distribution
            still_prompt (torch.Tensor): Flags of prompts that should not generate yet (n_prompts, )
            is_first (torch.Tensor): Flags of prompts that are on their first generation (n_prompts, )
            cur_pos (int): Current generation position
            n_token_sample (int): Number of tokens from model distribution to use
            
        Return:
            if standard beam search:
                attention_change_ids (torch.Tensor): New indices in kv cache (n_prompts, n_drafts)
            if mixed:
                token_weights (torch.Tensor): Mixing weights (n_prompts, vocab_size)
        """        
        # Adjust input probabilities
        probs = self.get_top_k(probs, 32000, n_token_sample)
        reshaped_probs = probs.reshape(self.n_prompts, 1, -1)
        reshaped_probs = reshaped_probs.repeat(1, self.n_drafts, 1)
        # Ngram smoothing 
        if self.smoothing is not None:
            if self.smoothing == "geom":
                ngram_probs = self.ngram_probs(self.alive_seq, cur_pos, probs=probs)              
                # Make mask and normalize
                prob_mask = reshaped_probs != 0
                ngram_probs *= prob_mask    
                # Calculate logprobs and interpolate distributions
                llm_log_probs = torch.log(reshaped_probs)
                ngram_log_probs = torch.log(ngram_probs)
                log_probs = (1 - self.alpha) * llm_log_probs + self.alpha * ngram_log_probs
                # Apply penalty to drafts where no interpolation occurred
                is_all_inf = (log_probs != float("-inf")).sum(dim=-1, keepdims=True) == 0 
                log_probs = torch.where(is_all_inf, (1 - self.alpha) * llm_log_probs - self.penalty, log_probs)
            elif self.smoothing == "all":
                ngram_probs = self.ngram_probs(self.alive_seq, cur_pos, probs=None)              
                log_probs = torch.log(ngram_probs)
        else:
            log_probs = torch.log(reshaped_probs)
        curr_log_probs = self.alive_log_probs.unsqueeze(dim=2) + log_probs # [n_prompts, n_drafts, vocab_size]
        # Warning if nan
        if (torch.any(torch.isnan(curr_log_probs)).item()):
            raise RuntimeWarning("nan in sequence log probs", file=self.output_file)
        # Potential Sequences
        flat_curr_log_probs = curr_log_probs.reshape(-1, self.vocab_size*self.n_drafts)
        topk_log_probs, topk_idx = torch.topk(flat_curr_log_probs, 2 * self.n_drafts, dim=-1)
        topk_beam_id = topk_idx // self.vocab_size # [n_prompts, 2 * n_drafts]
        topk_idx = topk_idx % self.vocab_size # [n_prompts, 2 * n_drafts]
        # First timestep uses top-k next tokens
        is_first_idx = is_first.nonzero(as_tuple=True)[0]
        if len(is_first_idx) != 0:
            first_time_log_probs = log_probs[is_first_idx][:, 0, :].squeeze(dim=1)
            first_time_log_probs, first_time_topk_idx = torch.topk(first_time_log_probs, 2 * self.n_drafts, dim=1)
            topk_idx[is_first_idx] = first_time_topk_idx
            topk_log_probs[is_first_idx] = self.alive_log_probs[is_first_idx, 0].unsqueeze(dim=1) + first_time_log_probs 
        # New sequences
        topk_seq = torch.take_along_dim(self.alive_seq, topk_beam_id.unsqueeze(2), dim=1) # [n_prompts, 2 * n_drafts, vocab_size]
        topk_seq[:, :, cur_pos] = topk_idx
        topk_finished = topk_idx == self.tokenizer.eos_id
        # Only update sequences for those that have begun generating
        new_alive_seq, new_alive_log_probs = self.grow_alive(topk_seq, topk_log_probs, topk_finished)
        new_fin_seq, new_fin_log_probs = self.grow_fin(topk_seq, topk_log_probs, topk_finished)
        still_prompt_probs = still_prompt.reshape(-1, 1)
        still_prompt_seqs = still_prompt.reshape(-1, 1, 1)
        self.alive_seq = torch.where(still_prompt_seqs, self.alive_seq, new_alive_seq)
        self.alive_log_probs = torch.where(still_prompt_probs, self.alive_log_probs, new_alive_log_probs) 
        self.fin_seq = torch.where(still_prompt_seqs, self.fin_seq, new_fin_seq)
        self.fin_log_probs = torch.where(still_prompt_probs, self.fin_log_probs, new_fin_log_probs)
        # Create superposition matrix and return it
        topk_idx = self.alive_seq[:, :, cur_pos].reshape(self.n_prompts, -1)
        token_weights = self.superposition_matrix(topk_idx)
        return token_weights
        
    def grow_alive(self, topk_seq, topk_log_probs, topk_finished):
        """
        Extend running generations.
        Args:
            topk_seq (torch.Tensor): Top k sequences (n_prompts, 2 * n_drafts, vocab_size)
            topk_log_probs (torch.Tensor): Log probabilities (n_prompts, 2 * n_drafts)
            topk_finished (torch.Tensor): Whether a sequence is finished (n_prompts, 2 * n_drafts) 
        Returns:
            new_alive_seq, new_alive_log_probs
        """
        topk_log_probs = topk_log_probs + topk_finished * -INF 
        new_alive_log_probs, new_alive_idx = torch.topk(topk_log_probs, self.n_drafts, dim=1)
        new_alive_seq = torch.take_along_dim(topk_seq, new_alive_idx.unsqueeze(2), dim=1)
        return new_alive_seq, new_alive_log_probs
        
    def grow_fin(self, topk_seq, topk_log_probs, topk_finished):
        """
        Update stopped generations. 
        Args:
            topk_seq (torch.Tensor): Top k sequences (n_prompts, 2 * n_drafts, vocab_size)
            topk_log_probs (torch.Tensor): Log probabilities (n_prompts, 2 * n_drafts)
            topk_finished (torch.Tensor): Whether a sequence is finished (n_prompts, 2 * n_drafts) 
            
        Returns:
            new_fin_seq, new_fin_log_probs
        """
        topk_log_probs = topk_log_probs + ~topk_finished * -INF 
        new_fin_seq = torch.cat([self.fin_seq, topk_seq], dim=1)
        new_fin_log_probs = torch.cat([self.fin_log_probs, topk_log_probs], dim=1)
        new_fin_log_probs, new_fin_idx = torch.topk(new_fin_log_probs, self.n_drafts, dim=1)
        new_fin_seq = torch.take_along_dim(new_fin_seq, new_fin_idx.unsqueeze(2), dim=1)
        return new_fin_seq, new_fin_log_probs

    def get_top_k(self, probs, m, k):
        """
        Zero out all but top-k tokens in a probability distribution.
        Args:
            probs (torch.Tensor): Probability distribution tensor.
            m (float): Number of tokens to consider (only relevant when sampling).
            k (int): Number of tokens to sample/keep.
        Returns:
            torch.Tensor: New probability distribution based on renormalized probabilities. 
        """
        n_prompts, _ = probs.shape 
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        top_k_mask = torch.arange(probs.shape[-1])
        top_k_mask = top_k_mask.expand(probs.shape[0], -1)
        top_k_mask = top_k_mask >= m # Set to 1 past k elements
        probs_sort[top_k_mask] = 0.0 # Zero wherever mask = 1
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        next_token = torch.gather(probs_idx, -1, torch.topk(probs_sort, k, dim=-1)[1])       
        # Set all other probs to 0
        new_probs_map = torch.zeros(probs.shape).bool()
        new_probs_map[torch.repeat_interleave(torch.arange(n_prompts), k), torch.flatten(next_token)] = True
        new_probs = torch.where(new_probs_map, probs, 0)
        # Renormalize
        new_probs.div_(new_probs.sum(dim=-1, keepdim=True))
        return new_probs
    
    def superposition_matrix(self, tokens):
        """
        Create superposition matrix based on provided tokens.
        Args:
            tokens (torch.Tensor): Tokens to mix (n_prompts, n_drafts)
        Returns:
            SUperposition matrix
        """
        # Create superposition matrix
        mixing_matrix = torch.zeros(self.n_prompts, self.vocab_size)
        # Convert draft log probs to probabilities
        weightings = log_prob_to_prob(self.alive_log_probs)
        # Update probabilities in superposition matrix with draft probabilities
        for p_idx in range(self.n_prompts):
            for d_idx in range(self.n_drafts):
                tok_idx = tokens[p_idx][d_idx]
                mixing_matrix[p_idx][tok_idx] += weightings[p_idx][d_idx]
        # Renormalize
        mixing_matrix.div_(mixing_matrix.sum(dim=-1, keepdims=True))
        return mixing_matrix
    
    def ngram_probs(self, alive_seq, cur_pos, probs):
        """
        Calculate and return next token distribution using ngram models.
        Args:
            alive_seq (torch.Tensor): Current drafts (n_prompts, n_drafts, seqlen)
            cur_pos (int): Current timestep
            probs (torch.Tensor): Current next probability distribution from model (n_prompts, vocab_size).
            As described in the paper, only tokens w/nonzero probability in `prob` are considered for the
            ngram distribution. However, passing in `None` as `probs` will consider all tokens.
        Returns:
            Next token distribution for each draft (n_prompts, n_drafts, vocab_size)
        """
        if self.get_time:
            # Start timer
            start_time = datetime.now()
        # Create distribution matrix
        next_token_probs = torch.zeros(self.n_prompts, self.n_drafts, 32000)
        if probs is not None:
            # Loop over all prefixes
            for p_idx in range(len(alive_seq)):
                # List of possible tokens for the prefix
                nz = torch.nonzero(probs[p_idx, :], as_tuple=True)[0].tolist()
                # Generate next token distribution
                for draft_idx in range(self.n_drafts):
                    i_mask = torch.sum(torch.tensor(self.i_length) <= cur_pos)
                    new_i_weights = self.interpolation_weights[:i_mask]
                    new_i_length = self.i_length[:i_mask]
                    # For each next token
                    for nt in nz:
                        # Calculate probability using ngram interpolation
                        for i, weight in zip(new_i_length, new_i_weights):
                            if cur_pos - i >= 0:
                                key = tuple(alive_seq[p_idx, draft_idx, cur_pos-i:cur_pos].tolist())
                                if i == 1:
                                    prob = self.bigram.prob(key, nt)
                                elif i == 2:
                                    prob = self.trigram.prob(key, nt)
                                elif i == 3:
                                    prob = self.fourgram.prob(key, nt)
                                elif i == 4:
                                    prob = self.fivegram.prob(key, nt)
                                elif i == 5:
                                    prob = self.sixgram.prob(key, nt)
                                elif i == 6:
                                    prob = self.sevengram.prob(key, nt)
                            if prob >= 0:
                                next_token_probs[p_idx, draft_idx, nt] += weight * prob
        else:
            for p_idx in range(len(alive_seq)):
                for draft_idx in range(self.n_drafts):
                    i_mask = torch.sum(torch.tensor(self.i_length) <= cur_pos)
                    new_i_weights = self.interpolation_weights[:i_mask]
                    new_i_length = self.i_length[:i_mask]
                    for i, weight in zip(new_i_length, new_i_weights):
                        if cur_pos - i >= 0:
                            key = tuple(alive_seq[p_idx, draft_idx, cur_pos-i:cur_pos].tolist())
                            if i == 1:
                                ntd = self.bigram.ntd(key)
                            elif i == 2:
                                ntd = self.trigram.ntd(key)
                            elif i == 3:
                                ntd = self.fourgram.ntd(key)
                            elif i == 4:
                                ntd = self.fivegram.ntd(key)
                            elif i == 5:
                                ntd = self.sixgram.ntd(key)
                            elif i == 6:
                                ntd = self.sevengram.ntd(key)
                        if ntd is not None:
                            next_token_probs[p_idx, draft_idx, :] += weight * ntd
        if self.get_time:    
            total_time = datetime.now() - start_time
            self.lookup_time = total_time if self.lookup_time is None else self.lookup_time + total_time
        return next_token_probs

    def return_results(self, prompt_len=None):
        """
        Return generations and perplexities
        
        Args:
            prompt_len (int): Length of prompt in tokens. If is None, then ppl is not calculated.
        Returns:
            (self.alive_seq, alive_ppl), (self.fin_seq, fin_ppl) 
            OR
            (self.alive_seq, alive_ppl), (self.fin_seq, fin_ppl), self.lookup_time
        """
        # PPL
        alive_ppl = 0
        fin_ppl = 0
        if prompt_len is not None:
            alive_ppl = torch.exp(self.alive_log_probs / (-1 * (self.alive_seq.size(dim=-1)-prompt_len)))      
            # Fin ppl
            fin_seq_lengths = (self.fin_seq != self.tokenizer.pad_id).sum(dim=-1)
            fin_ppl = torch.exp(self.fin_log_probs / (-1 * (fin_seq_lengths - prompt_len)))
            fin_ppl += ((fin_ppl == 0) * float("inf"))
        # print time
        if not self.get_time:
            return (self.alive_seq.to(torch.long), alive_ppl), (self.fin_seq.to(torch.long), fin_ppl)
        else:
            return (self.alive_seq.to(torch.long), alive_ppl), (self.fin_seq.to(torch.long), fin_ppl), self.lookup_time