import pickle
import sys

import torch

class NGram():
  def __init__(self, corpus, corpus_counts, type):
    self.corpus = corpus
    self.counts = corpus_counts
    self.type = type

  def prob(self, key, next):
    """
    Args:
      key (tuple): tuple of token ID's forming prior
      next (int): probability of next token
    """
    l = len(key)
    if self.type == "bigram":
      assert l == 1
      key = key[0]
    elif self.type == "trigram":
      assert l == 2
    elif self.type == "fourgram":
      assert l == 3
    elif self.type == "fivegram":
      assert l == 4
    elif self.type == "sixgram":
      assert l == 5
    elif self.type == "sevengram":
      assert l == 6
      
    count = 0
    if key in self.corpus:
      count = self.corpus[key].get(next, 0)
      total = sum(self.corpus[key].values())
      return count / total
    else:
      return -1
    
  def ntd(self, key, vocab_size=32000):
    """
    Args:
      key (tuple): tuple of token ID's forming prior
    Returns:
      prob_tensor (torch.Tensor): (vocab_size, ) of full next token probabilities
    """
    if key in self.corpus:
      prob_tensor = torch.zeros(vocab_size)
      total = sum(self.corpus[key].values())
      for next_token in self.corpus[key]:
        prob_tensor[next_token] = self.corpus[key][next_token] / total
      return prob_tensor
    else:
      return None

def make_models(ckpt_path, bigram, trigram, fourgram, fivegram, sixgram, sevengram):
  """
  Loads and returns a list correspoding to bigram to sevengram models, containing
  the models that whose parameters are `True`. See below for expected corpus names.
  Args:
    ckpt_path (str): Location of ngram models
    bigram-sevengram: Which models to load
  Returns:
    List of n-gram models
  """
  models = []
  if bigram:
    print("Making bigram...")
    with open(f"{ckpt_path}/b_d_final.pkl", "rb") as f:
        bigram = pickle.load(f)
    bigram_model = NGram(bigram, None, "bigram")    
    models.append(bigram_model)
    print(sys.getsizeof(bigram))
    
  if trigram:
    print("Making trigram...")
    with open(f"{ckpt_path}/t_d_final.pkl", "rb") as f:
        trigram = pickle.load(f)
    trigram_model = NGram(trigram, None, "trigram")
    models.append(trigram_model)
    print(sys.getsizeof(trigram))
    
  if fourgram:
    print("Making fourgram...")
    with open(f"{ckpt_path}/fo_d_final.pkl", "rb") as f:
        fourgram = pickle.load(f)
    fourgram_model = NGram(fourgram, None, "fourgram")
    models.append(fourgram_model)
    print(sys.getsizeof(fourgram))
  
  if fivegram:
    print("Making fivegram...")
    with open(f"{ckpt_path}/fi_d_final.pkl", "rb") as f:
        fivegram = pickle.load(f)
    fivegram_model = NGram(fivegram, None, "fivegram")
    models.append(fivegram_model)
    print(sys.getsizeof(fivegram))
      
  if sixgram:
    print("Making sixgram...")
    with open(f"{ckpt_path}/si_d_final.pkl", "rb") as f:
        sixgram = pickle.load(f)
    sixgram_model = NGram(sixgram, None, "sixgram")
    models.append(sixgram_model)
    print(sys.getsizeof(sixgram))

  if sevengram:
    print("Making sevengram...")
    with open(f"{ckpt_path}/se_d_final.pkl", "rb") as f:
        sevengram = pickle.load(f)
    sevengram_model = NGram(sevengram, None, "sevengram")
    models.append(sevengram_model)

  return models