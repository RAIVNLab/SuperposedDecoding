# SuperposedDecoding [Demo](https://huggingface.co/spaces/ethanlshen/SuperposedDecoding?logs=container)

This is the repository for the paper ["Superposed Decoding: Multiple Generations from a Single Autoregressive Inference Pass"](https://arxiv.org/abs/2405.18400). We provide:
1.  Implementation of Superposed Decoding on Llama-2-7B, 13B, and 70B.
2.  Code to quickly create n-gram models of any size n from an arbitrary set of documents for custom downstream applications.
3.  Evaluation code for TriviaQA, Natural Questions, and Perplexity.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install Superposed Decoding.

```bash
pip install -r requirements.txt
python setup.py develop
```
If you face any problems with nltk, please manually setup [the package](https://github.com/nltk/nltk).

## Model Weights and N-Gram Corpus Download

In order to use this repository, download [Llama-2 model weights](https://github.com/meta-llama/llama) and one of the n-gram corpuses provided at [this link](https://drive.google.com/drive/folders/1waa-NHFDL7GkttupaxtWATLO4st6Rs8C?usp=sharing). N-gram corpuses are labelled by the number of documents they were trained on. Folders should be stored in the primary working directory.

## N-Gram Model Creation

We provide scaffolding to easily create n-gram models on an arbitrary text dataset using any HuggingFace tokenizer. We only require that the dataset be iterable, with each item having a "text" field. Any HuggingFace dataset can be passed in via the ```--dset_name``` field. Alternatively, local datasets can be used through the field ```--dset_path```.
```
cd superposed/ngrams
```
Example Commands:
1. Create n-gram models on the first 1000 documents (0 to 1000) in RedPajama using the Llama tokenizer. Store results in ./ckpts-test/. Use 10 processes.
```
python make_corpus.py ./ckpts-test/ 0 1000 10 --tok_name=llama --dset_name=togethercomputer/RedPajama-Data-1T-Sample --bigram=y --trigram=y --fourgram=y --fivegram=y --sixgram=y
```
2. Create n-gram models using the BERT tokenizer instead of the Llama tokenizer. Use HuggingFace names for tokenizers.
```
python make_corpus.py ./ckpts-test/ 0 1000 10 --tok_name=google-bert/bert-base-cased --dset_name=togethercomputer/RedPajama-Data-1T-Sample --bigram=y --trigram=y --fourgram=y --fivegram=y --sixgram=y
```
3. Create n-gram models on a custom dataset, example at [test.json](superposed/ngrams/test.json).
```
python make_corpus.py ./ckpts-test/ 0 4 1 --tok_name=llama --dset_path=test.json --bigram=y --trigram=y --fourgram=y --fivegram=y --sixgram=y
```
To use these custom n-grams for Superposed Decoding, simply call make_models() from [ngram_models.py](superposed/ngrams/ngram_models.py) and pass in the result folder. The returned list can be directly plugged into evaluate_mixed_losses() from [eval.py](eval.py) or beam_generate() from [superposed_generation.py](superposed/llama/superposed_generation.py).

## Experiments

We provide notebooks to quickly run experiments using Superposed Decoding.
```
cd superposed/notebooks
```
```nq.ipynb``` and ```triviaqa.ipynb``` contain evaluation for Natural Questions and TriviaQA respectively. ```custom.ipynb``` provides a setup to run Superposed Decoding on arbitrary prompts.

## Citation
You can cite our work with the following entry:
```
@article{shen2024superposed,
  title={Superposed Decoding: Multiple Generations from a Single Autoregressive Inference Pass},
  author={Shen, Ethan and Fan, Alan and Pratt, Sarah M and Park, Jae Sung and Wallingford, Matthew and Kakade, Sham M and Holtzman, Ari and Krishna, Ranjay and Farhadi, Ali and Kusupati, Aditya},
  year={2024},
  url={https://arxiv.org/abs/2405.18400}
}

```
