# Conceptor Debias

A PyTorch implementation for debiasing language models using conceptors ([EMNLP 2023](https://aclanthology.org/2023.emnlp-main.661.pdf)).

## Usage

```
conda create cad python=3.9 cupy pkg-config compilers libjpeg-turbo opencv cudatoolkit=11.3 numba -c conda-forge
conda activate cad
pip install -r requirements.txt
pip install -e transformers
```

1. Generate conceptor negation matrix for a model, using different corpus, wordlist, and subspace type:

```
sbatch ./src/scripts/run_conceptor_negation.sh
```

2. Evaluate the debiasing performance of conceptor negation matrix on the SEAT tasks:

```
sbatch ./src/scripts/run_seat.sh
```

3. Evaluate the semantic maintenance of conceptor negation matrix on the GLUE tasks: Refer to [Huggingface's GLUE](https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue.py).

## Code Hierarchy

```
conceptor-debias-llm/
│
├── README.md
├── requirements.txt
│
└── src/
    ├── conceptor.py                    # Core conceptor implementation
    ├── conceptor_negation.py           # Conceptor negation matrix generation
    ├── run_seat.py                     # SEAT evaluation script
    ├── dataloader.py                   # Data loading utilities
    ├── constants.py                    # Configuration constants
    ├── utils.py                        # Utility functions
    │
    ├── model/
    │   ├── __init__.py
    │   └── models.py                   # Custom model implementations
    │
    ├── scripts/
    │   ├── run_conceptor_negation.sh   # Conceptor negation generation script
    │   └── run_seat.sh                 # SEAT evaluation script
    │
    └── data/
        ├── corpora/                    # Text corpora for training
        │   ├── brown/                  # Brown corpus files
        │   ├── reddit.txt              # Reddit corpus
        │   └── sst.txt                 # Stanford Sentiment Treebank
        │
        ├── weat_seat/                  # WEAT/SEAT bias evaluation datasets
        │   ├── weat1.jsonl - weat10.jsonl
        │   ├── sent-weat*.jsonl        # Sentence-level WEAT tests
        │   ├── heilman_double_bind_*.jsonl
        │   └── angry_black_woman_stereotype*.jsonl
        │
        └── wordlist/                   # Bias attribute word lists
            ├── cmu/                    # CMU gender word lists
            │   ├── female.txt
            │   └── male.txt
            ├── corefBias/              # Coreference bias word lists
            ├── gn_glove/               # Gender-neutral GloVe word lists
            └── survey/                 # Survey-based bias attributes
                └── bias_attribute_words.json

# Output Structure (generated during execution)
output/
└───corpus_type/                        # brown, sst, reddit, wikipedia-2.5, wikipedia-10
    └───model_version/                  # bert-base-uncased, bert-large-uncased, gpt2, gpt2-large, gpt-j
        ├───corpus-embeds/              # Corpus embeddings storage
        │   └───layer_[0-N]/            # Layer-specific embeddings
        │           └───corpus_embeds.pickle  # {corpus_sents, corpus_embeds}
        │
        └───bias_type[special_token]/   # gender, race, gender-umap, gender-tsne
            └───layer_[0-N]/            # Layer-specific results
                └───wordlist_percentile[0.1-1.0]/  # Wordlist percentile filtering
                    └───embed_type/     # avg (embedding type)
                        ├───result-seat.csv        # SEAT evaluation results
                        └───subspace_type/         # all, and, extended, pronouns, propernouns, name, geo
                            ├───pick_embeddings_result.pkl  # Selected embeddings
                            └───negc.pkl                   # Conceptor negation matrix
```

## Citation

```bibtex
@inproceedings{yifei2023conceptor,
  title={Conceptor-Aided Debiasing of Large Language Models},
  author={Yifei, Li S and Ungar, Lyle and Sedoc, Jo{\~a}o},
  booktitle={The 2023 Conference on Empirical Methods in Natural Language Processing}
}
```