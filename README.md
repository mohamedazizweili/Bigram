# Character-Level GPT from Scratch

A small decoder-only Transformer (GPT-style) language model written from scratch in PyTorch and trained on the [Tiny Shakespeare](https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare) dataset. It generates text one character at a time.

The project goes step by step from a simple bigram model to a full Transformer with multi-head self-attention, feed-forward layers, residual connections, LayerNorm and dropout.

## Project structure

```
.
├── gpt/
│   ├── config.py      # model and training hyperparameters (dataclasses)
│   ├── data.py        # dataset download, character tokenizer, batching
│   └── model.py       # Head, MultiHeadAttention, FeedForward, Block, GPT + bigram baseline
├── notebooks/
│   └── exploration.ipynb   # step-by-step development notebook
├── train.py           # training script (saves a checkpoint)
├── generate.py        # sample text from a checkpoint
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

The dataset is downloaded to `data/input.txt` automatically the first time you train.

## Usage

Train with the default settings (4 layers, 4 heads, 128-dim embeddings, 64-character context, about 0.8M parameters):

```bash
python train.py
```

Every hyperparameter can be overridden from the command line:

```bash
python train.py --max-iters 5000 --n-layer 6 --n-embd 192 --lr 3e-4
python train.py --help
```

Generate text from the trained model:

```bash
python generate.py --tokens 500
python generate.py --prompt "ROMEO:" --tokens 300
```

## Results

With the default configuration, the validation loss drops from about 4.16 at the start (close to ln 65 ≈ 4.17, which is random guessing over a 65-character vocabulary) to about **1.82** by step 3,500, and keeps going down after that. Sample output after training:

```
Second Senator:
That he lighty thus what he
Be the your eaptions
Farewed with thee say a day to my tress'd endier
```

## Model

| Component | Details |
|---|---|
| Embeddings | token + learned positional |
| Blocks | `n_layer` × (pre-LayerNorm → masked multi-head self-attention → residual, pre-LayerNorm → MLP (4× expansion, ReLU) → residual) |
| Output | final LayerNorm → linear head over the vocabulary |
| Regularization | dropout on attention weights, attention projection and MLP output |

## Acknowledgements

Based on Andrej Karpathy's ["Let's build GPT: from scratch, in code, spelled out"](https://www.youtube.com/watch?v=kCc8FmEb1nY) and the paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
