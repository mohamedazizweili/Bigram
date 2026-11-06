"""Dataset download, character-level tokenizer and batching."""

import urllib.request
from pathlib import Path

import torch

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "input.txt"


def load_text(path: Path = DEFAULT_DATA_PATH) -> str:
    """Read the dataset, downloading Tiny Shakespeare first if it is missing."""
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading dataset to {path} ...")
        urllib.request.urlretrieve(DATA_URL, path)
    return path.read_text(encoding="utf-8")


class CharTokenizer:
    """Maps each unique character to an integer id and back."""

    def __init__(self, chars):
        self.chars = sorted(set(chars))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, ids) -> str:
        return "".join(self.itos[i] for i in ids)


def train_val_split(data: torch.Tensor, train_fraction: float = 0.9):
    n = int(train_fraction * len(data))
    return data[:n], data[n:]


def get_batch(data: torch.Tensor, batch_size: int, block_size: int, device="cpu"):
    """Sample random (input, target) chunks; target is the input shifted by one."""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)
