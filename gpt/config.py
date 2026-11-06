from dataclasses import dataclass


@dataclass
class GPTConfig:
    """Model hyperparameters."""

    vocab_size: int = 65
    block_size: int = 64  # maximum context length
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 4
    dropout: float = 0.2


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    batch_size: int = 32
    max_iters: int = 10_000
    eval_interval: int = 500
    eval_iters: int = 200
    learning_rate: float = 3e-4
    train_split: float = 0.9
    seed: int = 1337
