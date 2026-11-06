"""Train the character-level GPT on Tiny Shakespeare.

Usage:
    python train.py                      # default settings
    python train.py --max-iters 2000     # shorter run
"""

import argparse
from dataclasses import asdict
from pathlib import Path

import torch

from gpt import CharTokenizer, GPTConfig, GPTLanguageModel, TrainConfig, get_batch, load_text, train_val_split


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    t, m = TrainConfig(), GPTConfig()
    p.add_argument("--data", type=Path, default=None, help="path to a text file (default: Tiny Shakespeare)")
    p.add_argument("--out", type=Path, default=Path("checkpoints/model.pt"))
    p.add_argument("--max-iters", type=int, default=t.max_iters)
    p.add_argument("--eval-interval", type=int, default=t.eval_interval)
    p.add_argument("--eval-iters", type=int, default=t.eval_iters)
    p.add_argument("--batch-size", type=int, default=t.batch_size)
    p.add_argument("--lr", type=float, default=t.learning_rate)
    p.add_argument("--block-size", type=int, default=m.block_size)
    p.add_argument("--n-embd", type=int, default=m.n_embd)
    p.add_argument("--n-head", type=int, default=m.n_head)
    p.add_argument("--n-layer", type=int, default=m.n_layer)
    p.add_argument("--dropout", type=float, default=m.dropout)
    p.add_argument("--seed", type=int, default=t.seed)
    return p.parse_args()


@torch.no_grad()
def estimate_loss(model, splits, args, device):
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = torch.zeros(args.eval_iters)
        for k in range(args.eval_iters):
            xb, yb = get_batch(data, args.batch_size, args.block_size, device)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    text = load_text(args.data) if args.data else load_text()
    tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    train_data, val_data = train_val_split(data)
    splits = {"train": train_data, "val": val_data}

    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        dropout=args.dropout,
    )
    model = GPTLanguageModel(config).to(device)
    print(f"device: {device} | parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for step in range(args.max_iters + 1):
        if step % args.eval_interval == 0 or step == args.max_iters:
            losses = estimate_loss(model, splits, args, device)
            print(f"step {step}: train {losses['train']:.4f} | val {losses['val']:.4f}")
        if step == args.max_iters:
            break

        xb, yb = get_batch(train_data, args.batch_size, args.block_size, device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "config": asdict(config), "chars": tokenizer.chars},
        args.out,
    )
    print(f"saved checkpoint to {args.out}")


if __name__ == "__main__":
    main()
