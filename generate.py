"""Generate text from a trained checkpoint.

Usage:
    python generate.py --tokens 500
    python generate.py --prompt "ROMEO:" --tokens 300
"""

import argparse
from pathlib import Path

import torch

from gpt import CharTokenizer, GPTConfig, GPTLanguageModel


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/model.pt"))
    p.add_argument("--prompt", type=str, default="\n")
    p.add_argument("--tokens", type=int, default=500)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.checkpoint, map_location=device)
    tokenizer = CharTokenizer(ckpt["chars"])
    model = GPTLanguageModel(GPTConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    context = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
    out = model.generate(context, max_new_tokens=args.tokens)
    print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
