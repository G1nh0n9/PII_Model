#!/usr/bin/env python3
"""
main_baseline.py — Baseline generation (no logits modification)

This script runs the local LM "as-is" and prints the generated text.
It is intended to be used as the baseline counterpart to your PII/privacy version.

Usage:
  python main_baseline.py --model gpt2 --prompt "..." --max_new_tokens 120
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="gpt2", help="HF model name or local path")
    p.add_argument("--prompt", type=str, required=True, help="Prompt to generate from")
    p.add_argument("--max_new_tokens", type=int, default=120)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return p.parse_args()


def resolve_device(choice: str) -> torch.device:
    if choice == "cpu":
        print("Using CPU for inference.")
        return torch.device("cpu")
    if choice == "cuda":
        print("Using CUDA for inference.")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    # auto
    if torch.cuda.is_available():
        print("CUDA is available. Using CUDA for inference.")
    else:
        print("CUDA is not available. Using CPU for inference.")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = resolve_device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    model.eval()

    # Some tokenizers (e.g., GPT-2) have no pad token by default
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)

    out_ids = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        pad_token_id=tokenizer.pad_token_id,
    )

    print(tokenizer.decode(out_ids[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
