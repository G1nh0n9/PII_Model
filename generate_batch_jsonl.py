#!/usr/bin/env python3
"""
generate_batch_jsonl.py

Create OpenAI Batch API JSONL inputs for 3 variants per question:

1) passthrough:
   - Send the original (raw) question directly to the cloud LLM.

2) local_baseline:
   - Run a local LM "as-is" on the raw question to produce a rewritten prompt,
     then send that rewritten prompt to the cloud LLM.

3) local_privacy:
   - Run the local LM with STRICT decoding-time logits modification:
        z'(t) = z(t) - λ * m_{R(q)}(t)
     using main_fixed.py components (query-adaptive greedy R(q)),
     then send that rewritten prompt to the cloud LLM.

This script ONLY generates the Batch input JSONL. It does not call the cloud API.

Example:
  python3 generate_batch_jsonl.py \
    --question_jsonl batch_questions.jsonl \
    --local_model meta-llama/Llama-3.2-1B \
    --cloud_model gpt-4o-mini \
    --out_jsonl gpt_input_batch.jsonl \
    --out_report report.json \
    --tau_utility 0.6 \
    --lambda_suppress 12.0

Input JSONL format:
  Each line is a JSON object containing a question string in one of these keys:
    - "question", "prompt", "text", "input", "query"
  (You can override with --question_key.)

Output JSONL format:
  OpenAI Batch format lines:
    {"custom_id": "...", "method": "POST", "url": "/v1/chat/completions", "body": {...}}

Notes:
- This is a reference implementation; local generation settings can be tuned via CLI.
- For reproducibility, you can also emit a local report with the 3 generated prompts.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, Iterable, List, Optional, Set

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessorList,
    set_seed,
)

# Import from your existing privacy code (main_fixed.py).
# Assumes generate_batch_jsonl.py is in the same directory as main_fixed.py
from main_fixed import (
    PrivacyOntology,
    PhiSpanMapper,
    RqProviderWithGreedyFixed,
    StrictPrivacyLogitsProcessor,
)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no} in {path}: {e}") from e
    return rows


def write_jsonl(path: str, items: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def extract_question(obj: Dict[str, Any], preferred_key: Optional[str] = None) -> List[str]:
    if preferred_key:
        if preferred_key not in obj or not isinstance(obj[preferred_key], str):
            raise KeyError(f'question_key="{preferred_key}" not found (or not str) in: {obj}')
        return obj[preferred_key].strip()

    for k in ("question", "prompt", "text", "input", "query"):
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()

    raise KeyError(
        "Could not find a question string in keys "
        "['question','prompt','text','input','query']: "
        f"{obj}"
    )


def resolve_device(choice: str) -> torch.device:
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def generate_local(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    logits_processor: Optional[LogitsProcessorList] = None,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    out_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        logits_processor=logits_processor,
        pad_token_id=tokenizer.pad_token_id,
    )

    decoded = tokenizer.decode(out_ids[0], skip_special_tokens=True)

    # If the model echoes the prompt, strip it
    if decoded.startswith(prompt):
        decoded = decoded[len(prompt):].lstrip()

    return decoded.strip()


def build_default_ontology() -> PrivacyOntology:
    return PrivacyOntology(
        granularities={
            "identifier": ["none", "partial", "exact"],
            "contact": ["none", "domain_only", "exact"],
            "location": ["none", "country", "city", "street_or_lower"],
            "financial": ["none", "bucket", "exact"],
            "age": ["none", "decade", "exact"],
            "occupation": ["none", "sector", "job_family", "exact_role"],
            "affiliation": ["none", "industry", "org_type", "exact_org"],
        }
    )


def build_chat_completions_request(
    custom_id: str,
    cloud_model: str,
    system_prompt: str,
    user_content: str,
    temperature: float,
    max_tokens: int,
) -> Dict[str, Any]:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": cloud_model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        },
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--question_jsonl", required=True, help="Input questions JSONL path")
    p.add_argument("--question_key", default=None, help="Optional explicit key name for question string")
    p.add_argument("--out_jsonl", default="gpt_input_batch.jsonl", help="Output Batch JSONL path")
    p.add_argument("--out_report", default=None, help="Optional JSON report of generated prompts")

    p.add_argument("--local_model", required=True, help="Local HF model name/path, e.g., meta-llama/Llama-3.2-1B")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--local_max_new_tokens", type=int, default=96)
    p.add_argument("--local_temperature", type=float, default=0.7)
    p.add_argument("--local_top_p", type=float, default=0.9)

    p.add_argument("--tau_utility", type=float, default=0.6)
    p.add_argument("--lambda_suppress", type=float, default=12.0)
    p.add_argument("--hard_mask", action="store_true")
    p.add_argument("--window_chars", type=int, default=160)

    p.add_argument("--cloud_model", default="gpt-4o-mini")
    p.add_argument("--cloud_temperature", type=float, default=0.2)
    p.add_argument("--cloud_max_tokens", type=int, default=400)
    p.add_argument(
        "--system_prompt",
        default="You are a helpful assistant. Answer the user's question clearly and correctly.",
    )

    p.add_argument(
        "--variants",
        default="passthrough,local_baseline,local_privacy",
        help="Comma-separated: passthrough,local_baseline,local_privacy",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    rows = read_jsonl(args.question_jsonl)

    tokenizer = AutoTokenizer.from_pretrained(args.local_model)
    model = AutoModelForCausalLM.from_pretrained(args.local_model).to(device)
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build privacy processor
    ontology = build_default_ontology()
    rq_provider = RqProviderWithGreedyFixed(
        ontology=ontology,
        tau_utility=args.tau_utility,
        max_steps=16,
        cache=True,
    )
    phi_mapper = PhiSpanMapper(tokenizer=tokenizer, window_chars=args.window_chars)
    privacy_proc = StrictPrivacyLogitsProcessor(
        tokenizer=tokenizer,
        rq_provider=rq_provider,
        phi_mapper=phi_mapper,
        lambda_suppress=args.lambda_suppress,
        hard_mask=args.hard_mask,
    )
    privacy_lp = LogitsProcessorList([privacy_proc])

    want: Set[str] = {v.strip() for v in args.variants.split(",") if v.strip()}
    valid = {"passthrough", "local_baseline", "local_privacy"}
    unknown = want - valid
    if unknown:
        raise ValueError(f"Unknown variants: {sorted(unknown)}. Valid: {sorted(valid)}")

    batch_lines: List[Dict[str, Any]] = []
    report: Dict[str, Any] = {"input": args.question_jsonl, "local_model": args.local_model, "rows": []}

    for i, obj in enumerate(rows):
        q = extract_question(obj, args.question_key)

        baseline_rewrite = None
        privacy_rewrite = None

        if "local_baseline" in want:
            baseline_rewrite = generate_local(
                model=model,
                tokenizer=tokenizer,
                prompt=q,
                device=device,
                max_new_tokens=args.local_max_new_tokens,
                temperature=args.local_temperature,
                top_p=args.local_top_p,
                logits_processor=None,
            )

        if "local_privacy" in want:
            privacy_rewrite = generate_local(
                model=model,
                tokenizer=tokenizer,
                prompt=q,
                device=device,
                max_new_tokens=args.local_max_new_tokens,
                temperature=args.local_temperature,
                top_p=args.local_top_p,
                logits_processor=privacy_lp,
            )

        if "passthrough" in want:
            batch_lines.append(
                build_chat_completions_request(
                    custom_id=f"q{i:06d}_passthrough",
                    cloud_model=args.cloud_model,
                    system_prompt=args.system_prompt,
                    user_content=q,
                    temperature=args.cloud_temperature,
                    max_tokens=args.cloud_max_tokens,
                )
            )

        if "local_baseline" in want:
            batch_lines.append(
                build_chat_completions_request(
                    custom_id=f"q{i:06d}_local_baseline",
                    cloud_model=args.cloud_model,
                    system_prompt=args.system_prompt,
                    user_content=baseline_rewrite or "",
                    temperature=args.cloud_temperature,
                    max_tokens=args.cloud_max_tokens,
                )
            )

        if "local_privacy" in want:
            batch_lines.append(
                build_chat_completions_request(
                    custom_id=f"q{i:06d}_local_privacy",
                    cloud_model=args.cloud_model,
                    system_prompt=args.system_prompt,
                    user_content=privacy_rewrite or "",
                    temperature=args.cloud_temperature,
                    max_tokens=args.cloud_max_tokens,
                )
            )

        report["rows"].append(
            {
                "idx": i,
                "raw_question": q,
                "passthrough": q,
                "local_baseline": baseline_rewrite,
                "local_privacy": privacy_rewrite,
            }
        )

    write_jsonl(args.out_jsonl, batch_lines)

    if args.out_report:
        with open(args.out_report, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Wrote Batch JSONL: {args.out_jsonl}  (requests={len(batch_lines)})")
    if args.out_report:
        print(f"Wrote report: {args.out_report}")


if __name__ == "__main__":
    main()
