#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from typing import Any, Dict, Iterable, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList, set_seed

from main_fixed import (
    PrivacyOntology,
    PhiSpanMapper,
    RqProviderWithGreedyFixed,
    StrictPrivacyLogitsProcessor,
)


# ---------- IO ----------
def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no}: {e}") from e
            rows.append(obj)
    return rows


def write_jsonl(path: str, items: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# ---------- Batch OUTPUT -> questions ----------
def extract_questions_from_batch_output(obj: Dict[str, Any]) -> List[str]:
    """
    Your input format:
      obj["response"]["body"]["choices"][0]["message"]["content"] == JSON string
      parsed["questions"] == list[str]
    """
    content = obj["response"]["body"]["choices"][0]["message"]["content"]
    if not isinstance(content, str) or not content.strip():
        return []
    parsed = json.loads(content)  # must be JSON string
    qs = parsed.get("questions", [])
    if not isinstance(qs, list):
        return []
    out = [q.strip() for q in qs if isinstance(q, str) and q.strip()]
    return out


def flatten_questions(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Returns list of:
      { "qid": int, "source_line": int, "source_custom_id": str|None, "question": str }
    """
    out: List[Dict[str, Any]] = []
    qid = 0
    for line_no, obj in enumerate(rows, start=1):
        custom_id = obj.get("custom_id")
        qs = extract_questions_from_batch_output(obj)
        for j, q in enumerate(qs):
            out.append(
                {
                    "qid": qid,
                    "source_line": line_no,
                    "source_custom_id": custom_id,
                    "question_idx_in_line": j,
                    "question": q,
                }
            )
            qid += 1
    return out


# ---------- Local generation ----------
def resolve_device(choice: str) -> torch.device:
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def generate_local_text(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
    logits_processor: Optional[LogitsProcessorList],
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    out_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
        logits_processor=logits_processor,
        pad_token_id=tokenizer.pad_token_id,
    )

    decoded = tokenizer.decode(out_ids[0], skip_special_tokens=True)
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


# ---------- OpenAI Batch INPUT line ----------
def build_batch_input_line(
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
    p.add_argument("--question_jsonl", required=True)  # batch output jsonl
    p.add_argument("--local_model", required=True)
    p.add_argument("--cloud_model", default="gpt-4o-mini")
    p.add_argument("--out_jsonl", default="gpt_input_batch.jsonl")
    p.add_argument("--out_report", default="local_prompts_report.json")

    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--seed", type=int, default=7)

    # local generation
    p.add_argument("--local_max_new_tokens", type=int, default=96)
    p.add_argument("--deterministic", action="store_true", help="force do_sample=False (recommended for verification)")
    p.add_argument("--local_temperature", type=float, default=0.7)
    p.add_argument("--local_top_p", type=float, default=0.9)

    # privacy logits
    p.add_argument("--tau_utility", type=float, default=0.6)
    p.add_argument("--lambda_suppress", type=float, default=12.0)
    p.add_argument("--hard_mask", action="store_true")
    p.add_argument("--window_chars", type=int, default=160)

    # cloud request
    p.add_argument("--cloud_temperature", type=float, default=0.2)
    p.add_argument("--cloud_max_tokens", type=int, default=400)
    p.add_argument("--system_prompt", default="You are a helpful assistant.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    rows = read_jsonl(args.question_jsonl)
    items = flatten_questions(rows)
    if not items:
        raise RuntimeError("No questions extracted from batch output JSONL.")

    tokenizer = AutoTokenizer.from_pretrained(args.local_model)
    model = AutoModelForCausalLM.from_pretrained(args.local_model).to(device)
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # privacy processor
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

    do_sample = False if args.deterministic else True

    batch_lines: List[Dict[str, Any]] = []
    report_rows: List[Dict[str, Any]] = []

    for it in items:
        qid = it["qid"]
        q = it["question"]

        # 1) passthrough local output = identity
        passthrough_prompt = q

        # 2) baseline local output
        baseline_prompt = generate_local_text(
            model=model,
            tokenizer=tokenizer,
            prompt=q,
            device=device,
            max_new_tokens=args.local_max_new_tokens,
            temperature=args.local_temperature,
            top_p=args.local_top_p,
            do_sample=do_sample,
            logits_processor=None,
        )

        # 3) logit local output
        logit_prompt = generate_local_text(
            model=model,
            tokenizer=tokenizer,
            prompt=q,
            device=device,
            max_new_tokens=args.local_max_new_tokens,
            temperature=args.local_temperature,
            top_p=args.local_top_p,
            do_sample=do_sample,
            logits_processor=privacy_lp,
        )

        # build cloud batch input lines (3 variants)
        batch_lines.append(
            build_batch_input_line(
                custom_id=f"q{qid:06d}_passthrough",
                cloud_model=args.cloud_model,
                system_prompt=args.system_prompt,
                user_content=passthrough_prompt,
                temperature=args.cloud_temperature,
                max_tokens=args.cloud_max_tokens,
            )
        )
        batch_lines.append(
            build_batch_input_line(
                custom_id=f"q{qid:06d}_baseline",
                cloud_model=args.cloud_model,
                system_prompt=args.system_prompt,
                user_content=baseline_prompt,
                temperature=args.cloud_temperature,
                max_tokens=args.cloud_max_tokens,
            )
        )
        batch_lines.append(
            build_batch_input_line(
                custom_id=f"q{qid:06d}_logit",
                cloud_model=args.cloud_model,
                system_prompt=args.system_prompt,
                user_content=logit_prompt,
                temperature=args.cloud_temperature,
                max_tokens=args.cloud_max_tokens,
            )
        )

        report_rows.append(
            {
                **it,
                "passthrough": passthrough_prompt,
                "baseline": baseline_prompt,
                "logit": logit_prompt,
            }
        )

    write_jsonl(args.out_jsonl, batch_lines)
    with open(args.out_report, "w", encoding="utf-8") as f:
        json.dump(
            {
                "input": args.question_jsonl,
                "n_questions": len(items),
                "n_requests": len(batch_lines),
                "local_model": args.local_model,
                "cloud_model": args.cloud_model,
                "deterministic": args.deterministic,
                "tau_utility": args.tau_utility,
                "lambda_suppress": args.lambda_suppress,
                "rows": report_rows,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Wrote {args.out_jsonl} (requests={len(batch_lines)})")
    print(f"Wrote {args.out_report} (questions={len(items)})")


if __name__ == "__main__":
    main()
