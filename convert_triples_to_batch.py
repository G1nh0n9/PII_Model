#!/usr/bin/env python3
"""
convert_triples_to_batch.py

triples_out.jsonl의 각 항목에서:
- question_raw → passthrough
- baseline_answer → baseline
- logit_answer → logit

각각을 OpenAI Batch API 입력 형태로 변환합니다.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """JSONL 파일 읽기"""
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


def write_jsonl(path: str, items: List[Dict[str, Any]]) -> None:
    """JSONL 파일 쓰기"""
    with open(path, "w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def build_batch_request(
    custom_id: str,
    model: str,
    system_prompt: str,
    user_content: str,
    temperature: float = 0.2,
    max_tokens: int = 400,
) -> Dict[str, Any]:
    """OpenAI Batch API 요청 형태 생성"""
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
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
    p.add_argument("--input", default="triples_out.jsonl", help="입력 JSONL 파일 경로")
    p.add_argument("--output", default="batch_input.jsonl", help="출력 Batch JSONL 파일 경로")
    p.add_argument("--model", default="gpt-4o-mini", help="사용할 클라우드 모델")
    p.add_argument("--system_prompt", default="You are a helpful assistant.", help="시스템 프롬프트")
    p.add_argument("--temperature", type=float, default=0.2, help="Temperature 설정")
    p.add_argument("--max_tokens", type=int, default=400, help="최대 토큰 수")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    
    print(f"입력 파일 읽는 중: {args.input}")
    rows = read_jsonl(args.input)
    print(f"총 {len(rows)}개의 항목을 읽었습니다.")
    
    batch_requests: List[Dict[str, Any]] = []
    
    for idx, item in enumerate(rows):
        qid = item.get("qid", f"unknown_{idx}")
        
        # 각 항목에서 3개의 필드 추출
        question_raw = item.get("question_raw", "").strip()
        baseline_answer = item.get("baseline_answer", "").strip()
        logit_answer = item.get("logit_answer", "").strip()
        
        # 1) passthrough (question_raw)
        if question_raw:
            batch_requests.append(
                build_batch_request(
                    custom_id=f"{qid}_passthrough",
                    model=args.model,
                    system_prompt=args.system_prompt,
                    user_content=question_raw,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
            )
        
        # 2) baseline (baseline_answer)
        if baseline_answer:
            batch_requests.append(
                build_batch_request(
                    custom_id=f"{qid}_baseline",
                    model=args.model,
                    system_prompt=args.system_prompt,
                    user_content=baseline_answer,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
            )
        
        # 3) logit (logit_answer)
        if logit_answer:
            batch_requests.append(
                build_batch_request(
                    custom_id=f"{qid}_logit",
                    model=args.model,
                    system_prompt=args.system_prompt,
                    user_content=logit_answer,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
            )
        
        if (idx + 1) % 100 == 0:
            print(f"  처리 중: {idx + 1}/{len(rows)}")
    
    print(f"\n총 {len(batch_requests)}개의 배치 요청 생성됨 (각 항목당 3개)")
    
    write_jsonl(args.output, batch_requests)
    print(f"✓ 출력 파일 저장 완료: {args.output}")


if __name__ == "__main__":
    main()
