#!/usr/bin/env python3

import argparse
import json
import logging
from typing import Any, Dict, Iterable, List, Optional

import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList, set_seed

from main_fixed import (
    PrivacyOntology,
    PhiSpanMapper,
    RqProviderWithGreedyFixed,
    StrictPrivacyLogitsProcessor,
)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ========== IO ==========
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


def write_jsonl(path: str, items: Iterable[Dict[str, Any]]) -> None:
    """JSONL 파일 쓰기"""
    with open(path, "w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# ========== EXTRACTION ==========
def extract_questions_from_batch_output(obj: Dict[str, Any]) -> List[str]:
    """배치 출력에서 질문 추출"""
    try:
        content = obj["response"]["body"]["choices"][0]["message"]["content"]
        if not isinstance(content, str) or not content.strip():
            return []
        parsed = json.loads(content)
        qs = parsed.get("questions", [])
        if not isinstance(qs, list):
            return []
        out = [q.strip() for q in qs if isinstance(q, str) and q.strip()]
        return out
    except (KeyError, json.JSONDecodeError, TypeError):
        return []


def flatten_questions(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """배치 출력을 평탄화하여 질문 리스트 생성"""
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


# ========== OLLAMA INTEGRATION ==========
def generate_with_ollama(
    question: str,
    model: str = "mistral",
    temperature: float = 0.7,
    top_p: float = 0.9,
    base_url: str = "http://localhost:11434",
    timeout: int = 60,
) -> str:
    """
    Ollama를 사용하여 질문 리라이팅
    System/User 역할 분리로 정확한 프롬프트 구조 사용
    """
    system_prompt = """You are a query rewriter for a cloud LLM.
Your task: Rewrite the user's question into a clean, self-contained question.

Rules:
- Output ONLY the rewritten question. No preface, no explanation.
- Preserve the user's intent and any constraints.
- Do NOT invent personal details or numbers.
- Do NOT create new questions; only refine the given question."""

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question.strip()},
        ],
        "temperature": temperature,
        "top_p": top_p,
        "stream": False,
    }

    try:
        response = requests.post(
            f"{base_url}/api/chat",
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        result = response.json()
        return result["message"]["content"].strip()
    except requests.exceptions.ConnectionError:
        logger.error(f"Ollama 서버에 연결할 수 없습니다: {base_url}")
        raise RuntimeError(
            f"Ollama 서버가 실행 중이 아닙니다.\n"
            f"실행: ollama serve\n"
            f"그리고: ollama pull {model}"
        )
    except Exception as e:
        logger.error(f"Ollama 생성 오류: {e}")
        raise


def check_ollama_health(base_url: str = "http://localhost:11434") -> bool:
    """Ollama 서버 상태 확인"""
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


# ========== LOCAL MODEL GENERATION ==========
def resolve_device(choice: str) -> torch.device:
    """디바이스 선택"""
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
    """로컬 모델을 사용한 텍스트 생성 (프라이버시 처리 포함)"""
    system_prompt = """You are a query rewriter for a cloud LLM.
Your task: Rewrite the user's question into a clean, self-contained question.

Rules:
- Output ONLY the rewritten question. No preface, no explanation.
- Preserve the user's intent and any constraints.
- Do NOT invent personal details or numbers.
- Do NOT create new questions; only refine the given question."""

    final_prompt = system_prompt + "\n" + prompt.strip()
    inputs = tokenizer(final_prompt, return_tensors="pt").to(device)

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


# ========== PRIVACY ==========
def build_default_ontology() -> PrivacyOntology:
    """기본 프라이버시 온톨로지 구성"""
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


# ========== BATCH OUTPUT ==========
def build_batch_input_line(
    custom_id: str,
    cloud_model: str,
    system_prompt: str,
    user_content: str,
    temperature: float,
    max_tokens: int,
) -> Dict[str, Any]:
    """OpenAI 배치 입력 라인 구성"""
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


# ========== ARGUMENT PARSING ==========
def parse_args() -> argparse.Namespace:
    """명령어 인자 파싱"""
    p = argparse.ArgumentParser(
        description="Ollama 기반 Query Rewriter (System/User 프롬프트 구조)"
    )

    # 입출력
    p.add_argument("--question_jsonl", required=True, help="배치 출력 JSONL 파일")
    p.add_argument("--out_jsonl", default="gpt_input_batch.jsonl", help="출력 JSONL 파일")
    p.add_argument("--out_report", default="local_prompts_report.json", help="리포트 JSON 파일")

    # Ollama 설정
    p.add_argument(
        "--ollama_model",
        default="mistral",
        help="Ollama 모델 (mistral, llama2, neural-chat 등)"
    )
    p.add_argument(
        "--ollama_base_url",
        default="http://localhost:11434",
        help="Ollama 서버 URL"
    )

    # 프라이버시 로짓 프로세서
    p.add_argument("--tau_utility", type=float, default=0.6)
    p.add_argument("--lambda_suppress", type=float, default=12.0)
    p.add_argument("--hard_mask", action="store_true")
    p.add_argument("--window_chars", type=int, default=160)

    # 클라우드 요청
    p.add_argument("--cloud_model", default="gpt-4o-mini")
    p.add_argument("--cloud_temperature", type=float, default=0.2)
    p.add_argument("--cloud_max_tokens", type=int, default=400)
    p.add_argument(
        "--system_prompt",
        default="You are a helpful assistant."
    )

    return p.parse_args()


# ========== MAIN ==========
def main() -> None:
    """메인 실행 함수"""
    args = parse_args()
    set_seed(args.seed)

    logger.info("=" * 60)
    logger.info("Query Rewriter 시작")
    logger.info("=" * 60)

    # JSONL 읽기
    rows = read_jsonl(args.question_jsonl)
    logger.info(f"✓ {len(rows)}개 라인 읽음: {args.question_jsonl}")

    items = flatten_questions(rows)
    if not items:
        raise RuntimeError("배치 출력 JSONL에서 질문을 추출할 수 없습니다.")
    logger.info(f"✓ {len(items)}개 질문 추출")

    if not check_ollama_health(args.ollama_base_url):
        raise RuntimeError(
            f"Ollama 서버에 연결할 수 없습니다.\n"
            f"실행하세요:\n"
            f"  1) ollama serve\n"
            f"  2) ollama pull {args.ollama_model}"
        )

    batch_lines: List[Dict[str, Any]] = []
    report_rows: List[Dict[str, Any]] = []

        # 파일 초기화
    open(args.out_jsonl, 'w').close()

    for idx, it in enumerate(items, start=1):
        qid = it["qid"]
        q = it["question"]

        logger.info(f"\n[{idx}/{len(items)}] qid={qid}")
        logger.info(f"원본: {q[:80]}...")

        try:
            rewritten = generate_with_ollama(
                question=q,
                model=args.ollama_model,
                temperature=args.local_temperature,
                top_p=args.local_top_p,
                base_url=args.ollama_base_url,
            )
            logger.info(f"리라이팅: {rewritten[:80]}...")

                # 배치 입력 라인 생성
            batch_line = build_batch_input_line(
                custom_id=f"q{qid:06d}_rewritten",
                cloud_model=args.cloud_model,
                system_prompt=args.system_prompt,
                user_content=rewritten,
                temperature=args.cloud_temperature,
                max_tokens=args.cloud_max_tokens,
            )
            batch_lines.append(batch_line)

                # 리포트 행
            report_rows.append(
                {
                    **it,
                    "rewritten": rewritten,
                }
            )

            # 임시 저장
            with open(args.out_jsonl, 'a', encoding='utf-8') as f:
                f.write(json.dumps(batch_line, ensure_ascii=False) + "\n")

        except Exception as e:
            logger.error(f"오류 (qid={qid}): {e}")
            continue

    # 리포트 저장
    with open(args.out_report, "w", encoding="utf-8") as f:
        json.dump(
            {
                "mode": "ollama",
                "input_file": args.question_jsonl,
                "n_questions": len(items),
                "n_batch_requests": len(batch_lines),
                "cloud_model": args.cloud_model,
                "local_model": args.local_model if not args.use_ollama else args.ollama_model,
                "parameters": {
                    "tau_utility": args.tau_utility,
                    "lambda_suppress": args.lambda_suppress,
                    "temperature": args.local_temperature,
                    "top_p": args.local_top_p,
                },
                "questions": report_rows,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    logger.info("\n" + "=" * 60)
    logger.info(f"✓ {args.out_jsonl} 저장 (요청={len(batch_lines)})")
    logger.info(f"✓ {args.out_report} 저장 (질문={len(items)})")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
