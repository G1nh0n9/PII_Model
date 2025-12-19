#!/usr/bin/env python3
"""
main_fixed.py — Fixed version addressing 3 critical misalignments:
1. Query-adaptive utility: utility_of now depends on query_text
2. Attribute-specific cost differentiation: base_costs vary by attribute
3. Cache key bug: now uses (batch_idx, prompt_text) tuple

All fixes maintain strict formula equivalence:
  z'(t) = z(t) - λ * m_R(q)(t)
  m_R(q)(t) = 0 if φ(t) ∈ R(q) or φ(t) = ∅
            = 1 otherwise
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
    PreTrainedTokenizerBase,
    set_seed,
)

# ============================================================================
# Types / data structures
# ============================================================================
Attribute = str
Granularity = str
AttrGran = Tuple[Attribute, Granularity]


@dataclass(frozen=True)
class PrivacyOntology:
    """Attribute -> ordered granularities (coarse -> fine)."""

    granularities: Dict[Attribute, List[Granularity]]

    def all_pairs(self) -> List[AttrGran]:
        return [(a, g) for a, gs in self.granularities.items() for g in gs]


@dataclass(frozen=True)
class RqResult:
    """Policy R(q): allowed attribute–granularity pairs."""

    allowed: Set[AttrGran]


# ============================================================================
# FIX 1: Query-adaptive Utility
# ============================================================================
class QueryAwareUtilityEstimator:
    """
    Extracts attributes detected in query_text, then measures utility
    as overlap between selected and detected attributes.
    
    This makes utility_of genuinely depend on query_text (not just parameter).
    """
    
    def __init__(self):
        # Markers for detection
        self.location_markers = ["서울", "강남", "부산", "광주", "대구", "인천", "대전", 
                                 "Seoul", "Busan", "Gangnam", "New York", "LA", "Manhattan"]
        self.age_patterns = [r"\d{1,3}세", r"(\d{1,3})\s*년생", r"\d{1,3}\s*years?\s*old"]
        self.contact_patterns = [r"@", r"\d{2,4}[-\s]\d{3,4}[-\s]\d{4}"]
        self.medical_keywords = ["의료", "진단", "치료", "병원", "약", "증상", 
                                 "medical", "diagnosis", "treatment", "disease"]
        self.financial_patterns = [r"[\$₩€]\s?\d+", r"\d+\s?(USD|KRW|EUR|만원|원)"]
    
    def detect_attributes_in_query(self, query_text: str) -> Set[AttrGran]:
        """
        Detects which attributes are mentioned in query_text.
        Returns set of (attribute, granularity) pairs actually present.
        """
        detected: Set[AttrGran] = set()
        
        # Location detection
        if any(loc in query_text for loc in self.location_markers):
            detected.add(("location", "city"))
        
        # Age detection
        for pattern in self.age_patterns:
            if re.search(pattern, query_text):
                detected.add(("age", "exact"))
                break
        
        # Contact detection
        for pattern in self.contact_patterns:
            if re.search(pattern, query_text):
                detected.add(("contact", "exact"))
                break
        
        # Financial detection
        for pattern in self.financial_patterns:
            if re.search(pattern, query_text):
                detected.add(("financial", "exact"))
                break
        
        # Medical relevance (for appropriate granularity selection)
        has_medical = any(kw in query_text.lower() for kw in self.medical_keywords)
        
        return detected, has_medical
    
    def estimate_utility(self, selected: Set[AttrGran], query_text: str) -> float:
        """
        Utility = (overlap of selected with detected) * medical_relevance
        
        Now genuinely depends on query_text (FIX 1).
        """
        detected, has_medical = self.detect_attributes_in_query(query_text)
        
        if not detected:
            return 0.0
        
        # Overlap ratio
        overlap = len(selected & detected)
        overlap_ratio = overlap / len(detected)
        
        # Medical relevance multiplier
        medical_mult = 1.2 if has_medical else 0.8
        
        return min(1.0, overlap_ratio * medical_mult)


# ============================================================================
# FIX 2 & 3: Attribute-specific Cost + Fixed Cache Key
# ============================================================================
def compute_rq_greedy(
    query_text: str,
    ontology: PrivacyOntology,
    tau_utility: float,
    utility_estimator: QueryAwareUtilityEstimator,
    max_steps: int = 32,
) -> RqResult:
    """
    Greedy solution to:
        minimize sum c(pair)  s.t. U(selected, query_text) >= tau_utility

    FIX 2: cost_of now differentiates by attribute type.
    """
    
    # FIX 2: Attribute-specific base costs
    ATTR_BASE_COSTS = {
        "identifier": 10.0,    # Most sensitive
        "contact": 8.0,
        "financial": 5.0,
        "location": 3.0,
        "age": 2.0,
        "occupation": 1.5,     # Gray-zone
        "affiliation": 1.5,
    }
    
    def cost_of(pair: AttrGran) -> float:
        """Cost differentiated by attribute (FIX 2)."""
        a, g = pair
        
        # Base cost by attribute (FIX 2: not all attributes equal)
        base = ATTR_BASE_COSTS.get(a, 1.5)
        
        # Granularity multiplier (finer = higher cost)
        gs = ontology.granularities[a]
        rank = gs.index(g)
        granularity_mult = 1.0 + 0.3 * rank
        
        return base * granularity_mult
    
    candidates = ontology.all_pairs()
    selected: Set[AttrGran] = set()

    for _ in range(max_steps):
        # Utility depends on query_text (FIX 1)
        current_u = utility_estimator.estimate_utility(selected, query_text)
        if current_u >= tau_utility:
            break

        best_pair: Optional[AttrGran] = None
        best_score: float = float("-inf")

        for pair in candidates:
            if pair in selected:
                continue

            before = current_u
            after = utility_estimator.estimate_utility(selected | {pair}, query_text)
            gain = after - before
            cost = cost_of(pair)
            score = gain / max(cost, 1e-12)

            if score > best_score:
                best_score = score
                best_pair = pair

        if best_pair is None:
            break

        selected.add(best_pair)

    return RqResult(allowed=selected)


class RqProviderWithGreedyFixed:
    """
    FIX 3: Cache key is now (batch_idx, prompt_text) tuple (not just batch_idx).
    This prevents collision when different prompts share same batch_idx.
    """

    def __init__(
        self,
        ontology: PrivacyOntology,
        tau_utility: float = 0.6,
        max_steps: int = 16,
        cache: bool = True,
    ):
        self.ontology = ontology
        self.tau_utility = float(tau_utility)
        self.max_steps = int(max_steps)
        self.cache = bool(cache)
        self.utility_estimator = QueryAwareUtilityEstimator()
        
        # FIX 3: Cache key is now (batch_idx, prompt_text) tuple
        self._cache: Dict[Tuple[int, str], RqResult] = {}

    def __call__(self, batch_idx: int, prompt_text: str) -> RqResult:
        # FIX 3: Use (batch_idx, prompt_text) as cache key
        cache_key = (batch_idx, prompt_text)
        
        if self.cache and cache_key in self._cache:
            return self._cache[cache_key]

        rq = compute_rq_greedy(
            query_text=prompt_text,
            ontology=self.ontology,
            tau_utility=self.tau_utility,
            utility_estimator=self.utility_estimator,
            max_steps=self.max_steps,
        )
        if self.cache:
            self._cache[cache_key] = rq
        return rq


# ============================================================================
# φ(t): span-aware mapping (unchanged, but kept for completeness)
# ============================================================================
class PhiSpanMapper:
    """
    φ(t): maps candidate token (given prefix) to Set[AttrGran].
    Heuristic-based; replace with actual NER/regex logic as needed.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, window_chars: int = 160):
        self.tokenizer = tokenizer
        self.window_chars = int(window_chars)

        self.email_re = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
        self.long_id_re = re.compile(r"\b[A-Za-z0-9]{10,}\b")
        self.money_re = re.compile(r"(\$|₩|€)\s?\d+|(\d+)\s?(USD|KRW|EUR|만원|원)")
        self.phone_re = re.compile(r"(\+?\d{1,3}[-\s]?)?\d{2,4}[-\s]?\d{3,4}[-\s]?\d{4}")
        self.addr_markers = ["동", "로", "길", "번지", "아파트", "구", "시", "도", 
                             "Street", "St.", "Ave", "Road", "Rd"]

    def phi(self, prefix_text: str, candidate_token_id: int) -> Set[AttrGran]:
        tok = self.tokenizer.decode([candidate_token_id], skip_special_tokens=True)
        suffix = prefix_text[-self.window_chars :] if len(prefix_text) > self.window_chars else prefix_text
        span = suffix + tok

        out: Set[AttrGran] = set()

        if self.email_re.search(span):
            out.add(("contact", "exact"))
        if self.phone_re.search(span):
            out.add(("contact", "exact"))
        if any(m in span for m in self.addr_markers):
            out.add(("location", "street_or_lower"))
        if self.money_re.search(span) or any(ch.isdigit() for ch in tok):
            out.add(("financial", "exact"))
        if any(ch.isdigit() for ch in tok):
            out.add(("age", "exact"))
        if self.long_id_re.search(span):
            out.add(("identifier", "exact"))

        return out


# ============================================================================
# Strictly equivalent LogitsProcessor (unchanged structure)
# ============================================================================
class StrictPrivacyLogitsProcessor(LogitsProcessor):
    """
    EXACT implementation of:
      z'(t) = z(t) - λ * m_R(q)(t)

    m_R(q)(t)=1 iff (∃ mapped pair not allowed by R(q))
    m_R(q)(t)=0 iff φ(t)=∅ OR all mapped pairs are allowed by R(q)
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        rq_provider: Callable[[int, str], RqResult],
        phi_mapper: PhiSpanMapper,
        lambda_suppress: float = 12.0,
        hard_mask: bool = False,
    ):
        self.tokenizer = tokenizer
        self.rq_provider = rq_provider
        self.phi_mapper = phi_mapper
        self.lambda_suppress = float(lambda_suppress)
        self.hard_mask = bool(hard_mask)

    @torch.no_grad()
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        B, V = scores.shape

        for b in range(B):
            prefix_text = self.tokenizer.decode(input_ids[b], skip_special_tokens=True)
            rq = self.rq_provider(b, prefix_text)
            allowed = rq.allowed

            for token_id in range(V):
                mapped = self.phi_mapper.phi(prefix_text, token_id)
                if not mapped:
                    continue

                forbidden = any(pair not in allowed for pair in mapped)
                if forbidden:
                    if self.hard_mask:
                        scores[b, token_id] = float("-inf")
                    else:
                        scores[b, token_id] = scores[b, token_id] - self.lambda_suppress

        return scores


# ============================================================================
# Main
# ============================================================================
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="gpt2", help="HF model name or local path")
    p.add_argument("--prompt", type=str, required=True, help="Prompt to generate from")
    p.add_argument("--max_new_tokens", type=int, default=120)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--tau_utility", type=float, default=0.6)
    p.add_argument("--lambda_suppress", type=float, default=12.0)
    p.add_argument("--hard_mask", action="store_true", help="If set, forbidden tokens get -inf logits")
    p.add_argument("--window_chars", type=int, default=160)
    return p.parse_args()


def resolve_device(choice: str) -> torch.device:
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = resolve_device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    ontology = build_default_ontology()

    # FIX 1, 2, 3 all integrated in this provider
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

    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)

    out_ids = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        logits_processor=LogitsProcessorList([privacy_proc]),
        pad_token_id=tokenizer.pad_token_id,
    )

    print(tokenizer.decode(out_ids[0], skip_special_tokens=True))


if __name__ == "__main__":
    # sample clis
    # --model meta-llama/Llama-3.2-1B --prompt "Patient John Doe, a 45-year-old from Seoul, was diagnosed with diabetes."
    main()
