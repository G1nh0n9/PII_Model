# coding: utf-8
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any,Dict, List, Optional, Tuple, Iterable

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoTokenizer, AutoModel

def read_jsonl_questions(path: str) -> Iterable[Dict[str, Any]]:
    """Expect each line to contain at least {'qid': int/str, 'question': str}."""
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)

            resp = row.get("response")
            if not resp or resp.get("status_code") != 200:
                continue

            try:
                content = (
                    resp["body"]["choices"][0]["message"]["content"]
                )
            except (KeyError, IndexError):
                continue

            # content 자체가 JSON 문자열임
            try:
                payload = json.loads(content)
            except json.JSONDecodeError:
                continue

            questions = payload.get("questions", [])
            idx = payload.get("idx")

            for i, q in enumerate(questions):
                yield {
                    "batch_id": row.get("id"),
                    "custom_id": row.get("custom_id"),
                    "idx": idx,
                    "qid": f"{row.get('custom_id')}_{i}",
                    "question": q,
                    "source_line": line_no,
                }

def append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

# -----------------------------
# Ontology / lattices / policy
# -----------------------------
@dataclass
class RiskState:
    C: float = 0.0          # accumulated disclosure
    step: int = 0

@dataclass(frozen=True)
class Lattice:
    # least revealing -> most revealing
    levels: List[str]

    def idx(self, g: str) -> int:
        return self.levels.index(g)

    def succ(self, g: str) -> str:
        i = self.idx(g)
        return self.levels[min(i + 1, len(self.levels) - 1)]


@dataclass
class Policy:
    max_resolution: Dict[str, str]  # R(a)=g

    def __call__(self, a: str) -> str:
        return self.max_resolution[a]


# -----------------------------
# gen_a and gran induced by gen
# -----------------------------

class Generalizer:
    def __init__(self, lattices: Dict[str, Lattice]) -> None:
        self.lattices = lattices

    def gen(self, a: str, v: str, g: str) -> str:
        g = g.lower()
        if g == "none":
            return ""  # suppressed upstream

        if a == "contact":
            if g == "masked":
                if "@" in v:
                    user, domain = v.split("@", 1)
                    return user[:1] + "***@" + domain
                return re.sub(r"\d", "*", v)
            if g == "full":
                return v

        if a == "identifier":
            if g == "prefix":
                return v[:3] + "***"
            if g == "full":
                return v

        if a == "location":
            toks = v.split()
            if g == "city":
                return toks[0] if toks else v
            if g == "region":
                return " ".join(toks[:2]) if len(toks) >= 2 else v
            if g == "street":
                return v

        if a == "age":
            m = re.search(r"\d{1,3}", v)
            if not m:
                return v
            age = int(m.group(0))
            if g == "decade":
                return f"{(age // 10) * 10}s"
            if g == "exact":
                return str(age)

        if a == "financial":
            amt = _try_parse_amount(v)
            if amt is None:
                return v
            if g == "bucket":
                return _bucket_amount(amt)
            if g == "exact":
                return v

        # fallback
        return v

    def gran(self, a: str, v: str) -> str:
        """
        gran(v): finest g in G_a s.t. gen_a(v,g) == v.
        Induced by the same rule hierarchy as gen_a.
        """
        lat = self.lattices[a]
        for g in reversed(lat.levels):
            if g == "none":
                continue
            if self.gen(a, v, g) == v:
                return g
        return lat.levels[-1]


def _try_parse_amount(s: str) -> Optional[float]:
    m = re.search(r"(\d[\d,]*\.?\d*)", s.replace(",", ""))
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", ""))
    except Exception:
        return None


def _bucket_amount(x: float) -> str:
    if x < 100:
        return "<100"
    if x < 1000:
        return "100-999"
    if x < 10000:
        return "1k-9.9k"
    if x < 100000:
        return "10k-99k"
    return ">=100k"


# -----------------------------
# Extractor E(q) (simple, deterministic)
# -----------------------------

@dataclass
class Extracted:
    a: str
    v: str
    c: float  # confidence


class Extractor:
    def extract(self, q: str) -> List[Extracted]:
        out: List[Extracted] = []
        # email
        for m in re.finditer(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", q):
            out.append(Extracted("contact", m.group(0), 1.0))
        # phone
        for m in re.finditer(r"(?:\+?\d{1,3}[-\s]?)?(?:\d{2,4}[-\s]?)\d{3,4}[-\s]?\d{4}", q):
            out.append(Extracted("contact", m.group(0), 0.95))
        # amounts
        for m in re.finditer(r"(?:\$|₩|￦|€|£)?\s?\d[\d,]*\.?\d*\s?(?:USD|KRW|원|달러|eur|gbp)?", q, flags=re.IGNORECASE):
            s = m.group(0).strip()
            if _try_parse_amount(s) is not None:
                out.append(Extracted("financial", s, 0.8))
        # age
        for m in re.finditer(r"\b(\d{1,3})\s*(?:years old|yo|세)\b", q, flags=re.IGNORECASE):
            out.append(Extracted("age", m.group(1), 0.9))
        # id-like
        for m in re.finditer(r"\b\d{8,}\b", q):
            out.append(Extracted("identifier", m.group(0), 0.8))
        return out


def build_U_and_M(
    q: str,
    extractor: Extractor,
    policy: Policy,
    generalizer: Generalizer,
    delta: Dict[str, float],
) -> Tuple[List[Extracted], List[Tuple[str, str, float]]]:
    """
    U(q) = {(a,v,c) in E(q) | c >= delta_a}
    M(q,R) = {v | (a,v,c) in U(q), gran(v) ≻ R(a)}  (value-level, no closure)
    We keep (v, a, c) so that phi can return attribute/confidence from argmax.
    """
    raw = extractor.extract(q)
    U: List[Extracted] = [e for e in raw if e.c >= float(delta.get(e.a, 0.0))]

    M: List[Tuple[str, str, float]] = []
    for e in U:
        g_v = generalizer.gran(e.a, e.v)
        if generalizer.lattices[e.a].idx(g_v) > generalizer.lattices[e.a].idx(policy(e.a)):
            M.append((e.v, e.a, e.c))
    return U, M


# -----------------------------
# External fixed encoder ψ(·)
# -----------------------------

class FixedEncoder:
    def __init__(self, model_name: str, device: str) -> None:
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.enc = AutoModel.from_pretrained(model_name).to(device)
        self.enc.eval()

    @torch.no_grad()
    def encode(self, texts: List[str], max_len: int = 128) -> torch.Tensor:
        batch = self.tok(
            texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt"
        ).to(self.device)
        out = self.enc(**batch).last_hidden_state  # [B,T,H]
        attn = batch["attention_mask"].unsqueeze(-1)  # [B,T,1]
        pooled = (out * attn).sum(dim=1) / attn.sum(dim=1).clamp(min=1.0)
        return F.normalize(pooled, p=2, dim=-1)  # [B,H]


# -----------------------------
# Core as-is functions: viol, phi, m_R, C_t update
# -----------------------------

@dataclass
class PhiOut:
    a_t: str            # attribute label (or empty)
    c_t: float          # confidence in [0,1]
    viol: bool
    max_sim: float
    argmax_idx: int     # which mask item matched

class NullGater:
    """
    No-op gater for baseline generation.
    Keeps the same interface as DisclosureGaterCategoryFree
    but applies no logit modification.
    """

    def __init__(self, lm_tokenizer):
        self.lm_tok = lm_tokenizer
        self.max_ctx_chars = 0  # interface compatibility
        self.C = 0.0            # interface compatibility

    def apply_gating(self, input_ids, logits, prompt_len: int):
        # return logits unchanged
        return logits

    def update_C_after_choice(self, ctx_text: str, chosen_token_id: int):
        # do nothing
        return None

class DisclosureGaterCategoryFree:
    """
    Category-free enforcement (uniform thresholds/penalties):
      - Ontology/policy/resolution still exists (used to build M(q,R))
      - BUT enforcement does not treat attributes differently:
          * single tau_embed
          * single lambda
          * C_t uses no w_a (equivalent to w_a=1)
    """

    def __init__(
        self,
        lm_tokenizer,
        encoder,
        mask_items: List[Tuple[str, str, float]],  # (v, a, base_conf) from M(q,R)
        tau_embed: float,
        lam: float,
        f_mode: str = "linear",      # f(C)=1+C (or keep as-is)
        device: str = "cpu",
        max_ctx_chars: int = 600,
    ) -> None:
        self.lm_tok = lm_tokenizer
        self.encoder = encoder
        self.tau_embed = tau_embed
        self.lam = lam
        self.f_mode = f_mode
        self.device = device
        self.max_ctx_chars = max_ctx_chars

        self.C = 0.0  # accumulated disclosure state (category-free)

        self.mask_vals = [m[0] for m in mask_items]
        self.mask_attrs = [m[1] for m in mask_items]         # kept for logging
        self.mask_base_confs = [m[2] for m in mask_items]    # not used here

        self.mask_emb = self.encoder.encode(self.mask_vals) if self.mask_vals else None

    def f(self, C: float) -> float:
        if self.f_mode == "linear":
            return 1.0 + C
        # add other monotone options if desired
        return 1.0 + C

    @torch.no_grad()
    def phi_and_viol(self, ctx_text: str, token_id: int) -> PhiOut:
        """
        EXACT as-is with ψ(x;y_<t)=ψ(string(y_<t||x)):
          viol(x;y)=1[max_v sim(ψ(ctx||token), ψ(v)) >= tau_embed]
          phi derived from maximizing pair (argmax) but NOT used for weighting.
        """
        if self.mask_emb is None:
            return PhiOut(a_t="", c_t=0.0, viol=False, max_sim=0.0, argmax_idx=-1)

        tok_text = self.lm_tok.decode([token_id], skip_special_tokens=False)
        if len(ctx_text) > 200:           # 200~400 사이 추천
            ctx_text = ctx_text[-200:]
        x_repr = ctx_text + tok_text  # ψ(string(y_<t||x_t))

        x_emb = self.encoder.encode([x_repr])[0:1]     # [1,H], L2-normalized
        sim = (x_emb @ self.mask_emb.T)[0]             # [M]
        max_sim, idx = torch.max(sim, dim=0)
        print("[PHI] sim_max =", float(max_sim),
      "tau =", self.tau_embed,
      "viol =", max_sim >= self.tau_embed)

        s = float(max_sim.item())
        k = int(idx.item())
        viol = (s >= self.tau_embed)

        if not viol:
            return PhiOut(a_t="", c_t=0.0, viol=False, max_sim=s, argmax_idx=k)

        # category-free: a_t is kept only for debugging/logging
        a_t = self.mask_attrs[k]
        c_t = max(0.0, min(1.0, s))  # continuous sensitivity score from similarity
        return PhiOut(a_t=a_t, c_t=c_t, viol=True, max_sim=s, argmax_idx=k)

    @torch.no_grad()
    def apply_gating(self, input_ids: torch.LongTensor, logits: torch.FloatTensor,  prompt_len: int) -> torch.FloatTensor:
        print("[GATE] called. mask_emb is None?", self.mask_emb is None, "C=", getattr(self, "C", None))

        """
        Apply l_tilde = l - m_R approximately (top-k only):
          - choose top-k candidate tokens from logits
          - evaluate viol(x;y_<t) only for those candidates
          - apply penalty to violating candidates only
        """
        if self.mask_emb is None:
            return logits

        gen_prefix_ids = input_ids[0, prompt_len:]
        ctx = self.lm_tok.decode(gen_prefix_ids, skip_special_tokens=True)
        if len(ctx) > self.max_ctx_chars:
            ctx = ctx[-self.max_ctx_chars:]

        base = 10.0                      # 최소 페널티 (튜닝 파라미터)
        penalty = base + self.lam * self.f(self.C)

        print("[GATE] penalty=", float(penalty), "C=", float(self.C), "lam=", self.lam)

        new_logits = logits.clone()

        # Only evaluate top-k candidates (critical for speed)
        TOPK = 50
        k = min(TOPK, logits.shape[-1])
        topk_ids = torch.topk(logits, k=k, dim=-1).indices[0]   # [k]

        penalized = 0 

        for token_id in topk_ids.tolist():
            phi = self.phi_and_viol(ctx, int(token_id))
            if phi.viol:
                new_logits[0, int(token_id)] -= penalty
                penalized += 1

        print("[GATE] penalized_in_topk =", penalized, "topk =", len(topk_ids))

        return new_logits

    def update_C_after_choice(self, ctx_text: str, chosen_token_id: int) -> PhiOut:
        """
        EXACT update (category-free):
          C_t = C_{t-1} + c_t * 1[viol]
        """
        
        phi = self.phi_and_viol(ctx_text, chosen_token_id)
        print("[C] before", self.C, "chosen_token=", chosen_token_id, "phi.viol=", phi.viol, "phi.attr=", getattr(phi, "attr", None))
        if phi.viol:
            self.C = 0.98 * self.C + phi.c_t
        
        return phi
    
# -----------------------------
# End-to-end generation loop (exact)
# -----------------------------

@torch.no_grad()
def generate_with_exact_logit_gating(
    model_name: str,
    prompt: str,
    gater: DisclosureGater,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    device: str = "cpu",
) -> str:
    tok = gater.lm_tok
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()

    input_ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = input_ids.shape[1]

    for _ in range(max_new_tokens):
        out = model(input_ids=input_ids)
        logits = out.logits[:, -1, :]  # [1,V]

        # apply m_R to logits -> l_tilde
        gated_logits = gater.apply_gating(input_ids, logits, prompt_len)

        # sample next token
        if temperature != 1.0:
            gated_logits = gated_logits / max(1e-6, temperature)
        next_id = torch.argmax(gated_logits, dim=-1, keepdim=True)  # [1,1]

        # update C_t using chosen token x_t and the SAME ctx used in phi
        ctx = tok.decode(input_ids[0], skip_special_tokens=True)
        if len(ctx) > gater.max_ctx_chars:
            ctx = ctx[-gater.max_ctx_chars:]
        oldC = float(gater.C)
        gater.update_C_after_choice(ctx, int(next_id.item()))
        newC = float(gater.C)
        if newC != oldC:
            print("[C] updated", oldC, "->", newC)

        input_ids = torch.cat([input_ids, next_id.to(device)], dim=1)

        gen_ids = input_ids[0, prompt_len:]
        gen_text = tok.decode(gen_ids, skip_special_tokens=True)

        # 1) newline appears -> stop
        if "\n" in gen_text:
            break

        # 2) if it looks like a full question already -> stop
        if len(gen_text) >= 40 and gen_text.strip().endswith("?"):
            break
        if tok.eos_token_id is not None and int(next_id.item()) == tok.eos_token_id:
            break
    
    gen_ids = input_ids[0, prompt_len:]
    
    text = tok.decode(gen_ids, skip_special_tokens=True)
    text = text.strip().splitlines()[0].strip()

    return text

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    in_jsonl = "batch_692ed2c569c881909b1d76f69088a9d1_output (1).jsonl"      # q 파일 경로
    out_jsonl = "triples_out.jsonl"   # (q, baseline, logit) 저장 경로

    # Example lattices/policy (replace with your own exact tables)
    lattices = {
        "identifier": Lattice(["none", "prefix", "full"]),
        "contact": Lattice(["none", "masked", "full"]),
        "location": Lattice(["none", "city", "region", "street"]),
        "financial": Lattice(["none", "bucket", "exact"]),
        "age": Lattice(["none", "decade", "exact"]),
        "affiliation": Lattice(["none", "coarse", "full"]),
    }
    R0 = Policy({
        "identifier": "prefix",
        "contact": "masked",
        "location": "city",
        "financial": "bucket",
        "age": "decade",
        "affiliation": "coarse",
    })
    generalizer = Generalizer(lattices)

    extractor = Extractor()
    delta = {"identifier": 0.6, "contact": 0.8, "location": 0.5, "financial": 0.6, "age": 0.7, "affiliation": 0.6}

    # LLM + tokenizer
    lm_name = "Qwen/Qwen2.5-1.5B-Instruct"  # change to an available model
    lm_tok = AutoTokenizer.from_pretrained(lm_name, use_fast=True)

    # external fixed encoder ψ
    enc = FixedEncoder("sentence-transformers/all-MiniLM-L6-v2", device=device)

    def build_chat_prompt(user_text: str) -> str:
    # Base model friendly: no chat template
        return (
        "You are a privacy-preserving query rewriter.\n"
        "Task: Rewrite the user's question into a clean, self-contained question, "
        "REMOVING or GENERALIZING any personal identifiers.\n"
        "Do NOT ANSWER ANY QUESTIONS or Repeat the user's question.\n"
        "Privacy rules:\n"
        "- REMOVE real names, phone numbers, email addresses, exact street addresses, and account numbers.\n"
        "- Generalize precise ages, incomes, debts and balances into rough ranges or buckets.\n"
        "- KEEP only the structure of the situation that is needed to answer.\n"
        "- Output EXACTLY ONE LINE: the rewritten, anonymized question only.\n"
        "- Do NOT refuse. Do NOT add explanations.\n"
        "\n"
        f"USER QUESTION:\n{user_text}\n"
        "REWRITTEN ANONYMIZED QUESTION:\n"
        )

    n = 0
    for it in read_jsonl_questions(in_jsonl):
        qid = it["qid"]
        q = it["question"]

        q_raw = q
        print(f"[{n}] Processing qid={qid}: {q_raw}")

        baseline_text = generate_with_exact_logit_gating(
            model_name=lm_name,
            prompt=build_chat_prompt(q_raw),
            gater=NullGater(lm_tokenizer=lm_tok),
            max_new_tokens=200,
            temperature=0.9,
            device=device,
        )
        print(f"Baseline answer: {baseline_text}")

        U, M = build_U_and_M(q_raw, extractor, R0, generalizer, delta)

        gater = DisclosureGaterCategoryFree(
            lm_tokenizer=lm_tok,
            encoder=enc,
            mask_items=M,
            tau_embed=0.25,
            lam=8.0,
            f_mode="linear",
            device=device,
            max_ctx_chars=600,
        )
        print("[MASK] n_vals =", len(gater.mask_vals),"sample =", gater.mask_vals[:5])
        logit_text = generate_with_exact_logit_gating(
            model_name=lm_name,
            prompt=build_chat_prompt(q_raw),
            gater=gater,
            max_new_tokens=200,
            temperature=0.9,
            device=device,
        )
        print(f"Logit-gated answer: {logit_text}")

        append_jsonl(out_jsonl, {
            **it,
            "question_raw": q_raw,
            "baseline_answer": baseline_text,
            "logit_answer": logit_text,
        })
        n += 1
        print(f"[{n}] wrote qid={qid}")



if __name__ == "__main__":
    main()
