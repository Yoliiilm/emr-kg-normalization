"""Compare free-form LLM answers with raw-graph and normalized-graph evidence."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eval_g_raw_vs_g_norm import collect_freq, graph_counts
from ner_data import load_bio_docs
from normalize import build_canonical_map, lookup_with_source

DEFAULT_QUESTION = "2 型糖尿病常用什么药？"
DEFAULT_OUT = ROOT / "outputs" / "experiments" / "graph_qa" / "answers.json"


def disease_from_question(question: str, explicit: str | None = None) -> str:
    if explicit:
        return explicit.strip()
    text = re.sub(r"[？?！!。,.，、\s]+$", "", question.strip())
    text = re.sub(r"(常用什么药|用什么药|有哪些药|怎么治疗|如何治疗|的治疗方案)$", "", text)
    text = re.sub(r"^(请问|请问一下)", "", text)
    if not text:
        raise ValueError("Cannot extract a disease from the question; provide --disease")
    return text


def graph_evidence(docs, canon_map, disease: str, normalized: bool, top_k: int) -> dict:
    nodes, relations, _surfaces, _evidence = graph_counts(docs, canon_map if normalized else None)
    canonical, _source = lookup_with_source(disease, "疾病和诊断")
    if normalized:
        source_ids = [f"疾病和诊断::{canonical}"]
    else:
        exact_surfaces = {disease, re.sub(r"\s+", "", disease)}
        source_ids = [f"疾病和诊断::{surface}" for surface in exact_surfaces]

    counts: dict[str, int] = defaultdict(int)
    for (source, rel, target), weight in relations.items():
        if rel == "使用药物" and source in source_ids:
            counts[target] += weight
    drug_names = {nid: row["name"] for nid, row in nodes.items()}
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:top_k]
    return {
        "canonical_query": canonical,
        "disease_nodes": [source_id for source_id in source_ids if source_id in nodes],
        "drugs": [{"name": drug_names.get(nid, nid.split("::", 1)[-1]), "weight": weight} for nid, weight in ranked],
    }


def ask_model(question: str, model: str, api_base: str, api_key: str, evidence: dict | None) -> str:
    if evidence is None:
        system = "你是医学知识问答助手。请直接回答用户问题；没有可靠依据的内容要明确说明不确定，不要伪造来源。"
        user = question
    else:
        system = (
            "你是医学知识图谱问答助手。只能根据提供的图谱证据回答；证据没有覆盖的药物不要补充或猜测。"
            "说明证据来自图谱共现，不等同于临床指南或个体化处方。"
        )
        user = json.dumps({"question": question, "graph_evidence": evidence["drugs"]}, ensure_ascii=False)
    payload = json.dumps({
        "model": model,
        "temperature": 0,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
    }).encode("utf-8")
    request = urllib.request.Request(
        api_base.rstrip("/") + "/chat/completions",
        data=payload,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM request failed ({exc.code}): {detail}") from exc
    return result["choices"][0]["message"]["content"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", default=DEFAULT_QUESTION, help="natural-language disease/drug question")
    parser.add_argument("--disease", help="disease surface; optional override for graph retrieval")
    parser.add_argument("--bio", type=Path, default=ROOT / "data" / "processed" / "test_pred.bio")
    parser.add_argument("--model", default=os.getenv("GRAPH_QA_MODEL", "gpt-4o-mini"))
    parser.add_argument("--api-base", default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--prepare-only", action="store_true", help="save evidence without calling an LLM")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    question_disease = disease_from_question(args.question, args.disease)
    disease, _source = lookup_with_source(question_disease, "疾病和诊断")
    docs = load_bio_docs(args.bio)
    canon_map = build_canonical_map(collect_freq(docs))
    raw_evidence = graph_evidence(docs, canon_map, question_disease, normalized=False, top_k=args.top_k)
    norm_evidence = graph_evidence(docs, canon_map, question_disease, normalized=True, top_k=args.top_k)
    result = {
        "question": args.question,
        "disease_surface": question_disease,
        "canonical_disease": disease,
        "model": args.model,
        "data": str(args.bio),
        "conditions": {},
    }
    conditions = (("no_graph", None), ("raw_graph", raw_evidence), ("normalized_graph", norm_evidence))
    if args.prepare_only:
        result["conditions"] = {name: {"evidence": evidence} for name, evidence in conditions}
    else:
        api_key = os.getenv(args.api_key_env)
        if not api_key:
            raise SystemExit(f"Missing API key environment variable: {args.api_key_env}")
        for name, evidence in conditions:
            answer = ask_model(args.question, args.model, args.api_base, api_key, evidence)
            result["conditions"][name] = {"evidence": evidence, "answer": answer}
            print(f"\n[{name}]\n{answer}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
