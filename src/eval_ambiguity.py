"""Ambiguity in disease normalization, after Newman-Griffis et al. (JAMIA 2021).

1. CHIP-CDN: share of repeated strings annotated with more than one gold set, within train,
   within dev, and between train and dev (same string, different gold).
2. Composite mentions (one string → several concepts): CHIP-CDN gold, LLM answers on CCKS.
3. Dictionary: ICD-10 Beijing names that carry more than one code.
4. CCKS context dependence: site-less histology (中分化腺癌) mentions, how many distinct nodes
   the same surface resolves to across documents via the document's primary cancer, and what a
   context-free normalizer (the LLM map) gives for the same surfaces.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eval_cdn_baselines import DEV, TRAIN, gold_norms
from eval_g_raw_vs_g_norm import collect_freq
from ner_data import bio_to_entities, load_bio_docs
from normalize import (
    LLM_MAP,
    build_canonical_map,
    configure_normalization,
    histology_without_site,
    keep_entity,
    needs_context,
)

OUT = ROOT / "outputs" / "metrics"
PROCESSED = ROOT / "data" / "processed"
ICD = ROOT / "data" / "dicts" / "icd10_beijing_v601.json"
SETS = (("CCKS2019 test", "test.bio"), ("CCKS2017", "ccks2017_gold.bio"))
DISEASE = "疾病和诊断"


def key(s: str) -> str:
    return re.sub(r"\s+", "", s.lstrip("\ufeff"))


def loose_key(s: str) -> str:
    from train_cdn_cblue import str_q2b

    return re.sub(r"[\W_]+", "", str_q2b(s)).lower()


def chip_strings(path: Path) -> dict[str, list[frozenset[str]]]:
    """Strings keyed after dropping punctuation, spaces and full/half-width differences."""
    out: dict[str, list[frozenset[str]]] = defaultdict(list)
    for r in json.loads(path.read_text(encoding="utf-8")):
        t = loose_key(str(r.get("text", "")))
        g = gold_norms(r)
        if t and g:
            out[t].append(frozenset(key(x) for x in g))
    return out


def chip_part() -> tuple[list[str], dict]:
    tr, dv = chip_strings(TRAIN), chip_strings(DEV)
    rep: dict = {}
    L = ["一、CHIP-CDN：同一个写法被标成不同答案的比例（Newman-Griffis 表 2 的做法）",
         "  （原文逐字没有重复；这里去掉标点、空格、全半角差别后再比）"]
    for name, d in (("train", tr), ("dev", dv)):
        rows = sum(len(v) for v in d.values())
        rep_str = {t: v for t, v in d.items() if len(v) >= 2}
        amb = {t for t, v in rep_str.items() if len(set(v)) > 1}
        multi = sum(len(g) >= 2 for v in d.values() for g in v)
        rep[name] = {"rows": rows, "strings": len(d), "repeated": len(rep_str), "ambiguous": len(amb),
                     "multi_gold_rows": multi, "examples": [(t, [sorted(g) for g in set(d[t])]) for t in sorted(amb)[:5]]}
        L.append(f"  {name}：{rows} 条，{len(d)} 个不同写法；重复出现的写法 {len(rep_str)} 个，其中答案不一致 {len(amb)} 个"
                 f"（{len(amb) / max(1, len(rep_str)):.1%}）；一条对应 ≥2 个标准名的 {multi} 条（{multi / rows:.1%}）")
    common = sorted(set(tr) & set(dv))
    diff = [t for t in common if set(tr[t]) != set(dv[t]) and not (set(tr[t]) & set(dv[t]))]
    partial = [t for t in common if set(tr[t]) != set(dv[t]) and (set(tr[t]) & set(dv[t]))]
    rep["train_dev"] = {"common": len(common), "disjoint": len(diff), "partial": len(partial),
                        "examples": [(t, sorted(map(sorted, set(tr[t]))), sorted(map(sorted, set(dv[t])))) for t in diff[:6]]}
    L.append(f"  train 与 dev 共有写法 {len(common)} 个：两边答案完全不同 {len(diff)} 个（{len(diff) / max(1, len(common)):.1%}），"
             f"部分重合 {len(partial)} 个")
    for t, a, b in rep["train_dev"]["examples"]:
        L.append(f"    例：{t}  train={'/'.join('##'.join(x) for x in a)}  dev={'/'.join('##'.join(x) for x in b)}")
    return L, rep


def icd_part() -> tuple[list[str], dict]:
    n2c = json.loads(ICD.read_text(encoding="utf-8"))["name_to_codes"]
    multi = {n: c for n, c in n2c.items() if len(set(c)) > 1}
    ex = sorted(multi.items(), key=lambda kv: -len(set(kv[1])))[:5]
    L = ["二、ICD-10 北京临床版：一个标准名对应多个编码",
         f"  标准名 {len(n2c)} 个，对应 ≥2 个编码的 {len(multi)} 个（{len(multi) / len(n2c):.1%}）",
         "    例：" + "；".join(f"{n}（{len(set(c))} 个）" for n, c in ex)]
    return L, {"names": len(n2c), "multi_code": len(multi)}


def ccks_part() -> tuple[list[str], dict]:
    llm = json.loads(LLM_MAP.read_text(encoding="utf-8")) if LLM_MAP.exists() else {}
    L = ["三、CCKS 病历里依赖上下文的写法：没写部位的组织学词（中分化腺癌、鳞癌…）"]
    rep: dict = {}
    configure_normalization()
    for label, fname in SETS:
        docs = load_bio_docs(PROCESSED / fname)
        canon = build_canonical_map(collect_freq(docs))
        n_dis = n_hist = n_ctx = 0
        targets: dict[str, Counter] = defaultdict(Counter)
        for chars, tags in docs:
            ents = [e for e in bio_to_entities(chars, tags)
                    if e["text"].strip() and keep_entity(e["text"], e["label"]) and e["label"] == DISEASE]
            names = []
            for e in ents:
                c = canon.get((DISEASE, e["text"]), e["text"])
                names.append(c if isinstance(c, tuple) else (c,))
            primary = Counter(n for ns in names for n in ns
                              if n.endswith("恶性肿瘤") and not n.endswith("继发恶性肿瘤"))
            for e, ns in zip(ents, names):
                n_dis += 1
                if not histology_without_site(e["text"]):
                    continue
                n_hist += 1
                if primary and needs_context(e["text"], DISEASE, ns):
                    n_ctx += 1
                    targets[e["text"]][primary.most_common(1)[0][0]] += 1
                else:
                    targets[e["text"]]["（未补部位）" + ns[0]] += 1
        multi = {t: c for t, c in targets.items() if sum(1 for k in c if not k.startswith("（")) >= 2}
        llm_view = {t: llm.get(t) for t in sorted(targets, key=lambda t: -sum(targets[t].values()))[:6]}
        rep[label] = {"disease_mentions": n_dis, "histology_no_site": n_hist, "resolved_by_context": n_ctx,
                      "surfaces": len(targets), "surfaces_multi_node": len(multi),
                      "mentions_in_multi": sum(sum(c.values()) for c in multi.values()),
                      "examples": {t: dict(c.most_common(4)) for t, c in sorted(multi.items(), key=lambda kv: -sum(kv[1].values()))[:5]},
                      "llm_context_free": llm_view}
        r = rep[label]
        L.append(f"  【{label}】疾病提及 {n_dis}；没写部位的组织学词 {n_hist}（{n_hist / max(1, n_dis):.1%}），"
                 f"其中借本病历主癌补上部位 {n_ctx}；涉及写法 {len(targets)} 个，"
                 f"同一写法在不同病历里落到 ≥2 个不同节点的 {len(multi)} 个（{r['mentions_in_multi']} 次提及）")
        for t, c in r["examples"].items():
            L.append(f"    {t}：" + "、".join(f"{k} {v}" for k, v in c.items()))
        L.append("    不看上下文的大模型答案：" + "；".join(f"{t} → {'##'.join(v) if v else '保持原样'}"
                                             for t, v in llm_view.items() if v is not None))
    return L, rep


def llm_composite() -> tuple[list[str], dict]:
    key_path = ROOT / "outputs" / "llm_norm" / "key.json"
    if not key_path.exists() or not LLM_MAP.exists():
        return [], {}
    items = json.loads(key_path.read_text(encoding="utf-8"))
    llm = json.loads(LLM_MAP.read_text(encoding="utf-8"))
    out = {}
    for src, flag in (("CHIP-CDN dev", "cdn"), ("CCKS 建图写法", "graph")):
        texts = [it["text"] for it in items if it.get(flag) and it["text"] in llm]
        out[src] = {"n": len(texts), "multi": sum(len(llm[t]) >= 2 for t in texts),
                    "zero": sum(not llm[t] for t in texts)}
    L = ["四、一个写法对应多个概念（复合写法）：大模型答案里多选 / 答 0 的比例"]
    for src, v in out.items():
        L.append(f"  {src}：{v['n']} 条，多选 {v['multi']}（{v['multi'] / max(1, v['n']):.1%}），"
                 f"答 0 {v['zero']}（{v['zero'] / max(1, v['n']):.1%}）")
    return L, out


def main() -> None:
    head = ["疾病标准化里的歧义（参照 Newman-Griffis 等 2021, JAMIA）", "=" * 72]
    parts = [chip_part(), icd_part(), ccks_part(), llm_composite()]
    text = "\n".join(head + [line for L, _ in parts for line in L + [""]])
    rep = dict(zip(("chip_cdn", "icd", "ccks_context", "llm_composite"), (r for _, r in parts)))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ambiguity.txt").write_text(text, encoding="utf-8")
    (OUT / "ambiguity.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=list), encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
