"""Would the external-support gain survive a drug alias table built without the test set?

data/dicts/drug_alias.json was curated on CCKS test drug surfaces. This keeps only the entries whose
key also occurs among CCKS train+dev drug surfaces (gold entities), i.e. the part of the table one
could have written from training notes alone, and re-runs the G_raw vs G_norm external comparison
on the test set.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import normalize
from eval_ablation import FULL
from eval_external_extra import PROCESSED, compare
from eval_external_kg import doc_level_graph, load_cmeie, load_medicalkg
from eval_g_raw_vs_g_norm import collect_freq
from ner_data import load_bio_docs
from normalize import build_canonical_map, configure_normalization, drug_core, light_clean

OUT = ROOT / "outputs" / "metrics"


def key_forms(s: str) -> set[str]:
    c = light_clean(s)
    return {f for f in (c, c.upper(), drug_core(c), drug_core(c).upper()) if f}


def restrict_drug_alias(idx, allowed_forms: set[str]) -> int:
    kept = {k: v for k, v in idx.drug_alias.items() if key_forms(k) & allowed_forms}
    idx.drug_alias = kept
    idx.drug_targets = set(kept.values())
    idx.drug_alias_norm = {light_clean(k).upper(): v for k, v in kept.items() if light_clean(k)}
    for k, v in list(idx.drug_alias_norm.items()):
        core = drug_core(k)
        if core and core.upper() not in idx.drug_alias_norm:
            idx.drug_alias_norm[core.upper()] = v
    return len(kept)


def graph_with(docs, allowed_forms: set[str] | None) -> tuple[dict, int]:
    configure_normalization(**FULL)
    idx = normalize.get_index()
    n = len(idx.drug_alias) if allowed_forms is None else restrict_drug_alias(idx, allowed_forms)
    canon = build_canonical_map(collect_freq(docs))
    g = doc_level_graph(docs, canon)
    configure_normalization()
    return g, n


def main() -> None:
    sources = {"cmeie": load_cmeie(ROOT / "data" / "dicts" / "raw" / "CMeIE"),
               "medicalkg": load_medicalkg(ROOT / "data" / "dicts" / "raw" / "medicalkg" / "medical.json")}
    train_docs = load_bio_docs(PROCESSED / "train.bio") + load_bio_docs(PROCESSED / "dev.bio")
    allowed = set().union(*(key_forms(s) for s in collect_freq(train_docs).get("药物", {})))

    res = {}
    lines = ["药物别名表只保留训练+开发集里出现过的条目，测试集外部印证率（疾病→药物）", "=" * 72]
    for name, title in (("test_pred_ner", "测试集 预测实体"), ("test_gold_ner", "测试集 标注实体")):
        docs = load_bio_docs(PROCESSED / ("test_pred.bio" if name == "test_pred_ner" else "test.bio"))
        raw = doc_level_graph(docs, None)
        res[name] = {}
        for key, forms in (("full_table", None), ("train_only_table", allowed)):
            g, n_alias = graph_with(docs, forms)
            res[name][key] = {"n_drug_alias": n_alias} | {s: compare(raw, g, ext) for s, ext in sources.items()}
        for s, st in (("cmeie", "CMeIE"), ("medicalkg", "寻医问药")):
            parts = []
            for key, kt in (("full_table", "完整别名表"), ("train_only_table", "仅训练集可得")):
                r = res[name][key][s]
                p = r["paired"]
                parts.append(f"{kt}({res[name][key]['n_drug_alias']}条) {r['g_norm']['support_rate']:.3f} "
                             f"差 {p['delta']:+.3f} [{p['delta_ci95'][0]:+.3f}, {p['delta_ci95'][1]:+.3f}]")
            g_raw = res[name]["full_table"][s]["g_raw"]["support_rate"]
            lines.append(f"  {title:10s}{st:6s} G_raw {g_raw:.3f}  " + "  ".join(parts))
    text = "\n".join(lines) + "\n"
    (OUT / "alias_heldout.txt").write_text(text, encoding="utf-8")
    (OUT / "alias_heldout.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
