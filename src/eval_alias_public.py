"""Does the external-support gain hold with a drug alias table taken only from public sources?

Drug alias tables compared (everything else = the FULL normalization setting):
  none               no table, drug_core heuristic only
  hand_full          data/dicts/drug_alias.json (curated on CCKS test drug surfaces)
  hand_train_only    the hand entries whose key also occurs among train+dev drug surfaces
  public             data/dicts/drug_alias_public.json (src/build_public_drug_alias.py)
  public_plus_train  public ∪ hand_train_only (hand entries win on conflicts)

For each document set it reports the share of drug mentions a table rewrites (a table tuned on the
test set rewrites far more test mentions than train ones) and the G_raw vs G_norm external support
on 疾病→药物 edges. Writes outputs/metrics/alias_public.{txt,json}.
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
from eval_alias_heldout import key_forms
from eval_cross_dataset import load_ccks2017
from eval_external_extra import PROCESSED, compare
from eval_external_kg import doc_level_graph, load_cmeie, load_medicalkg
from eval_g_raw_vs_g_norm import collect_freq
from ner_data import load_bio_docs
from normalize import build_canonical_map, close_alias, configure_normalization, drug_core, light_clean

DICTS = ROOT / "data" / "dicts"
OUT = ROOT / "outputs" / "metrics"
TABLES = (("none", "无别名表"), ("hand_full", "手工完整表"), ("hand_train_only", "手工表·仅训练集可得"),
          ("public", "公开商品名表"), ("public_plus_train", "公开表+训练集可得"))
DOC_SETS = (("test_pred_ner", "测试集 预测实体"), ("test_gold_ner", "测试集 标注实体"),
            ("train_dev_gold_ner", "训练+开发 标注实体"), ("ccks2017_gold_ner", "CCKS2017 标注实体"))


def set_drug_alias(idx, table: dict[str, str]) -> None:
    idx.drug_alias = close_alias(table)
    idx.drug_targets = set(idx.drug_alias.values())
    idx.drug_alias_norm = {light_clean(k).upper(): v for k, v in idx.drug_alias.items() if light_clean(k)}
    for k, v in list(idx.drug_alias_norm.items()):
        core = drug_core(k)
        if core and core.upper() not in idx.drug_alias_norm:
            idx.drug_alias_norm[core.upper()] = v


def rewrite_share(idx, drug_freq: dict[str, int]) -> dict:
    total = changed = 0
    for s, c in drug_freq.items():
        raw = light_clean(s)
        name, how = idx.lookup_drug(s)
        total += c
        if how == "drug_alias" and name != (drug_core(raw) or raw):
            changed += c
    return {"mentions": total, "rewritten": changed, "share": changed / total if total else 0.0}


def norm_graph_with(docs, table: dict[str, str]) -> tuple[dict, dict]:
    configure_normalization(**FULL)
    idx = normalize.get_index()
    set_drug_alias(idx, table)
    freq = collect_freq(docs)
    share = rewrite_share(idx, freq.get("药物", {}))
    g = doc_level_graph(docs, build_canonical_map(freq))
    configure_normalization()
    return g, share


def table_overlap(hand: dict[str, str], public: dict[str, str]) -> dict:
    pub_forms = {f: v for k, v in public.items() for f in key_forms(k)}
    found = [k for k in hand if key_forms(k) & pub_forms.keys()]
    agree = [k for k in found if any(pub_forms.get(f) == hand[k] for f in key_forms(k))]
    return {"hand_keys": len(hand), "in_public": len(found), "same_target": len(agree), "in_public_keys": found}


def main() -> None:
    hand = json.loads((DICTS / "drug_alias.json").read_text(encoding="utf-8"))["alias"]
    public = json.loads((DICTS / "drug_alias_public.json").read_text(encoding="utf-8"))["alias"]
    train_docs = load_bio_docs(PROCESSED / "train.bio") + load_bio_docs(PROCESSED / "dev.bio")
    allowed = set().union(*(key_forms(s) for s in collect_freq(train_docs).get("药物", {})))
    hand_train = {k: v for k, v in hand.items() if key_forms(k) & allowed}
    tables = {"none": {}, "hand_full": hand, "hand_train_only": hand_train, "public": public,
              "public_plus_train": {**public, **hand_train}}
    sources = {"cmeie": load_cmeie(DICTS / "raw" / "CMeIE"),
               "medicalkg": load_medicalkg(DICTS / "raw" / "medicalkg" / "medical.json")}
    doc_sets = {"test_pred_ner": load_bio_docs(PROCESSED / "test_pred.bio"),
                "test_gold_ner": load_bio_docs(PROCESSED / "test.bio"),
                "train_dev_gold_ner": train_docs,
                "ccks2017_gold_ner": load_ccks2017()[0]}

    res = {"table_sizes": {k: len(v) for k, v in tables.items()}, "hand_vs_public": table_overlap(hand, public)}
    for name, docs in doc_sets.items():
        raw = doc_level_graph(docs, None)
        res[name] = {"n_docs": len(docs)}
        for key, table in tables.items():
            g, share = norm_graph_with(docs, table)
            res[name][key] = {"rewrite": share} | {s: compare(raw, g, ext) for s, ext in sources.items()}
            print(name, key, f"rewrite {share['share']:.3f}",
                  {s: round(res[name][key][s]["paired"]["delta"], 4) for s in sources}, flush=True)

    ov = res["hand_vs_public"]
    lines = ["公开药品商品名表 vs 手工药物别名表：外部印证率（疾病→药物，G_norm 相对 G_raw 的配对差）", "=" * 78,
             "别名表条数：" + "  ".join(f"{t}={res['table_sizes'][k]}" for k, t in TABLES),
             f"手工表 {ov['hand_keys']} 个键中，公开表也有 {ov['in_public']} 个，其中目标一致 {ov['same_target']} 个", ""]
    for name, title in DOC_SETS:
        lines.append(f"[{title}]  {res[name]['n_docs']} 份病历")
        for key, t in TABLES:
            r = res[name][key]
            cells = []
            for s, st in (("cmeie", "CMeIE"), ("medicalkg", "寻医问药")):
                p = r[s]["paired"]
                cells.append(f"{st} {r[s]['g_norm']['support_rate']:.3f} 差 {p['delta']:+.3f} "
                             f"[{p['delta_ci95'][0]:+.3f}, {p['delta_ci95'][1]:+.3f}]")
            lines.append(f"  {t:14s} 改写药物提及 {r['rewrite']['share']:6.1%}   " + "   ".join(cells))
        g_raw = {s: res[name]["none"][s]["g_raw"]["support_rate"] for s in sources}
        lines.append(f"  （G_raw：CMeIE {g_raw['cmeie']:.3f}  寻医问药 {g_raw['medicalkg']:.3f}）")
        lines.append("")
    text = "\n".join(lines)
    (OUT / "alias_public.txt").write_text(text, encoding="utf-8")
    (OUT / "alias_public.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
