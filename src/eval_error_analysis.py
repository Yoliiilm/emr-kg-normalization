"""Error analysis for normalization: categorized misses (漏并) and wrong merges (误并)."""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bert_linker import ORGANS
from normalize import (
    DictIndex,
    clean_surface,
    configure_normalization,
    keep_entity,
    lookup_with_source,
)
from ner_data import bio_to_entities, load_bio_docs

OUT = ROOT / "outputs" / "metrics"
DEV = ROOT / "data" / "dicts" / "raw" / "CHIP-CDN_dev.json"
TRAIN = ROOT / "data" / "dicts" / "raw" / "CHIP-CDN_train.json"
BOOT = OUT / "bootstrap_query_f1.json"

N_SHOW = 15
SITES = ORGANS + (
    "鼻咽", "咽", "喉", "甲状腺", "乳腺", "乳", "胆囊", "胆管", "胰", "脾", "肾上腺", "输尿管",
    "前列腺", "睾丸", "阴道", "外阴", "输卵管", "腹膜", "淋巴结", "骨", "椎", "颈", "胸", "腹",
    "盆腔", "心", "脑", "血管", "静脉", "动脉", "皮肤", "腮腺", "舌", "口", "鼻", "眼", "耳",
)
MORPH_RE = re.compile(r"(癌|瘤|肉瘤|腺癌|鳞癌|Ca|CA|ca)")
GRADE_ONLY_RE = re.compile(r"^[（(]?[中低高至\-～~、ⅠⅡⅢIV0-9级期度分化危极]+[)）]?$")
FINDING_WORDS = (
    "心律", "功能减退", "增生", "增厚", "钙化", "结节", "占位", "肿物", "包块", "阴影", "积液",
    "改变", "病变", "血供", "升高", "降低", "血糖", "盗血", "狭窄", "扩张",
)
COMPOUND_RE = re.compile(r"[、及和或+＋]|合并|伴")
TREATMENT_WORDS = ("术后", "化疗", "放疗", "复查", "开药", "随访", "治疗", "定期", "维持")

MISS_CATS = {
    "compound": "复合/多诊断写在一起",
    "paren_site_lost": "清洗时括号内部位被丢弃",
    "histology_no_site": "组织学词无部位（如 中分化腺癌）",
    "grade_fragment": "分化/分期/危险度残片（NER 切碎）",
    "finding": "检查所见/生理描述，非规范诊断",
    "metastasis": "转移写法未改写为继发恶性肿瘤",
    "too_specific": "部位或修饰过细，ICD 无对应粒度",
    "other": "其他长尾",
}
WRONG_CATS = {
    "multi_gold": "多诊断仅链到其中一个（或都没链对）",
    "site": "部位不符",
    "laterality": "单侧/双侧/左右信息丢失",
    "coarse": "粒度过粗（链到更上位的病名）",
    "fine": "粒度过细（链到更下位的病名）",
    "near_synonym": "近义但非标准名",
}
CDN_MISS_CATS = {
    "multi_gold": "多诊断复合",
    "treatment": "含治疗/就诊描述",
    "abbrev": "缩写/英文/口语",
    "other": "其他长尾",
}


def gold_norms(row: dict) -> list[str]:
    raw = str(row.get("normalized_result", "")).strip()
    return [x.strip() for x in raw.split("##") if x.strip() and x.strip() != "癌"]


def acc_hit(pred: str, golds: list[str]) -> bool:
    if not pred or not golds:
        return False
    p = re.sub(r"\s+", "", pred)
    for g in golds:
        g2 = re.sub(r"\s+", "", g)
        if p == g2 or p in g2 or g2 in p:
            return True
    return False


def sites_in(text: str) -> set[str]:
    found = {s for s in SITES if s in text}
    return {s for s in found if not any(s != o and s in o for o in found)}


def sites_overlap(a: set[str], b: set[str]) -> bool:
    """肝脏/肝、胃窦/胃 count as the same site (granularity, not a site error)."""
    return any(x in y or y in x for x in a for y in b)


def miss_category(surface: str, kept: str) -> str:
    if COMPOUND_RE.search(re.sub(r"[(（][^)）]*[)）]", "", surface)):
        return "compound"
    paren = "".join(re.findall(r"[(（]([^)）]*)[)）]", surface))
    if paren and sites_in(paren) and not sites_in(kept):
        return "paren_site_lost"
    if GRADE_ONLY_RE.match(kept) or kept in {"极高危", "高危", "中危", "低危"}:
        return "grade_fragment"
    if "转移" in kept:
        return "metastasis"
    if MORPH_RE.search(kept) and not sites_in(kept):
        return "histology_no_site"
    if any(w in kept for w in FINDING_WORDS) or len(kept) <= 2:
        return "finding"
    if sites_in(kept):
        return "too_specific"
    return "other"


def wrong_category(text: str, pred: str, golds: list[str]) -> str:
    if len(golds) >= 2:
        return "multi_gold"
    g = golds[0]
    ps, gs = sites_in(pred), sites_in(g)
    if ps and gs and not sites_overlap(ps, gs):
        return "site"
    if any(w in g for w in ("单侧", "双侧", "左", "右")) and not any(
        w in pred for w in ("单侧", "双侧", "左", "右")
    ):
        return "laterality"
    if SequenceMatcher(None, pred, g).ratio() >= 0.75:
        return "near_synonym"
    if set(pred) < set(g) or (len(pred) < len(g) and set(pred) <= set(g + text)):
        return "coarse"
    if set(g) < set(pred) or len(pred) > len(g):
        return "fine"
    return "near_synonym"


def cdn_miss_category(text: str, golds: list[str]) -> str:
    if len(golds) >= 2:
        return "multi_gold"
    if any(w in text for w in TREATMENT_WORDS):
        return "treatment"
    if re.search(r"[A-Za-z]", text):
        return "abbrev"
    return "other"


def collect_emr(idx: DictIndex) -> dict:
    freq: Counter[str] = Counter()
    for chars, tags in load_bio_docs(ROOT / "data" / "processed" / "test_pred.bio"):
        for e in bio_to_entities(chars, tags):
            if e["label"] == "疾病和诊断" and keep_entity(e["text"], e["label"]):
                freq[e["text"]] += 1

    misses: dict[str, list[dict]] = defaultdict(list)
    merges = []
    for name, c in freq.most_common():
        canon, src = lookup_with_source(name, "疾病和诊断")
        if src == "heuristic":
            misses[miss_category(name, canon)].append({"surface": name, "freq": c, "kept_as": canon})
        elif name != canon:
            merges.append(
                {"surface": name, "to": canon, "source": src, "freq": c, "icd": idx.preferred_code(canon)}
            )

    clusters: dict[str, list[str]] = defaultdict(list)
    for m in merges:
        clusters[m["to"]].append(m["surface"])

    suspicious = []
    for m in merges:
        s, t = m["surface"], m["to"]
        if ("左" in s and "右" in t and "左" not in t) or ("右" in s and "左" in t and "右" not in t):
            suspicious.append({**m, "reason": "左右翻转"})
        elif re.match(r"^[0-9IVⅠⅡⅢⅣ]+型", s) and not re.search(r"[0-9IVⅠⅡⅢⅣ]+型", t):
            suspicious.append({**m, "reason": "分型被剥离（如 1型/2型 并成同一病）"})
        elif sites_in(s) and sites_in(t) and not sites_overlap(sites_in(s), sites_in(t)):
            suspicious.append({**m, "reason": "部位不符"})
        elif "术后" in s and "术后" not in t and not MORPH_RE.search(t):
            suspicious.append({**m, "reason": "术后状态丢失"})

    worse = []
    if BOOT.exists():
        boot = json.loads(BOOT.read_text(encoding="utf-8"))
        for rel, r in boot["relations"].items():
            for x in r.get("worse_examples", []):
                canon_aliases = clusters.get(x["canonical"], [])
                worse.append({**x, "rel": rel, "cluster_aliases": canon_aliases[:8]})

    return {
        "n_surfaces": len(freq),
        "n_merges": len(merges),
        "n_misses": sum(len(v) for v in misses.values()),
        "misses": misses,
        "suspicious": suspicious,
        "worse_queries": worse,
    }


def collect_cdn(idx: DictIndex) -> dict:
    train_alias = {}
    for row in json.loads(TRAIN.read_text(encoding="utf-8")):
        text = str(row.get("text", "")).strip()
        norms = gold_norms(row)
        if text and norms:
            train_alias[text] = norms[0]
    idx.alias = train_alias
    idx.alias_clean = {clean_surface(k): v for k, v in train_alias.items() if clean_surface(k)}

    wrong: dict[str, list[dict]] = defaultdict(list)
    miss: dict[str, list[dict]] = defaultdict(list)
    n = 0
    for row in json.loads(DEV.read_text(encoding="utf-8")):
        text = str(row.get("text", "")).lstrip("\ufeff").strip()
        golds = gold_norms(row)
        if not text or not golds:
            continue
        n += 1
        pred, src = idx.lookup_disease(text)
        if acc_hit(pred, golds):
            continue
        item = {"text": text, "pred": pred, "gold": "##".join(golds), "source": src}
        if src == "heuristic" or pred == clean_surface(text) or pred == text:
            miss[cdn_miss_category(text, golds)].append(item)
        else:
            wrong[wrong_category(text, pred, golds)].append(item)
    return {"n": n, "wrong": wrong, "miss": miss}


def section(lines: list[str], title: str, groups: dict[str, list[dict]], names: dict[str, str], fmt) -> None:
    total = sum(len(v) for v in groups.values()) or 1
    lines.append(title)
    for key in sorted(groups, key=lambda k: -len(groups[k])):
        items = groups[key]
        lines.append(f"  ■ {names[key]}：{len(items)}（{len(items) / total:.1%}）")
        for x in items[:4]:
            lines.append(f"      - {fmt(x)}")
    lines.append("")


def main() -> None:
    configure_normalization(use_chip_cdn=True, use_drug_alias=True, use_surgery_alias=True)
    idx = DictIndex.load()
    emr = collect_emr(idx)
    cdn = collect_cdn(idx)

    lines = [
        "归一化错误分析（论文用）",
        "=" * 64,
        "",
        "A. CCKS 测试集预测疾病表面词（本任务场景）",
        f"  表面词={emr['n_surfaces']}  已合并={emr['n_merges']}  原样保留(漏并候选)={emr['n_misses']}",
        "",
    ]
    section(
        lines,
        "【A1 漏并：原样保留的表面词按原因归类】",
        emr["misses"],
        MISS_CATS,
        lambda x: f"{x['surface']}  (freq={x['freq']}, 保留为「{x['kept_as']}」)",
    )
    lines.append("  注: 「组织学词无部位」在建图时按同篇病历主癌借部位（如 中分化腺癌 → 直肠恶性肿瘤），")
    lines.append("      这里按表面词单独统计，仍计为未链接。")
    lines.append("")

    lines.append("【A2 误并证据一：归一化后查询 F1 反而下降的疾病查询】")
    if not emr["worse_queries"]:
        lines.append("  （缺少 bootstrap_query_f1.json，先运行 python src/eval_bootstrap.py）")
    for x in emr["worse_queries"][:N_SHOW]:
        others = [a for a in x["cluster_aliases"] if a != x["query"]]
        lines.append(
            f"  - [{x['rel']}] {x['query']} → {x['canonical']}  "
            f"F1 {x['f_raw']:.2f}→{x['f_norm']:.2f}  同簇: {'、'.join(others[:5]) or '-'}"
        )
    lines.append("")

    lines.append("【A3 误并证据二：规则筛出的可疑合并】")
    if not emr["suspicious"]:
        lines.append("  （无）")
    for x in emr["suspicious"][:N_SHOW]:
        lines.append(f"  - {x['surface']} → {x['to']} [{x['source']}]  原因: {x['reason']}")
    lines.append("")

    n_wrong = sum(len(v) for v in cdn["wrong"].values())
    n_miss = sum(len(v) for v in cdn["miss"].values())
    lines.append("B. CHIP-CDN_dev（外部标注，train-only 词表，可对照 gold）")
    lines.append(f"  n={cdn['n']}  误并(链到错误标准名)={n_wrong}  漏并(未链接/近原样)={n_miss}")
    lines.append("")
    section(
        lines,
        "【B1 误并按原因归类】",
        cdn["wrong"],
        WRONG_CATS,
        lambda x: f"{x['text']} ⇒ {x['pred']} | gold={x['gold']} [{x['source']}]",
    )
    section(
        lines,
        "【B2 漏并按原因归类】",
        cdn["miss"],
        CDN_MISS_CATS,
        lambda x: f"{x['text']} | gold={x['gold']}",
    )

    lines.append("完整例子（每类最多 20 条）见 error_analysis.json")

    payload = {
        "emr_test_pred": {
            "n_surfaces": emr["n_surfaces"],
            "n_merges": emr["n_merges"],
            "n_misses": emr["n_misses"],
            "miss_counts": {MISS_CATS[k]: len(v) for k, v in emr["misses"].items()},
            "misses": {MISS_CATS[k]: v[:20] for k, v in emr["misses"].items()},
            "worse_queries": emr["worse_queries"],
            "suspicious_merges": emr["suspicious"][:20],
        },
        "chip_cdn_dev": {
            "n": cdn["n"],
            "wrong_counts": {WRONG_CATS[k]: len(v) for k, v in cdn["wrong"].items()},
            "miss_counts": {CDN_MISS_CATS[k]: len(v) for k, v in cdn["miss"].items()},
            "wrong": {WRONG_CATS[k]: v[:20] for k, v in cdn["wrong"].items()},
            "miss": {CDN_MISS_CATS[k]: v[:20] for k, v in cdn["miss"].items()},
        },
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "error_analysis.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "error_analysis.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
