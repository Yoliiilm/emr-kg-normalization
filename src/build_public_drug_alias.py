"""Brand → generic drug alias table built only from public sources (no CCKS notes involved).

Sources (data/dicts/raw/drug_brand/ and data/dicts/raw/medicalkg/):
  original-drug-list  qianguyihao/original-drug-list 01-data/*.json  (trade_name, drug_generic_name)
  yuanyanyao          dongzhenye/yuanyanyao data/*.md front matter   (brandName, genericName/productName)
  medicalkg           medical.json drug_detail "商品名(通用名)" where the outer part is not the generic itself

Targets are reduced with normalize.drug_core (dose form / salt stripped) plus a leading 注射用, so a
brand lands on the same node the heuristic would give its generic. A brand that maps to different
targets across or within sources is dropped as ambiguous.

Writes data/dicts/drug_alias_public.json.
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

from normalize import FORM_TAILS, SALT_HEADS, drug_core, light_clean

RAW = ROOT / "data" / "dicts" / "raw"
OUT = ROOT / "data" / "dicts" / "drug_alias_public.json"
COMPANY_TAILS = ("制药", "药业", "医药", "药厂", "生物", "公司", "集团", "股份")
NOT_A_BRAND = {"", "无", "暂无", "-", "—", "null", "None"}
# Generic names in drug registries carry the salt; surfaces in notes usually drop it.
TARGET_SALT_HEADS = SALT_HEADS + ("磷酸", "硫酸", "酒石酸", "枸橼酸", "琥珀酸", "醋酸", "氢溴酸", "苯甲酸",
                                  "乳酸", "门冬氨酸", "托西酸", "双羟萘酸", "依地酸")


def generic_target(name: str) -> str:
    s = drug_core(re.sub(r"^注射用", "", light_clean(name)))
    for tail in FORM_TAILS:
        if s.endswith(tail) and len(s) - len(tail) >= 2:
            s = s[: -len(tail)]
            break
    for head in TARGET_SALT_HEADS:
        if s.startswith(head) and len(s) - len(head) >= 2:
            return s[len(head):]
    return s


def clean_brand(name: str) -> str:
    s = light_clean(name).replace("®", "").replace("™", "").strip("\"'“”‘’ ")
    return re.sub(r"[(（][^)）]*[)）]", "", s)


def ok_pair(brand: str, target: str) -> bool:
    if brand in NOT_A_BRAND or not target or not 2 <= len(brand) <= 8:
        return False
    if brand == target or target in brand or brand in target:
        return False
    return not brand.endswith(COMPANY_TAILS)


def from_original_drug_list() -> list[tuple[str, str]]:
    pairs = []
    for f in sorted((RAW / "drug_brand" / "original-drug-list" / "01-data").glob("*.json")):
        for r in json.loads(f.read_text(encoding="utf-8")).get("data", []):
            pairs.append((r.get("trade_name") or "", r.get("drug_generic_name") or ""))
    return pairs


def from_yuanyanyao() -> list[tuple[str, str]]:
    pairs = []
    for f in sorted((RAW / "drug_brand" / "yuanyanyao" / "data").glob("*.md")):
        text = f.read_text(encoding="utf-8")
        meta = dict(re.findall(r"^(brandName|genericName|productName):\s*(.*?)\s*$", text, flags=re.M))
        pairs.append((meta.get("brandName", ""), meta.get("genericName") or meta.get("productName", "")))
    return pairs


def from_medicalkg() -> list[tuple[str, str]]:
    pairs = []
    path = RAW / "medicalkg" / "medical.json"
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        for d in json.loads(line).get("drug_detail") or []:
            m = re.fullmatch(r"([^()]+)\(([^()]+)\)", d.strip())
            if m:
                pairs.append((m.group(1), m.group(2)))
    return pairs


def main() -> None:
    sources = {"original-drug-list": from_original_drug_list(), "yuanyanyao": from_yuanyanyao(),
               "medicalkg": from_medicalkg()}
    targets: dict[str, set[str]] = defaultdict(set)
    origin: dict[str, set[str]] = defaultdict(set)
    stats: dict[str, dict] = {}
    for src, pairs in sources.items():
        kept = 0
        for b, g in pairs:
            brand, target = clean_brand(b), generic_target(g)
            if ok_pair(brand, target):
                targets[brand].add(target)
                origin[brand].add(src)
                kept += 1
        stats[src] = {"records": len(pairs), "brand_pairs": kept}
    alias, ambiguous = {}, {}
    for b, t in sorted(targets.items()):
        shortest = min(t, key=len)
        if all(shortest in x for x in t):
            alias[b] = shortest
        else:
            ambiguous[b] = sorted(t)
    per_source = Counter(s for b in alias for s in origin[b])
    stats["merged"] = {"brands": len(alias), "ambiguous_dropped": len(ambiguous),
                       "brands_by_source": dict(per_source)}
    OUT.write_text(json.dumps({"stats": stats, "ambiguous_examples": dict(list(ambiguous.items())[:30]),
                               "alias": alias}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=1))
    for k in ("希罗达", "泰素", "伯尔定", "乐沙定", "健择", "泰索帝", "美罗华", "赫赛汀", "艾恒", "艾素", "达菲"):
        print(k, "→", alias.get(k), ambiguous.get(k, ""))


if __name__ == "__main__":
    main()
