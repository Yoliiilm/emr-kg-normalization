"""Build the public release: the 1127 LLM-jury disease–drug pairs, the ranked answers of the three
graphs, a blank physician sheet, a standalone evaluate.py, and a copy of the code.

Only disease / drug names leave the repo (no record text). Writes release/.
"""

from __future__ import annotations

import csv
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import normalize
from eval_ablation import FULL
from llm_judge_score import JUDGES, JUDGE_DIR, canonical_drug_of, read_labels

OUT = ROOT / "release"
PHYSICIAN_N = 150
PRIVATE_SRC = {"notify.py", "make_release.py"}


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    key = json.loads((JUDGE_DIR / "key.json").read_text(encoding="utf-8"))
    items = list(csv.DictReader((JUDGE_DIR / "items.csv").open(encoding="utf-8-sig")))
    labels = {j: read_labels(j) for j, _ in JUDGES}

    normalize.configure_normalization(**FULL)
    idx = normalize.get_index()
    icd = {r["disease"]: idx.preferred_code(r["disease"]) or "" for r in items}
    normalize.configure_normalization()
    to_canon = canonical_drug_of()

    graphs_of: dict[str, set[str]] = {}
    kb_of: dict[str, bool] = {}
    for q in key["queries"]:
        for gn in key["graphs"]:
            for x in q["graphs"][gn]:
                graphs_of.setdefault(x["item"], set()).add(gn)
                if x["kb"] is not None:
                    kb_of[x["item"]] = kb_of.get(x["item"], False) or x["kb"]
    calib = {c["item"]: c["kind"] for c in key["calibration"]}

    rows = []
    for r in items:
        i = r["id"]
        votes = [labels[j][i] for j, _ in JUDGES]
        rows.append({
            "id": i, "disease": r["disease"], "icd10_beijing": icd[r["disease"]], "drug": r["drug"],
            "drug_normalized": to_canon.get(r["drug"], r["drug"]),
            "source": "+".join(s for s in (("graph" if i in graphs_of else ""), ("calibration" if i in calib else "")) if s),
            "graphs": ";".join(g for g in key["graphs"] if g in graphs_of.get(i, ())),
            "calibration": calib.get(i, ""),
            "kb_supported": "" if i not in kb_of else int(kb_of[i]),
            **{f"label_{j}": labels[j][i] for j, _ in JUDGES},
            "jury_treats": int(sum(v == "2" for v in votes) >= 2),
            "jury_treats_or_related": int(sum(v in ("1", "2") for v in votes) >= 2),
            "jury_not_a_drug": int(sum(v == "X" for v in votes) >= 2),
        })
    fields = list(rows[0])
    write_csv(OUT / "data" / "pairs.csv", fields, rows)

    queries = [{"disease": q["disease"], "icd10_beijing": icd.get(q["disease"], ""),
                "raw_surface": q["raw_node"].split("::", 1)[1], "kb_covered": q["kb_covered"],
                "ranked": {gn: [{"id": x["item"], "score": x["score"]} for x in q["graphs"][gn]] for gn in key["graphs"]}}
               for q in key["queries"]]
    (OUT / "data" / "queries.json").write_text(
        json.dumps({"k": key["k"], "graphs": key["graphs"], "queries": queries}, ensure_ascii=False, indent=1),
        encoding="utf-8")

    rng = random.Random(2026)
    pool = sorted(i for i in graphs_of)
    pick = set(rng.sample(pool, PHYSICIAN_N))
    sheet = [{"id": r["id"], "disease": r["disease"], "drug": r["drug"], "label": "", "note": ""}
             for r in rows if r["id"] in pick]
    rng.shuffle(sheet)
    write_csv(OUT / "physician" / "physician_sheet.csv", ["id", "disease", "drug", "label", "note"], sheet)

    code = OUT / "code"
    if code.exists():
        shutil.rmtree(code)
    (code / "src").mkdir(parents=True)
    for p in sorted(SRC.glob("*.py")):
        if p.name not in PRIVATE_SRC:
            shutil.copy2(p, code / "src" / p.name)
    shutil.copy2(ROOT / "requirements.txt", code / "requirements.txt")

    n_src = Counter(r["source"] for r in rows)
    print(f"pairs {len(rows)} {dict(n_src)}  queries {len(queries)}  physician sheet {len(sheet)}  "
          f"code files {len(list((code / 'src').glob('*.py')))}  ICD codes missing "
          f"{sum(not r['icd10_beijing'] for r in rows)}")


if __name__ == "__main__":
    main()
