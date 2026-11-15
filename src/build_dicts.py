"""Download / convert ICD-10 Beijing Clinical Edition and optional CHIP-CDN aliases."""

from __future__ import annotations

import json
import urllib.request
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "dicts" / "raw"
OUT = ROOT / "data" / "dicts"

ICD_URLS = [
    "https://hf-mirror.com/datasets/Aunderline/CHIP-CDEE/resolve/main/%E5%9B%BD%E9%99%85%E7%96%BE%E7%97%85%E5%88%86%E7%B1%BBICD-10%E5%8C%97%E4%BA%AC%E4%B8%B4%E5%BA%8A%E7%89%88v601.xlsx",
    "https://huggingface.co/datasets/Aunderline/CHIP-CDEE/resolve/main/%E5%9B%BD%E9%99%85%E7%96%BE%E7%97%85%E5%88%86%E7%B1%BBICD-10%E5%8C%97%E4%BA%AC%E4%B8%B4%E5%BA%8A%E7%89%88v601.xlsx",
]
CHIP_CDN_FILES = (
    "CHIP-CDN_train.json",
    "CHIP-CDN_dev.json",
)


def download(url: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "emr-kg-baseline"})
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            dest.write_bytes(resp.read())
        print(f"downloaded {dest} ({dest.stat().st_size} bytes)")
        return True
    except Exception as exc:
        print(f"skip {url}: {exc}")
        return False


def convert_icd(xlsx: Path, out_json: Path) -> dict:
    df = pd.read_excel(xlsx, header=None, names=["code", "name"])
    df = df.dropna(subset=["name"])
    df["code"] = df["code"].astype(str).str.strip()
    df["name"] = df["name"].astype(str).str.strip()
    df = df[(df["name"] != "") & (df["name"] != "nan")]
    name_to_codes: dict[str, list[str]] = defaultdict(list)
    for code, name in zip(df["code"], df["name"]):
        if code not in name_to_codes[name]:
            name_to_codes[name].append(code)
    payload = {
        "source": "ICD-10 Beijing Clinical Edition v601 (CBLUE CHIP-CDN vocabulary)",
        "xlsx": str(xlsx.name),
        "n_rows": int(len(df)),
        "n_names": len(name_to_codes),
        "name_to_codes": dict(name_to_codes),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out_json} names={payload['n_names']}")
    return payload


def convert_chip_cdn(raw_dir: Path, out_json: Path) -> dict:
    alias: dict[str, str] = {}
    n_rows = 0
    for fname in CHIP_CDN_FILES:
        path = raw_dir / fname
        if not path.exists():
            continue
        rows = json.loads(path.read_text(encoding="utf-8"))
        n_rows += len(rows)
        for row in rows:
            text = str(row.get("text", "")).strip()
            norms = [x.strip() for x in str(row.get("normalized_result", "")).split("##") if x.strip()]
            norms = [x for x in norms if x != "癌"]
            if not text or not norms:
                continue
            alias[text] = norms[0]
    payload = {
        "source": "CHIP-CDN train/dev mention-to-ICD aliases",
        "n_rows": n_rows,
        "n_alias": len(alias),
        "alias": alias,
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out_json} alias={payload['n_alias']} from_rows={n_rows}")
    return payload


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    xlsx = RAW / "icd10_beijing_v601.xlsx"
    if not xlsx.exists():
        ok = False
        for url in ICD_URLS:
            if download(url, xlsx):
                ok = True
                break
        if not ok:
            raise SystemExit("ICD xlsx download failed")
    convert_icd(xlsx, OUT / "icd10_beijing_v601.json")

    found = [RAW / f for f in CHIP_CDN_FILES if (RAW / f).exists()]
    if found:
        convert_chip_cdn(RAW, OUT / "chip_cdn_alias.json")
    else:
        print(
            "CHIP-CDN json not found. Place CHIP-CDN_train.json / CHIP-CDN_dev.json "
            f"in {RAW} (from https://tianchi.aliyun.com/dataset/95414) and rerun. "
            "ICD vocabulary matching still works."
        )
        if not (OUT / "chip_cdn_alias.json").exists():
            (OUT / "chip_cdn_alias.json").write_text(
                json.dumps({"source": "empty", "n_rows": 0, "n_alias": 0, "alias": {}}, ensure_ascii=False),
                encoding="utf-8",
            )


if __name__ == "__main__":
    main()
