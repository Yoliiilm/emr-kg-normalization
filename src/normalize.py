"""Dictionary-first entity normalization (ICD-10 Beijing Clinical + optional CHIP-CDN)."""

from __future__ import annotations

import json
import re
from pathlib import Path

from clinical_gates import morph_ok, site_ok

ROOT = Path(__file__).resolve().parents[1]
DICT_DIR = ROOT / "data" / "dicts"

NOISE_TAILS = [
    "术后改变",
    "术后复发",
    "姑息术后",
    "根治术后",
    "术后",
    "姑息",
    "可能性大",
    "可能",
    "性质待查",
    "待查",
    "待排",
    "待除外",
    "待鉴别",
    "？",
    "?",
]
NOISE_HEADS = [
    "复发性",
    "转移性",
    "原发性",
    "继发性",
    "不全性",
    "不完全性",
    "局部晚期",
    "晚期",
    "早期",
]
LAT_HEADS = ("左侧", "右侧", "双侧", "两侧", "左", "右", "两")

GRADE_RE = re.compile(r"(中至低|中低|中-低|高中|高|中|低)(度)?分化")
LEFT_OK = re.compile(
    r"^(?:[0-9IV]+级|[0-9IV]+期|型|症|急发|急性期|极高危组|极高危|转移|轻度|中度|重度|原因不能明确)*$"
)
_INDEX: "DictIndex | None" = None
_SETTINGS = {
    "use_chip_cdn": True,
    "use_drug_alias": True,
    "use_surgery_alias": True,
    "disease_rules_only": False,
    "use_split": True,
    "use_context": True,
    "use_new_rules": True,
    "use_curated": True,
    "supervised_cdn": "off",
    "supervised_min_p": 0.5,
    "retriever": "roberta",
    "retriever_fallback": "off",
    "llm_cdn": "off",
    "use_drug_class": False,
    "calibration_mode": "off",
    "calibration_model": str(ROOT / "outputs" / "models" / "normalization_calibrator.json"),
    "calibration_neural": True,
}

SITE_WORDS = (
    "乙状结肠", "升结肠", "横结肠", "降结肠", "十二指肠", "直肠", "结肠", "回肠", "空肠", "盲肠",
    "胃窦", "胃底", "胃体", "贲门", "食管", "卵巢", "子宫", "宫颈", "膀胱", "肝", "肺", "肾", "脾",
    "胃", "脑", "鼻咽", "咽", "喉", "甲状腺", "乳腺", "乳", "胆囊", "胆管", "胰", "肾上腺",
    "输尿管", "前列腺", "睾丸", "阴道", "外阴", "输卵管", "腹膜", "淋巴结", "骨", "肠", "舌",
    "腮腺", "口腔", "鼻", "皮肤", "胸膜", "纵隔", "内膜",
)
TYPE_PREFIX_RE = re.compile(r"^(IV|III|II|I|Ⅳ|Ⅲ|Ⅱ|Ⅰ|[1-4])型")
ROMAN_UNI = {"I": "Ⅰ", "II": "Ⅱ", "III": "Ⅲ", "IV": "Ⅳ"}
ARABIC_ROMAN = {"1": "I", "2": "II", "3": "III", "4": "IV"}
LEAD_PAREN_RE = re.compile(r"^[(（]([^)）]{1,10})[)）]")
CANCER_RE = re.compile(r"(癌|Ca|CA)")
STRONG_SPLIT_RE = re.compile(r"[、，,;；+＋]|伴有?|合并|并发")
WEAK_SPLIT_RE = re.compile(r"[、，,;；+＋]|伴有?|合并|并发|及|和")

# NER often tags radiotherapy planning volumes as 药物; drop at graph build time.
DRUG_BLOCKLIST = frozenset({"GTV", "CTV", "PTV"})


def configure_normalization(
    *,
    use_chip_cdn: bool = True,
    use_drug_alias: bool = True,
    use_surgery_alias: bool = True,
    disease_rules_only: bool = False,
    use_split: bool = True,
    use_context: bool = True,
    use_new_rules: bool = True,
    use_curated: bool = True,
    supervised_cdn: str = "off",
    supervised_min_p: float = 0.5,
    retriever: str = "roberta",
    retriever_fallback: str = "off",
    llm_cdn: str = "off",
    use_drug_class: bool = False,
    calibration_mode: str = "off",
    calibration_model: str | None = None,
    calibration_neural: bool = True,
) -> None:
    """Toggle normalization components for ablation; clears cached DictIndex.

    disease_rules_only: string rewriting only (no ICD / curated / BERT / CHIP lookup).
    use_split: a compound disease mention may map to several canonical names.
    use_context: site-less histology (中分化腺癌) borrows the document's primary cancer.
    use_new_rules: Ⅱ型/2型 prefix variants, leading bracketed site, X转移 → X继发恶性肿瘤,
        <site>…癌/腺瘤 → <site>恶性/良性肿瘤.
    use_curated: the hand-written long-tail alias table disease_alias_extra.json.
    supervised_cdn: off | fallback (only for otherwise-unlinked mentions) |
        primary (right after exact ICD match); model from src/train_cdn_cblue.py.
    supervised_min_p: keep a supervised name only if its ranker probability ≥ this
        (0.5 = the CBLUE commit rule, i.e. no extra filtering).
    retriever: ICD fallback encoder, roberta (off-the-shelf) | sapbert (src/train_sapbert.py).
    retriever_fallback: for mentions still unlinked at the end, off | threshold (retriever top-1
        if its score ≥ the retriever's calibrated min_score, no site/morph filters) | top1 (always).
    llm_cdn: off | fallback | primary, same positions as supervised_cdn; answers are read from
        outputs/llm_norm/map.json (src/llm_norm_sample.py → LLM labels → src/llm_norm_score.py).
    use_drug_class: drug class terms (抗生素, 降压药物) → ATC class nodes (src/drug_class.py).
    calibration_mode: off | abstain; use the fitted confidence calibrator before the
        legacy disease cascade and keep the surface form when confidence is low.
    calibration_model: JSON model written by src/fit_calibrator.py.
    calibration_neural: include BERT/SapBERT top-k candidates in calibrated mode.
    """
    global _INDEX, _SETTINGS
    if supervised_cdn not in ("off", "fallback", "primary"):
        raise ValueError(f"supervised_cdn must be off|fallback|primary, got {supervised_cdn!r}")
    if llm_cdn not in ("off", "fallback", "primary"):
        raise ValueError(f"llm_cdn must be off|fallback|primary, got {llm_cdn!r}")
    if llm_cdn != "off" and supervised_cdn != "off":
        raise ValueError("llm_cdn and supervised_cdn are alternatives; enable at most one")
    if not 0.5 <= supervised_min_p <= 1.0:
        raise ValueError(f"supervised_min_p must be in [0.5, 1], got {supervised_min_p!r}")
    if retriever not in ("roberta", "sapbert"):
        raise ValueError(f"retriever must be roberta|sapbert, got {retriever!r}")
    if retriever_fallback not in ("off", "threshold", "top1"):
        raise ValueError(f"retriever_fallback must be off|threshold|top1, got {retriever_fallback!r}")
    if calibration_mode not in ("off", "abstain"):
        raise ValueError(f"calibration_mode must be off|abstain, got {calibration_mode!r}")
    _SETTINGS = {
        "use_chip_cdn": bool(use_chip_cdn),
        "use_drug_alias": bool(use_drug_alias),
        "use_surgery_alias": bool(use_surgery_alias),
        "disease_rules_only": bool(disease_rules_only),
        "use_split": bool(use_split),
        "use_context": bool(use_context),
        "use_new_rules": bool(use_new_rules),
        "use_curated": bool(use_curated),
        "supervised_cdn": supervised_cdn,
        "supervised_min_p": float(supervised_min_p),
        "retriever": retriever,
        "retriever_fallback": retriever_fallback,
        "llm_cdn": llm_cdn,
        "use_drug_class": bool(use_drug_class),
        "calibration_mode": calibration_mode,
        "calibration_model": calibration_model or str(ROOT / "outputs" / "models" / "normalization_calibrator.json"),
        "calibration_neural": bool(calibration_neural),
    }
    _INDEX = None


def normalization_settings() -> dict:
    return dict(_SETTINGS)


def laterality_ok(query: str, hit: str) -> bool:
    """Reject CHIP/BERT hits that invent or flip 左/右."""
    q_has = any(s in query for s in ("左", "右"))
    h_left, h_right = "左" in hit, "右" in hit
    if not q_has and (h_left or h_right) and not any(s in hit for s in ("两", "双")):
        # e.g. 肺癌 → 右肺恶性肿瘤
        if re.search(r"[左右][侧]?[肺肝肾]", hit) or re.search(r"[肺肝肾][左右]", hit):
            return False
        if hit.startswith(("左", "右")) and len(hit) - len(query) >= 1:
            return False
    if "左" in query and "右" in hit and "左" not in hit:
        return False
    if "右" in query and "左" in hit and "右" not in hit:
        return False
    return True


def chip_alias_ok(surface: str, target: str) -> bool:
    """Conservative gate for CHIP-CDN aliases (many training rows are noisy)."""
    q = clean_surface(surface) or surface.strip()
    t = (target or "").strip()
    if not q or not t or q == t:
        return False
    if "伴多个并发症" in t and "伴多个并发症" not in surface:
        return False
    if "术后" in t and "术后" not in surface and "术后" not in q:
        # e.g. 膀胱癌 → 膀胱术后
        if "癌" in q or "瘤" in q or "恶性肿瘤" in q:
            return False
    if not laterality_ok(q, t) and not laterality_ok(surface, t):
        return False
    # Map clinical 癌/Ca to ICD 恶性肿瘤 before morph/site checks.
    q_icd = (
        q.replace("鳞癌", "鳞状细胞恶性肿瘤")
        .replace("腺癌", "腺恶性肿瘤")
        .replace("Ca", "恶性肿瘤")
        .replace("CA", "恶性肿瘤")
        .replace("癌", "恶性肿瘤")
    )
    morph_pass = morph_ok(q, t) or morph_ok(q_icd, t)
    site_pass = site_ok(q, t) or site_ok(q_icd, t)
    if not morph_pass or not site_pass:
        return False
    return True


def keep_entity(text: str, label: str) -> bool:
    if label != "药物":
        return True
    return light_clean(text).upper() not in DRUG_BLOCKLIST


def light_clean(text: str) -> str:
    s = to_halfwidth(text).strip()
    s = re.sub(r"\s+", "", s)
    return s.strip("\"'“”‘’") or text.strip()


def strip_grade(text: str) -> str:
    t = GRADE_RE.sub("", text)
    if len(t) < 3 or t in {"癌", "腺癌", "鳞癌", "肉瘤", "瘤"}:
        return text
    return t


def to_halfwidth(text: str) -> str:
    out = []
    for ch in text:
        code = ord(ch)
        if code == 0x3000:
            out.append(" ")
        elif 0xFF01 <= code <= 0xFF5E:
            out.append(chr(code - 0xFEE0))
        else:
            out.append(ch)
    return "".join(out)


def strip_affixes(text: str) -> str:
    s = text
    changed = True
    while changed:
        changed = False
        for tail in NOISE_TAILS:
            if s.endswith(tail) and len(s) > len(tail):
                s = s[: -len(tail)]
                changed = True
        for head in NOISE_HEADS + list(LAT_HEADS):
            if s.startswith(head) and len(s) - len(head) >= 2:
                s = s[len(head) :]
                changed = True
    return s


def _clean_once(text: str) -> str:
    s = to_halfwidth(text).strip()
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[(（][^)）]*[)）]", "", s)
    s = s.strip("\"'“”‘’()（）[]【】")
    s = strip_affixes(s)
    s = re.sub(r"^(I{1,3}|IV|[0-9]+)型", "", s)
    s = re.split(r"[，,;；]|伴", s, maxsplit=1)[0]
    s = re.sub(r"[，,。.;；:：/\\]+$", "", s)
    for _ in range(4):
        ns = re.sub(r"((?:I|II|III|IV|[A-D]|[0-9]+)期|T\dN\dM\d|[A-Za-z])$", "", s)
        if ns == s:
            break
        s = ns
    return s or text.strip()


def clean_surface(text: str) -> str:
    """Iterate to a fixed point so canonical names are idempotent under re-normalization."""
    s = _clean_once(text)
    for _ in range(3):
        ns = _clean_once(s)
        if ns == s:
            break
        s = ns
    return s


def icd_variants(text: str) -> list[str]:
    """ICD-10 CN uses 恶性肿瘤 where clinical notes often write 癌."""
    s = text
    out: list[str] = []
    seen: set[str] = set()

    def add(x: str) -> None:
        if x and x not in seen:
            seen.add(x)
            out.append(x)

    add(s)
    if s.endswith("恶性肿瘤"):
        add(s[:-4] + "癌")
        return out
    if s.endswith("鳞状细胞癌"):
        add(s[:-5] + "鳞癌")
        add(s[:-5] + "恶性肿瘤")
    if s.endswith("腺癌") and len(s) >= 4:
        add(s[:-2] + "恶性肿瘤")
        add(s[:-2] + "癌")
    if s.endswith("鳞癌") and len(s) >= 4:
        add(s[:-2] + "鳞状细胞癌")
        add(s[:-2] + "鳞状细胞恶性肿瘤")
        add(s[:-2] + "恶性肿瘤")
        add(s[:-2] + "癌")
    if s.endswith("细胞癌") and len(s) >= 5:
        add(s[:-3] + "恶性肿瘤")
    if s.endswith("癌") and not s.endswith("恶性肿瘤") and len(s) >= 2:
        add(s[:-1] + "恶性肿瘤")
    if s.endswith("肌瘤") and "平滑肌瘤" not in s and len(s) >= 4:
        add(s[:-2] + "平滑肌瘤")
    if s.endswith("转移瘤") and len(s) >= 4:
        add(s[:-3] + "继发恶性肿瘤")
    if s.endswith("转移癌") and len(s) >= 4:
        add(s[:-3] + "继发恶性肿瘤")
    if s.endswith("炎症") and len(s) >= 3:
        add(s[:-1])
    if s.endswith("炎") and not s.endswith("炎症") and len(s) >= 2:
        add(s + "症")
    if s.endswith("增生症") and len(s) >= 5:
        add(s[:-1])
    if s.endswith("性肝癌") and len(s) >= 5:
        add(s[:-3] + "癌")
    if "梗塞" in s:
        add(s.replace("梗塞", "梗死"))
    if "纤颤" in s:
        add(s.replace("纤颤", "颤动"))
        add(s.replace("纤颤", "颤动[心房纤颤]") if "心房" in s else s)
    return out


def type_prefix_variants(text: str) -> list[str]:
    """II型呼吸衰竭 ↔ Ⅱ型呼吸衰竭 ↔ 2型…; ICD mixes ASCII / Unicode roman / arabic."""
    s = light_clean(text)
    m = TYPE_PREFIX_RE.match(s)
    if not m:
        return []
    tok, rest = m.group(1), clean_surface(s[m.end() :])
    uni_to_ascii = {v: k for k, v in ROMAN_UNI.items()}
    ascii_tok = ARABIC_ROMAN.get(tok) or uni_to_ascii.get(tok) or tok
    arabic = {v: k for k, v in ARABIC_ROMAN.items()}.get(ascii_tok, tok)
    forms = [ascii_tok, ROMAN_UNI.get(ascii_tok, ascii_tok), arabic]
    return list(dict.fromkeys(f"{f}型{rest}" for f in forms))


def site_words_in(text: str) -> set[str]:
    found = {w for w in SITE_WORDS if w in text}
    return {w for w in found if not any(w != o and w in o for o in found)}


def histology_without_site(text: str) -> bool:
    """中分化腺癌 / 鳞癌 / 黏膜内癌: cancer morphology but no anatomical site."""
    t = strip_grade(clean_surface(text))
    return bool(CANCER_RE.search(t)) and not site_words_in(text) and "转移" not in text


RULE_TARGET_MARKERS = ("继发恶性肿瘤", "恶性肿瘤", "平滑肌瘤", "梗死", "颤动")


def rule_disease(text: str) -> str:
    """Dictionary-free baseline: affix/stage/grade stripping + 癌→恶性肿瘤 style rewrites."""
    base = strip_grade(clean_surface(text))
    for cand in icd_variants(base)[1:]:
        if any(m in cand for m in RULE_TARGET_MARKERS) and "[" not in cand:
            return cand
    return base


FORM_TAILS = (
    "肠溶胶囊",
    "缓释片",
    "分散片",
    "肠溶片",
    "注射液",
    "口服液",
    "气雾剂",
    "洗液",
    "笔芯",
    "胶囊",
    "颗粒",
    "片",
    "栓",
)
SALT_HEADS = ("盐酸", "苯磺酸", "富马酸", "甲磺酸", "马来酸", "葡萄糖酸")


def drug_core(text: str) -> str:
    s = light_clean(text)
    s = re.sub(r"[(（][^)）]*[)）]", "", s)
    s = s.strip("\"'“”‘’")
    # Keep short letter+digit codes (M2, R0…); only strip dose suffixes.
    if not re.fullmatch(r"[A-Za-z]{1,3}\d{1,2}", s):
        s = re.sub(r"\d+(\.\d+)?\s*(mg|g|ml|iu|μg|%|μg)?$", "", s, flags=re.I)
    changed = True
    while changed:
        changed = False
        for tail in FORM_TAILS:
            # Require >=3 char stem so 眩晕片 / 三金片 are not destroyed.
            if s.endswith(tail) and len(s) - len(tail) >= 3:
                s = s[: -len(tail)]
                changed = True
    for head in SALT_HEADS:
        if s.startswith(head) and len(s) - len(head) >= 2:
            s = s[len(head) :]
            break
    return s or light_clean(text)


def surgery_core(text: str) -> str:
    """Normalize punctuation / latin procedure tokens for surgery surfaces."""
    s = light_clean(text)
    s = s.replace("（", "(").replace("）", ")").replace("＋", "+").replace("、", ",")
    s = s.replace("’", "'").replace("‘", "'").replace("″", "'").replace("'", "'")
    s = re.sub(r"DIXONS?'?S?", "DIXON", s, flags=re.I)
    s = re.sub(r"\bMILES\b", "MILES", s, flags=re.I)
    s = re.sub(r"ROUX-?EN-?Y", "ROUX-EN-Y", s, flags=re.I)
    return s


def lab_core(text: str) -> str:
    s = light_clean(text)
    s = s.rstrip("%")
    if re.match(r"^[A-Za-z]{2,}\d+\.\d+$", s):
        s = re.sub(r"\d+\.\d+$", "", s)
    return s or light_clean(text)


def close_alias(alias: dict[str, str]) -> dict[str, str]:
    """Follow alias chains (A→B, B→C ⇒ A→C) so every target is terminal."""
    out = {}
    for k, v in alias.items():
        seen = {k}
        while v in alias and v not in seen:
            seen.add(v)
            v = alias[v]
        out[k] = v
    return out


class DictIndex:
    def __init__(
        self,
        names: dict[str, list[str]],
        alias: dict[str, str],
        drug_alias: dict[str, str] | None = None,
        curated: dict[str, str] | None = None,
        surgery_alias: dict[str, str] | None = None,
    ):
        self.names = names
        self.alias = alias
        self.drug_alias = close_alias(drug_alias or {})
        self.curated = curated or {}
        self.surgery_alias = close_alias(surgery_alias or {})
        self.drug_targets = set(self.drug_alias.values())
        self.surgery_targets = set(self.surgery_alias.values())
        compact = {re.sub(r"\s+", "", k): k for k in names}
        self.compact = compact
        self.alias_clean = {clean_surface(k): v for k, v in alias.items() if clean_surface(k)}
        self.curated_clean = {
            clean_surface(k): v for k, v in self.curated.items() if clean_surface(k)
        }
        self.surgery_alias_norm = {
            surgery_core(k): v for k, v in self.surgery_alias.items() if surgery_core(k)
        }
        self.drug_alias_norm = {
            light_clean(k).upper(): v for k, v in self.drug_alias.items() if light_clean(k)
        }
        for k, v in list(self.drug_alias_norm.items()):
            core = drug_core(k)
            if core and core.upper() not in self.drug_alias_norm:
                self.drug_alias_norm[core.upper()] = v
        heads: dict[str, list[str]] = {}
        for name in names:
            if len(name) >= 2:
                heads.setdefault(name[:2], []).append(name)
        self.by_head = heads

        def sites_for(suffix: str) -> set[str]:
            out = set()
            for name in names:
                if name.endswith(suffix) and not name.endswith("继发" + suffix):
                    site = name[: -len(suffix)]
                    if 1 <= len(site) <= 6 and not re.search(r"[性型的和及、,，未特指多发部位]", site):
                        out.add(site)
            return out

        self.cancer_sites = sites_for("恶性肿瘤")
        self.benign_sites = sites_for("良性肿瘤")
        self._multi_cache: dict[str, list[tuple[str, str]]] = {}

    @classmethod
    def load(cls, dict_dir: Path = DICT_DIR) -> "DictIndex":
        icd_path = dict_dir / "icd10_beijing_v601.json"
        if not icd_path.exists():
            raise FileNotFoundError(f"Missing {icd_path}; run python src/build_dicts.py")
        icd = json.loads(icd_path.read_text(encoding="utf-8"))
        alias: dict[str, str] = {}
        if _SETTINGS["use_chip_cdn"]:
            alias_path = dict_dir / "chip_cdn_alias.json"
            if alias_path.exists():
                alias = json.loads(alias_path.read_text(encoding="utf-8")).get("alias", {})
        drug_alias: dict[str, str] = {}
        if _SETTINGS["use_drug_alias"]:
            drug_path = dict_dir / "drug_alias.json"
            if drug_path.exists():
                drug_alias = json.loads(drug_path.read_text(encoding="utf-8")).get("alias", {})
        curated: dict[str, str] = {}
        curated_path = dict_dir / "disease_alias_extra.json"
        if _SETTINGS.get("use_curated", True) and curated_path.exists():
            curated = json.loads(curated_path.read_text(encoding="utf-8")).get("alias", {})
        surgery_alias: dict[str, str] = {}
        if _SETTINGS.get("use_surgery_alias", True):
            surg_path = dict_dir / "surgery_alias.json"
            if surg_path.exists():
                surgery_alias = json.loads(surg_path.read_text(encoding="utf-8")).get("alias", {})
        return cls(icd["name_to_codes"], alias, drug_alias, curated, surgery_alias)

    def lookup_curated(self, raw: str, cleaned: str) -> str | None:
        for key in (raw, cleaned):
            if not key:
                continue
            hit = self.curated.get(key) or self.curated_clean.get(key)
            if not hit:
                continue
            resolved = self.resolve_name(hit) or hit
            if self.resolve_name(resolved):
                return resolved
        return None

    def fuzzy_icd(self, text: str) -> str | None:
        """Conservative character similarity against ICD names with same 2-char head."""
        from difflib import SequenceMatcher

        q = clean_surface(text) or text.strip()
        if len(q) < 4:
            return None
        best, best_r = None, 0.0
        for name in self.by_head.get(q[:2], ()):
            if abs(len(name) - len(q)) > 5:
                continue
            r = SequenceMatcher(None, q, name).ratio()
            if r > best_r:
                best, best_r = name, r
        if best and best_r >= 0.88 and chip_alias_ok(q, best):
            return best
        return None

    def lookup_drug(self, text: str) -> tuple[str, str]:
        raw = light_clean(text)
        if not raw:
            return text.strip(), "heuristic"
        if raw in self.drug_targets:
            return raw, "drug_alias"
        for cand in (raw, raw.upper(), drug_core(raw), drug_core(raw).upper()):
            if not cand:
                continue
            hit = self.drug_alias.get(cand) or self.drug_alias_norm.get(cand.upper())
            if hit:
                return hit, "drug_alias"
        core = drug_core(raw)
        return core or raw, "heuristic"

    def lookup_surgery(self, text: str) -> tuple[str, str]:
        raw = text.strip()
        if raw in self.surgery_targets:
            return raw, "surgery_alias"
        core = surgery_core(raw)
        for cand in (raw, core, light_clean(raw)):
            if not cand:
                continue
            hit = self.surgery_alias.get(cand) or self.surgery_alias_norm.get(surgery_core(cand))
            if hit:
                return hit, "surgery_alias"
        # Rule-based collapses for common procedure tokens.
        up = core.upper()
        if "MILES" in up:
            return "腹会阴联合直肠癌根治术", "surgery_alias"
        if "DIXON" in up and ("直肠" in core or "DIXON" in up):
            if "腹腔镜" in core:
                return "腹腔镜直肠癌Dixon术", "surgery_alias"
            return "直肠癌Dixon术", "surgery_alias"
        if re.search(r"\bTACE\b", up):
            return "TACE", "surgery_alias"
        if core != raw:
            return core, "surgery_norm"
        return raw, "heuristic"

    def longest_prefix_hit(self, text: str) -> str | None:
        if len(text) < 4:
            return None
        best = ""
        for name in self.by_head.get(text[:2], ()):
            if text.startswith(name) and len(name) >= 4 and len(name) > len(best):
                leftover = text[len(name) :]
                leftover = strip_grade(leftover)
                if leftover and not LEFT_OK.match(leftover):
                    continue
                best = name
        return best or None

    def resolve_name(self, name: str) -> str | None:
        if name in self.names:
            return name
        compact = re.sub(r"\s+", "", name)
        return self.compact.get(compact)

    def preferred_code(self, name: str) -> str:
        codes = self.names.get(name, [])
        if not codes:
            return ""
        return min(codes, key=lambda c: (len(c), c))

    def metastasis_hit(self, text: str) -> str | None:
        """肠癌肝转移 / 肝多发转移 / 淋巴结转移 → <site>继发恶性肿瘤."""
        pos = text.rfind("转移")
        if pos <= 0:
            return None
        head = re.sub(r"(多发|多处|广泛|弥漫性?|内|性|灶|伴)+$", "", text[:pos])
        for k in range(min(10, len(head)), 0, -1):
            name = head[-k:] + "继发恶性肿瘤"
            if name in self.names:
                return name
        return None

    def site_morph_hit(self, text: str) -> str | None:
        """<site>…癌 → <site>恶性肿瘤; <site>…腺瘤 → <site>良性肿瘤 (site taken from the start)."""
        if "转移" in text or "继发" in text or "原位" in text:
            return None
        if CANCER_RE.search(text):
            sites, suffix = self.cancer_sites, "恶性肿瘤"
        elif "腺瘤" in text:
            sites, suffix = self.benign_sites, "良性肿瘤"
        else:
            return None
        site = next(
            (
                text[:k]
                for k in range(min(6, len(text) - 1), 0, -1)
                if text[:k] in sites and text[k] not in "状样"
            ),
            None,
        )
        if site is None:
            return None
        # 肝曲横结肠腺瘤 / 阑尾开口横结肠腺瘤: the site nearest the morphology is the head noun
        head = re.split(r"癌|Ca|CA|腺瘤", text, maxsplit=1)[0]
        if "交界" in head:
            joint = head.split("交界", 1)[0] + "连接处" + suffix
            if joint in self.names:
                return joint
        i = len(site)
        while i < len(head) - 1:
            later = next((head[i : i + k] for k in range(6, 1, -1) if head[i : i + k] in sites), None)
            if later:
                # 乳头状 / 管状 describe morphology, not a site
                descriptive = head[i + len(later) : i + len(later) + 1] in ("状", "样")
                if not descriptive and later not in site and site not in later:
                    site = later
                i += len(later)
            else:
                i += 1
        return site + suffix

    def lookup_disease(self, text: str) -> tuple[str, str]:
        raw = text.strip()
        if _SETTINGS.get("calibration_mode") == "abstain":
            from calibrated_normalize import calibrated_lookup

            decision = calibrated_lookup(
                raw,
                model_path=Path(_SETTINGS["calibration_model"]),
                include_neural=bool(_SETTINGS.get("calibration_neural", True)),
            )
            return decision.name, decision.source
        if _SETTINGS.get("disease_rules_only"):
            out = rule_disease(raw)
            return out, ("rule" if out != raw else "heuristic")
        exact = self.resolve_name(raw)
        if exact:
            return exact, "icd_exact"
        new_rules = _SETTINGS.get("use_new_rules", True)
        for cand in type_prefix_variants(raw) if new_rules else ():
            hit = self.resolve_name(cand)
            if hit:
                return hit, "icd_variant"

        sup_mode = _SETTINGS.get("supervised_cdn", "off")
        if sup_mode != "off":
            from cdn_linker import get_cdn_linker

            if get_cdn_linker().is_label(raw):
                return raw, "cdn_sup"
            if sup_mode == "primary":
                sup = self.supervised_names(raw)
                if sup:
                    return sup[0], "cdn_sup"

        llm_mode = _SETTINGS.get("llm_cdn", "off")
        if llm_mode == "primary":
            names = llm_names(raw)
            if names:
                return names[0], "llm"

        # （胃）溃疡型中分化腺癌: keep the leading bracketed site that clean_surface would drop.
        m = LEAD_PAREN_RE.match(light_clean(raw)) if new_rules else None
        if m:
            site = re.split(r"[、,，及和]", m.group(1))[0]
            if site in self.cancer_sites or site in self.benign_sites:
                return self.lookup_disease(site + light_clean(raw)[m.end() :])

        cleaned = clean_surface(raw)
        bases = [cleaned]
        graded = strip_grade(cleaned)
        if graded != cleaned:
            bases.append(graded)

        if new_rules and "转移" in cleaned:
            hit = self.metastasis_hit(cleaned)
            if hit:
                return hit, "icd_rule"

        # 1) ICD dictionary first (exact / variant / longest prefix).
        for base in bases:
            for cand in icd_variants(base):
                hit = self.resolve_name(cand)
                if hit:
                    source = "icd_exact" if hit in (cleaned, graded, base, raw) else "icd_variant"
                    return hit, source
            hit = self.longest_prefix_hit(base)
            if hit:
                return hit, "icd_retrieve"
            for cand in icd_variants(base):
                hit = self.longest_prefix_hit(cand)
                if hit and hit != cand:
                    return hit, "icd_retrieve"

        for base in bases if new_rules else ():
            hit = self.site_morph_hit(base)
            if hit:
                return hit, "icd_rule"

        # 1b) Curated long-tail aliases (CCKS-oriented, ICD-validated targets).
        curated_hit = self.lookup_curated(raw, cleaned)
        if curated_hit:
            return curated_hit, "curated"

        # 2) Unsupervised ICD retrieval.
        try:
            from bert_linker import get_linker

            linker = get_linker(_SETTINGS.get("retriever", "roberta"))
            seen_q: set[str] = set()
            for base in bases:
                for q in icd_variants(base):
                    if q in seen_q or len(q) < 4:
                        continue
                    seen_q.add(q)
                    hit = linker.retrieve(q)
                    if hit:
                        return hit[0], "bert_retrieve"
        except (FileNotFoundError, ImportError):
            pass

        # 3) CHIP-CDN only as fallback, with conservative acceptance filters.
        if self.alias:
            alias_hits: list[str] = []
            for key in (raw, cleaned):
                if not key:
                    continue
                if key in self.alias:
                    alias_hits.append(self.alias[key])
                hit2 = self.alias_clean.get(key)
                if hit2:
                    alias_hits.append(hit2)
            seen: set[str] = set()
            for alias_hit in alias_hits:
                if alias_hit in seen:
                    continue
                seen.add(alias_hit)
                resolved = self.resolve_name(alias_hit) or alias_hit
                try:
                    alias_safe = chip_alias_ok(raw, resolved) or chip_alias_ok(cleaned, resolved)
                except ImportError:
                    alias_safe = False
                if alias_safe:
                    return resolved, "chip_cdn"

        # 4) Conservative fuzzy ICD (same head, high similarity, site/morph OK).
        fuzzy = self.fuzzy_icd(cleaned) or self.fuzzy_icd(raw)
        if fuzzy:
            return fuzzy, "fuzzy_icd"

        if sup_mode == "fallback":
            sup = self.supervised_names(raw)
            if sup:
                return sup[0], "cdn_sup"

        if llm_mode == "fallback":
            names = llm_names(raw)
            if names:
                return names[0], "llm"

        fb_mode = _SETTINGS.get("retriever_fallback", "off")
        if fb_mode != "off":
            try:
                from bert_linker import get_linker

                linker = get_linker(_SETTINGS.get("retriever", "roberta"))
                hit = linker.top1(strip_grade(cleaned) or cleaned or raw)
                if hit and (fb_mode == "top1" or hit[1] >= linker.min_score):
                    return hit[0], "retrieve_fallback"
            except (FileNotFoundError, ImportError):
                pass

        return cleaned or raw, "heuristic"

    def supervised_names(self, raw: str) -> list[str]:
        """Also try the cleaned surface, so a kept-as-is canonical re-links to the same name."""
        from cdn_linker import get_cdn_linker

        linker = get_cdn_linker()
        min_p = _SETTINGS.get("supervised_min_p", 0.5)
        return linker.link(raw, min_p) or linker.link(clean_surface(raw) or raw, min_p)

    def lookup_disease_multi(self, text: str) -> list[tuple[str, str]]:
        """Compound mentions (胆囊结石伴慢性胆囊炎) → every linked part, whole-text hit first.

        Strong separators (、，；+ 伴 合并) always split; 及/和 only when the whole text is unlinked.
        A mention that is itself an ICD name is never split (肾和输尿管结石).
        """
        if text not in self._multi_cache:
            self._multi_cache[text] = self._lookup_disease_multi(text)
        return self._multi_cache[text]

    def _lookup_disease_multi(self, text: str) -> list[tuple[str, str]]:
        whole, src = self.lookup_disease(text)
        if src in ("cdn_sup", "llm"):
            names = self.supervised_names(text.strip()) if src == "cdn_sup" else llm_names(text.strip())
            if whole in names and _SETTINGS.get("use_split"):
                return [(n, src) for n in names]
            return [(whole, src)]
        if not _SETTINGS.get("use_split") or _SETTINGS.get("disease_rules_only"):
            return [(whole, src)]
        body = light_clean(text)
        if self.resolve_name(body):
            return [(whole, src)]
        body = re.sub(r"[(（][^)）]*[)）]", "", body)
        if whole.endswith("继发恶性肿瘤") and body.endswith("转移"):
            # 左下肺及肝转移: every listed site is a metastasis
            parts = [p if "转移" in p else p + "转移" for p in WEAK_SPLIT_RE.split(body) if p]
        else:
            splitter = WEAK_SPLIT_RE if src == "heuristic" else STRONG_SPLIT_RE
            parts = [p for p in splitter.split(body) if len(p) >= 3]
        if len(parts) < 2:
            return [(whole, src)]
        out: list[tuple[str, str]] = [] if src == "heuristic" else [(whole, src)]
        for part in parts:
            name, s = self.lookup_disease(part)
            if s != "heuristic" and all(name != o for o, _ in out):
                out.append((name, s))
        return out or [(whole, src)]


LLM_MAP = ROOT / "outputs" / "llm_norm" / "map.json"
LLM_MISSING: set[str] = set()
_LLM: dict[str, list[str]] | None = None


def llm_names(raw: str) -> list[str]:
    """LLM-chosen ICD names for a mention; [] = keep as is.

    Mentions absent from the map are collected in LLM_MISSING so src/llm_norm_sample.py can
    send them to the LLM.
    """
    global _LLM
    if _LLM is None:
        _LLM = json.loads(LLM_MAP.read_text(encoding="utf-8")) if LLM_MAP.exists() else {}
    if raw not in _LLM:
        LLM_MISSING.add(raw)
        return []
    return _LLM[raw]


def reload_llm_map() -> None:
    global _LLM
    _LLM = None


def get_index() -> DictIndex:
    global _INDEX
    if _INDEX is None:
        _INDEX = DictIndex.load()
    return _INDEX


def entity_key(text: str, label: str) -> str:
    if label == "疾病和诊断":
        return get_index().lookup_disease_multi(text)[0][0]
    if label == "药物":
        return get_index().lookup_drug(text)[0]
    if label == "手术":
        return get_index().lookup_surgery(text)[0]
    if label == "实验室检验":
        return lab_core(text)
    return light_clean(text)


def lookup_with_source(text: str, label: str) -> tuple[str, str]:
    if label == "疾病和诊断":
        return get_index().lookup_disease_multi(text)[0]
    if label == "药物":
        return get_index().lookup_drug(text)
    if label == "手术":
        return get_index().lookup_surgery(text)
    return entity_key(text, label), "heuristic"


def icd_code_for(text: str, label: str) -> str:
    if label != "疾病和诊断":
        return ""
    idx = get_index()
    name, _src = idx.lookup_disease(text)
    return idx.preferred_code(name)


def canonical_names(text: str, label: str) -> tuple[str, ...]:
    """All canonical names a mention maps to (several only for split compound diagnoses)."""
    if label == "疾病和诊断":
        return tuple(name for name, _ in get_index().lookup_disease_multi(text))
    if label == "药物" and _SETTINGS.get("use_drug_class"):
        from drug_class import classify
        classes = classify(text)
        if classes:
            return classes
    return (entity_key(text, label),)


def needs_context(text: str, label: str, names: tuple[str, ...]) -> bool:
    """Site-less histology left unlinked; resolve with the document's primary cancer instead."""
    return (
        _SETTINGS.get("use_context", False)
        and not _SETTINGS.get("disease_rules_only")
        and label == "疾病和诊断"
        and len(names) == 1
        and "恶性肿瘤" not in names[0]
        and histology_without_site(text)
    )


def build_canonical_map(
    names_by_type: dict[str, dict[str, int]],
) -> dict[tuple[str, str], str | tuple[str, ...]]:
    """Mention → canonical name; a tuple when a compound diagnosis is split."""
    mapping: dict[tuple[str, str], str | tuple[str, ...]] = {}
    for label, freq in names_by_type.items():
        for name in freq:
            names = canonical_names(name, label)
            mapping[(label, name)] = names[0] if len(names) == 1 else names
    return mapping


def coverage_report(names_by_type: dict[str, dict[str, int]]) -> dict:
    rows = []
    counts = {
        "chip_cdn": 0,
        "curated": 0,
        "fuzzy_icd": 0,
        "icd_exact": 0,
        "icd_variant": 0,
        "icd_retrieve": 0,
        "icd_rule": 0,
        "bert_retrieve": 0,
        "heuristic": 0,
    }
    n_split = 0
    for name in names_by_type.get("疾病和诊断", {}):
        canon, source = lookup_with_source(name, "疾病和诊断")
        if len(canonical_names(name, "疾病和诊断")) > 1:
            n_split += 1
        counts[source] = counts.get(source, 0) + 1
        rows.append({"from": name, "to": canon, "source": source})
    n = len(rows) or 1

    drug_rows = []
    drug_counts = {"drug_alias": 0, "heuristic": 0}
    for name in names_by_type.get("药物", {}):
        canon, source = lookup_with_source(name, "药物")
        drug_counts[source] = drug_counts.get(source, 0) + 1
        drug_rows.append({"from": name, "to": canon, "source": source})
    n_drug = len(drug_rows) or 1

    surg_rows = []
    surg_counts = {"surgery_alias": 0, "surgery_norm": 0, "heuristic": 0}
    for name in names_by_type.get("手术", {}):
        canon, source = lookup_with_source(name, "手术")
        surg_counts[source] = surg_counts.get(source, 0) + 1
        surg_rows.append({"from": name, "to": canon, "source": source})
    n_surg = len(surg_rows) or 1

    return {
        "n_disease_surfaces": len(rows),
        "n_chip_cdn_alias": len(get_index().alias),
        "n_icd_names": len(get_index().names),
        "source_counts": counts,
        "n_split_compound": n_split,
        "icd_hit_rate": (n - counts["heuristic"] - counts.get("rule", 0)) / n if rows else 0.0,
        "examples": [r for r in rows if r["from"] != r["to"]][:25],
        "misses": [r for r in rows if r["source"] == "heuristic"][:15],
        "drug": {
            "n_drug_surfaces": len(drug_rows),
            "n_drug_alias": len(get_index().drug_alias),
            "source_counts": drug_counts,
            "alias_hit_rate": drug_counts.get("drug_alias", 0) / n_drug,
            "examples": [r for r in drug_rows if r["from"] != r["to"]][:20],
        },
        "surgery": {
            "n_surgery_surfaces": len(surg_rows),
            "n_surgery_alias": len(get_index().surgery_alias),
            "source_counts": surg_counts,
            "alias_hit_rate": (
                (surg_counts.get("surgery_alias", 0) + surg_counts.get("surgery_norm", 0)) / n_surg
            ),
            "examples": [r for r in surg_rows if r["from"] != r["to"]][:20],
        },
    }
