"""Lightweight clinical compatibility gates shared by normalization and retrieval."""

ORGANS = (
    "乙状结肠", "升结肠", "横结肠", "降结肠", "十二指肠", "直肠", "结肠",
    "胃窦", "胃底", "贲门", "食管", "卵巢", "子宫", "宫颈", "膀胱",
    "肝脏", "肝", "肺", "肾", "脾", "肩", "膝", "胃", "脑",
)
KEY_MORPHS = (
    "肉瘤", "腺癌", "鳞癌", "癌", "肝炎", "肝硬化", "梗死", "梗阻",
    "出血", "感染", "囊肿", "炎",
)


def morph_ok(query: str, hit: str) -> bool:
    for morph in KEY_MORPHS:
        if morph in query and morph not in hit:
            return False
    for letter in ("甲", "乙", "丙", "丁", "戊"):
        tag = letter + "型"
        if tag in query and tag not in hit and letter + "肝" not in hit:
            if letter in query and letter not in hit:
                return False
    return True


def site_ok(query: str, hit: str) -> bool:
    if "原发" in query and "继发" in hit:
        return False
    if "继发" in hit and "继发" not in query and "转移" not in query:
        return False
    if "非萎缩" in query and "萎缩" in hit:
        return False
    if "非霍奇金" in query and "霍奇金" in hit and "非霍奇金" not in hit:
        return False
    if "下" in query and "上" in hit and "下" not in hit:
        return False
    if "上" in query and "下" in hit and "上" not in hit:
        return False
    if "陈旧" in hit and "陈旧" not in query:
        return False
    if "消化道" in query and "上" not in query and "下" not in query and ("上" in hit or "下" in hit):
        return False
    if "急性" in hit and "急性" not in query:
        return False
    if "慢性" in hit and "慢性" not in query:
        return False
    query_organs = [organ for organ in ORGANS if organ in query]
    if hit in {"恶性未分化肿瘤", "恶性肿瘤", "癌", "未分化癌"}:
        return False
    if not query_organs and ("分化" in query or "癌" in query or "恶性肿瘤" in query or query.endswith("瘤")):
        return False
    if "内膜" in hit and "内膜" not in query:
        return False
    if "冠" in hit and "冠" not in query:
        return False
    for organ in query_organs:
        if organ not in hit:
            return False
    for organ in ORGANS:
        if organ in hit and organ not in query:
            return False
    return True
