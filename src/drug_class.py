"""Drug class terms (抗生素, 降压药物, 止疼、活血化瘀等药物) → WHO ATC therapeutic classes.

A class term names a group of drugs, not a drug, so no drug dictionary can resolve it; what can be done
is to give it a standard class node. The lexicon below is written from the ATC level-2/3 groups and their
usual Chinese clinical wording, not from any dataset. Only unambiguous wordings are listed: 抗炎/消炎
(antibacterial in many Chinese notes, NSAID in others), 营养神经, 活血化瘀, 软组织修复, 对症 and 中药
have no single ATC group and stay as they are.

A mention is a class term if, after stripping the leading verb (应用/口服/静点…), it ends in 药物/药/类/
拮抗剂/抑制剂/阻滞剂 or is one of the bare class nouns (抗生素, 激素). It is split on 、，,及和与 and 等;
every part must match some class, otherwise the mention is left unmapped (a partly understood phrase is
not rewritten). Several matched classes → several class nodes, like a split compound diagnosis.
"""

from __future__ import annotations

import re

ATC: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("J01D", "头孢菌素类", ("头孢",)),
    ("J01C", "青霉素类", ("青霉素",)),
    ("J01F", "大环内酯类", ("大环内酯",)),
    ("J01M", "喹诺酮类", ("喹诺酮", "沙星类")),
    ("J01E", "磺胺类", ("磺胺",)),
    ("J01G", "氨基糖苷类", ("氨基糖苷",)),
    ("J01", "全身用抗菌药", ("抗生素", "抗菌", "抗感染")),
    ("C02", "抗高血压药", ("降压", "调节血压", "抗高血压", "降血压")),
    ("A10", "降血糖药", ("降糖", "降血糖")),
    ("C10", "调节血脂药", ("调节血脂", "调脂", "降脂", "降血脂")),
    ("C01", "心脏治疗药", ("治疗冠心病", "抗心律失常", "强心")),
    ("B01", "抗血栓药", ("抗凝", "抗血小板", "抗血栓")),
    ("B02", "止血药", ("止血",)),
    ("B05", "补液和营养支持", ("补液", "营养支持", "肠外营养")),
    ("A02", "抑酸护胃药", ("质子泵", "抑酸", "护胃", "胃药")),
    ("A05", "保肝药", ("保肝", "护肝")),
    ("H02", "全身用皮质激素", ("激素", "糖皮质")),
    ("M03", "肌肉松弛药", ("肌松", "肌肉松弛", "缓解肌肉紧张")),
    ("M05", "抗骨质疏松药", ("抗骨质疏松",)),
    ("N02", "镇痛药", ("止痛", "止疼", "镇痛")),
    ("N05", "镇静催眠药", ("镇静", "催眠", "安眠")),
    ("R05", "祛痰镇咳药", ("祛痰", "化痰", "止咳", "镇咳")),
)
BARE = {"抗生素", "激素"}
LEAD_RE = re.compile(r"^(应用|使用|给予|予以|予|口服|静点|静滴|静脉|注射|外用)+")
TAIL_RE = re.compile(r"(类药物|类药|药物|药|类|拮抗剂|抑制剂|阻滞剂)$")
SPLIT_RE = re.compile(r"[、，,及和与]|等")


def is_class_term(text: str) -> bool:
    t = LEAD_RE.sub("", text.strip())
    return t in BARE or bool(TAIL_RE.search(t))


def classify(text: str) -> tuple[str, ...]:
    """ATC class nodes for a class term, () if it is not one or any part is not understood."""
    if not is_class_term(text):
        return ()
    t = LEAD_RE.sub("", text.strip())
    out: list[str] = []
    for part in (p for p in SPLIT_RE.split(t) if p.strip()):
        hits = []
        for code, name, keys in ATC:
            if any(k in part for k in keys) and not any(h.startswith(code) or code.startswith(h) for h in hits):
                hits.append(code)
        if not hits:
            return ()
        out += [f"ATC {code} {name}" for code, name, _ in ATC if code in hits]
    return tuple(dict.fromkeys(out))
