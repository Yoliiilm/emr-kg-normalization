"""Rule-based assertion status of disease mentions in Chinese EMRs (NegEx-style).

A mention is negated (否认高血压 / 未见骨折 / 肝炎（-）), uncertain (不排除肝癌 / 胃癌？/ 疑似肺炎),
family (母亲患糖尿病) or affirmed. The scope of a pre-cue is the rest of its clause (clauses end at
。；，！？ and line breaks; 、 / 及 / 和 lists stay inside), cut short by a terminator such as 出现 / 诊断 / 示.
The last cue before the mention wins.

filter_docs() retags negated (and optionally uncertain / family) disease mentions as O, so the
same graph code builds the filtered graph for both gold and predicted entities.
"""

from __future__ import annotations

import re

DISEASE = "疾病和诊断"
BOUNDARY = re.compile(r"[。；;，,！!？?\n]")
UNCERTAIN_PRE = re.compile(r"不排除|不能排除|不除外|未除外|待排除?|疑似|可疑|疑为")
NEGATED_PRE = re.compile(r"否认|未见|未发现|未闻及|未触及|未诉|从未|没有|已?排除|已?除外|不伴|"
                         r"无(?!痛|创|菌|水|功能|明显诱因|诱因|意中)")
FAMILY_PRE = re.compile(r"父亲|母亲|父母|兄弟|姐妹|兄|弟|姐|妹|子女|儿子|女儿|祖父|祖母|外祖|家族")
TERMINATOR = re.compile(r"出现|诊断|确诊|提示|考虑|[示但因予行后]")
UNCERTAIN_POST = re.compile(r"^[^，,。；;]{0,2}([?？]|可能(?!性大)|待排|不除外|待查)")
NEGATED_POST = re.compile(r"^[^，,。；;]{0,3}(（-）|\(-\)|（－）|阴性|已排除|已除外)")


def classify(text: str, start: int, end: int) -> str:
    """Assertion of text[start:end]: affirmed | negated | uncertain | family."""
    left = start
    while left > 0 and not BOUNDARY.match(text[left - 1]):
        left -= 1
    pre = text[left:start]
    post = text[end:end + 8]
    if NEGATED_POST.search(post):
        return "negated"
    if UNCERTAIN_POST.search(post):
        return "uncertain"

    unc = [(m.start(), m.end(), "uncertain") for m in UNCERTAIN_PRE.finditer(pre)]
    neg = [(m.start(), m.end(), "negated") for m in NEGATED_PRE.finditer(pre)
           if not any(a <= m.start() < b for a, b, _ in unc)]
    fam = [(m.start(), m.end(), "family") for m in FAMILY_PRE.finditer(pre)]
    cues = sorted(unc + neg + fam)
    if not cues:
        return "affirmed"
    _, cue_end, kind = cues[-1]
    if TERMINATOR.search(pre[cue_end:]):
        return "affirmed"
    if kind == "family" and any(c[2] == "negated" for c in cues):
        return "negated"
    return kind


def mentions(chars: list[str], tags: list[str], label: str = DISEASE):
    """(start, end) of every mention with the given label."""
    i, n = 0, len(tags)
    while i < n:
        if tags[i] == f"B-{label}":
            j = i + 1
            while j < n and tags[j] == f"I-{label}":
                j += 1
            yield i, j
            i = j
        else:
            i += 1


def filter_docs(docs, drop: tuple[str, ...] = ("negated",), label: str = DISEASE):
    """Copy of docs with mentions whose assertion is in drop retagged as O; also returns the counts."""
    out, counts = [], {"affirmed": 0, "negated": 0, "uncertain": 0, "family": 0}
    for chars, tags in docs:
        text = "".join(chars)
        new = list(tags)
        for s, e in mentions(chars, tags, label):
            a = classify(text, s, e)
            counts[a] += 1
            if a in drop:
                new[s:e] = ["O"] * (e - s)
        out.append((chars, new))
    return out, counts
