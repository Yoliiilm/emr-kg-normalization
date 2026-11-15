"""Load BIO files and encode character sequences for Chinese NER."""

from __future__ import annotations

from pathlib import Path

try:
    import torch
    from torch.utils.data import Dataset
except ModuleNotFoundError:  # BIO graph utilities do not require the model-training dependency.
    torch = None

    class Dataset:
        pass


def load_label_list(path: Path) -> list[str]:
    labels = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if labels[0] != "O":
        raise ValueError("label_list.txt must start with O")
    return labels


def load_bio_docs(path: Path) -> list[tuple[list[str], list[str]]]:
    docs: list[tuple[list[str], list[str]]] = []
    chars: list[str] = []
    tags: list[str] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                if chars:
                    docs.append((chars, tags))
                    chars, tags = [], []
                continue
            parts = line.split(" ")
            if len(parts) < 2:
                ch, tag = parts[0], "O"
            else:
                ch, tag = parts[0], parts[-1]
            if not ch:
                ch = " "
            chars.append(ch)
            tags.append(tag)
    if chars:
        docs.append((chars, tags))
    return docs


def chunk_doc(chars: list[str], tags: list[str], max_chars: int) -> list[tuple[list[str], list[str]]]:
    if len(chars) <= max_chars:
        return [(chars, tags)]
    chunks = []
    for start in range(0, len(chars), max_chars):
        chunks.append((chars[start : start + max_chars], tags[start : start + max_chars]))
    return chunks


class NERCharDataset(Dataset):
    def __init__(self, docs, tokenizer, label2id: dict[str, int], max_len: int = 512):
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_len = max_len
        self.max_chars = max_len - 2
        self.examples: list[tuple[list[str], list[str]]] = []
        self.origins: list[tuple[int, int, int]] = []
        for doc_idx, (chars, tags) in enumerate(docs):
            if not chars:
                chars, tags = ["。"], ["O"]
            for start in range(0, len(chars), self.max_chars):
                chunk_chars = chars[start : start + self.max_chars]
                chunk_tags = tags[start : start + self.max_chars]
                self.examples.append((chunk_chars, chunk_tags))
                self.origins.append((doc_idx, start, len(chunk_chars)))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if torch is None:
            raise RuntimeError("PyTorch is required to create NER training tensors")
        chars, tags = self.examples[idx]
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        pad_id = self.tokenizer.pad_token_id
        unk_id = self.tokenizer.unk_token_id

        input_ids = [cls_id]
        labels = [0]
        crf_mask = [1]  # pytorch-crf requires mask[0] == True
        for ch, tag in zip(chars, tags):
            tid = self.tokenizer.convert_tokens_to_ids(ch)
            if tid is None:
                tid = unk_id
            input_ids.append(tid)
            labels.append(self.label2id.get(tag, 0))
            crf_mask.append(1)
        input_ids.append(sep_id)
        labels.append(0)
        crf_mask.append(1)

        attn = [1] * len(input_ids)
        pad_n = self.max_len - len(input_ids)
        if pad_n > 0:
            input_ids += [pad_id] * pad_n
            attn += [0] * pad_n
            labels += [0] * pad_n
            crf_mask += [0] * pad_n

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "crf_mask": torch.tensor(crf_mask, dtype=torch.bool),
        }


def bio_to_entities(chars: list[str], tags: list[str]) -> list[dict]:
    ents = []
    i = 0
    n = len(chars)
    while i < n:
        tag = tags[i]
        if tag.startswith("B-"):
            label = tag[2:]
            j = i + 1
            while j < n and tags[j] == f"I-{label}":
                j += 1
            ents.append(
                {
                    "text": "".join(chars[i:j]),
                    "label": label,
                    "start": i,
                    "end": j,
                }
            )
            i = j
        else:
            i += 1
    return ents
