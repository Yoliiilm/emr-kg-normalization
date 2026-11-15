"""Does the normalization gain on CCKS2017 come back once NER is adapted to the new hospital?

REPORT 3.8: with annotated entities the top-5 dedup gain on CCKS2017 matches CCKS2019, with the
CCKS2019-trained NER it halves. Here the CCKS2019 model (outputs/models/best.pt) is fine-tuned on
CCKS2017 by two-fold cross-fitting: split the 300 notes in half, fine-tune on one half, predict
the other, swap. Every note is predicted by a model that never saw it. CCKS2017 is annotated for
disease / anatomy / drug / surgery only, so the other types are learned as O; scoring uses the
shared types as in eval_ccks2017_graph.

  python src/finetune_ccks2017.py          # → data/processed/ccks2017_pred_ft.bio,
                                           #   outputs/metrics/ccks2017_finetune.{txt,json}
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import torch
from seqeval.metrics import classification_report, f1_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from eval_bootstrap import cluster_bootstrap, sign_flip_test
from eval_ccks2017_graph import GOLD_BIO, PRED_BIO, RELS, rows_by_key, shared_only
from eval_g_raw_vs_g_norm import build_graphs
from ner_data import NERCharDataset, load_bio_docs
from ner_model import BertBiLstmCrf
from normalize import configure_normalization
from train_ner import build_optimizer, predict_docs, set_seed, write_bio

FT_BIO = ROOT / "data" / "processed" / "ccks2017_pred_ft.bio"
OUT = ROOT / "outputs" / "metrics"
EPOCHS = 4
LR_BERT = 1e-5
LR_HEAD = 5e-4
BATCH = 16
SEED = 0


def load_base(device):
    ckpt = torch.load(ROOT / "outputs" / "models" / "best.pt", map_location=device, weights_only=False)
    a = ckpt["args"]
    model = BertBiLstmCrf(a["model_name"], num_labels=len(ckpt["label2id"]),
                          lstm_hidden=a["lstm_hidden"], dropout=a["dropout"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    return model, ckpt["label2id"], a


def finetune_predict(train_docs, test_docs, device):
    set_seed(SEED)
    model, label2id, a = load_base(device)
    id2label = {i: lab for lab, i in label2id.items()}
    tok = AutoTokenizer.from_pretrained(a["model_name"])
    loader = DataLoader(NERCharDataset(train_docs, tok, label2id, max_len=a["max_len"]),
                        batch_size=BATCH, shuffle=True, num_workers=0)
    opt = build_optimizer(model, LR_BERT, LR_HEAD, 0.01)
    total = len(loader) * EPOCHS
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)
    for ep in range(EPOCHS):
        model.train()
        loss_sum = 0.0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            opt.zero_grad()
            loss = model(batch["input_ids"], batch["attention_mask"], labels=batch["labels"], crf_mask=batch["crf_mask"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            loss_sum += float(loss.item())
        print(f"  epoch {ep + 1}/{EPOCHS} loss {loss_sum / len(loader):.4f}", flush=True)
    return predict_docs(model, tok, test_docs, id2label, a["max_len"], device)


def graph_gain(gold_docs, pred_docs) -> dict:
    configure_normalization()
    g = build_graphs(gold_docs, pred_docs)
    out = {}
    for rel in RELS:
        vals = list(rows_by_key(g, rel).values())
        bs = cluster_bootstrap(vals, "canonical")
        out[rel] = {"n": len(vals),
                    "f1_raw": sum(r["f_raw"] for r in vals) / len(vals),
                    "f1_norm": sum(r["f_norm"] for r in vals) / len(vals),
                    "ci95": bs["delta_ci95"],
                    "p": sign_flip_test([r["delta"] for r in vals]),
                    "better": sum(r["delta"] > 1e-9 for r in vals),
                    "worse": sum(r["delta"] < -1e-9 for r in vals)}
    return out


def main() -> None:
    gold = load_bio_docs(GOLD_BIO)
    zero = load_bio_docs(PRED_BIO)
    if FT_BIO.exists():
        ft = load_bio_docs(FT_BIO)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        idx = list(range(len(gold)))
        random.Random(SEED).shuffle(idx)
        folds = [sorted(idx[: len(idx) // 2]), sorted(idx[len(idx) // 2:])]
        tags = [None] * len(gold)
        for k, test_idx in enumerate(folds):
            train_idx = folds[1 - k]
            print(f"fold {k + 1}: train {len(train_idx)} test {len(test_idx)}", flush=True)
            pred = finetune_predict([gold[i] for i in train_idx], [gold[i] for i in test_idx], device)
            for i, t in zip(test_idx, pred):
                tags[i] = t
        ft = [(chars, t) for (chars, _), t in zip(gold, tags)]
        write_bio(FT_BIO, ft)

    y_gold = [shared_only(t) for _, t in gold]
    res = {"ner": {}, "graph": {}}
    L = ["CCKS2017：NER 在新医院微调后，标准化的收益能不能回来", "=" * 72,
         f"两折交叉：300 份对半分，从 CCKS2019 模型出发在一半上微调 {EPOCHS} 轮、预测另一半，再交换；每份病历都由没见过它的模型预测。",
         "标准答案与 2.10 节相同（人工标注实体建图，固定用本方法归一化）。", ""]
    for name, title, docs in (("zero_shot", "直接用 CCKS2019 模型（2.10 节）", zero), ("finetuned", "在 CCKS2017 上微调", ft)):
        y = [shared_only(t) for _, t in docs]
        res["ner"][name] = {"f1": f1_score(y_gold, y), "report": classification_report(y_gold, y, digits=3, output_dict=True)}
        L += [f"【{title}】", classification_report(y_gold, y, digits=3)]
        gg = res["graph"][name] = graph_gain(gold, docs)
        for rel in RELS:
            e = gg[rel]
            L.append(f"  {rel}  查询 {e['n']}  不标准化 {e['f1_raw']:.3f} → 本方法 {e['f1_norm']:.3f}  "
                     f"差 {e['f1_norm'] - e['f1_raw']:+.3f} [{e['ci95'][0]:+.3f}, {e['ci95'][1]:+.3f}]  "
                     f"变好/变差 {e['better']}/{e['worse']}  p={e['p']:.4f}")
        L.append("")
    text = "\n".join(L) + "\n"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ccks2017_finetune.txt").write_text(text, encoding="utf-8")
    (OUT / "ccks2017_finetune.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
