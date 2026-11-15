"""RoBERTa / MacBERT + BiLSTM + CRF for character-level Chinese NER."""

from __future__ import annotations

import torch
from torch import nn
from torchcrf import CRF
from transformers import AutoModel


class BertBiLstmCrf(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_labels: int,
        lstm_hidden: int = 256,
        lstm_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size
        self.lstm = nn.LSTM(
            hidden,
            lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_hidden * 2, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def emissions(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        seq, _ = self.lstm(out.last_hidden_state)
        return self.classifier(self.dropout(seq))

    def forward(self, input_ids, attention_mask, labels=None, crf_mask=None):
        emm = self.emissions(input_ids, attention_mask)
        mask = (crf_mask if crf_mask is not None else attention_mask).to(dtype=torch.bool)
        if mask.size(1) == 0:
            raise RuntimeError("empty CRF mask")
        if not mask[:, 0].all():
            mask = mask.clone()
            mask[:, 0] = True
        if labels is not None:
            return -self.crf(emm, labels, mask=mask, reduction="mean")
        return self.crf.decode(emm, mask=mask)
