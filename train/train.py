#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import time
import json
import random
import argparse
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import T5Tokenizer, T5EncoderModel
from transformers import AutoTokenizer, AutoModel

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score,
    roc_auc_score, average_precision_score,
    matthews_corrcoef, confusion_matrix
)
from sklearn.preprocessing import label_binarize

import matplotlib.pyplot as plt


# ============================================================
# 0) Utils
# ============================================================
def set_seed(seed: int = 42, deterministic: bool = True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def now_ts():
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


# ============================================================
# 1) Attention pooling
# ============================================================
def attention_pooling(hidden: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    """
    hidden:    (B, L, D)
    attn_mask: (B, L)  (1=valid, 0=pad)
    """
    weights = hidden.mean(dim=-1)  # (B, L)
    weights = weights.masked_fill(attn_mask == 0, -1e9)
    weights = torch.softmax(weights, dim=1)
    pooled = torch.sum(hidden * weights.unsqueeze(-1), dim=1)  # (B, D)
    return pooled


# ============================================================
# 2) Robust hidden extraction (FIX)
# ============================================================
def _first_3d_tensor_from_tuple(obj) -> Optional[torch.Tensor]:
    """From tuple/list outputs, find the first tensor with dim==3."""
    if not isinstance(obj, (tuple, list)):
        return None
    for x in obj:
        if torch.is_tensor(x) and x.dim() == 3:
            return x
    if len(obj) > 0 and torch.is_tensor(obj[0]) and obj[0].dim() == 3:
        return obj[0]
    return None


def extract_token_hidden(outputs) -> Optional[torch.Tensor]:
    """
    Try to extract token-level hidden states (B,L,D) from various output types.
    """
    if hasattr(outputs, "last_hidden_state") and torch.is_tensor(outputs.last_hidden_state):
        if outputs.last_hidden_state.dim() == 3:
            return outputs.last_hidden_state

    if hasattr(outputs, "hidden_states"):
        hs = getattr(outputs, "hidden_states")
        if isinstance(hs, (list, tuple)) and len(hs) > 0 and torch.is_tensor(hs[-1]) and hs[-1].dim() == 3:
            return hs[-1]

    if hasattr(outputs, "sequence_hidden_states") and torch.is_tensor(outputs.sequence_hidden_states):
        if outputs.sequence_hidden_states.dim() == 3:
            return outputs.sequence_hidden_states

    if isinstance(outputs, dict):
        if "last_hidden_state" in outputs and torch.is_tensor(outputs["last_hidden_state"]):
            if outputs["last_hidden_state"].dim() == 3:
                return outputs["last_hidden_state"]

        hs = outputs.get("hidden_states", None)
        if isinstance(hs, (list, tuple)) and len(hs) > 0 and torch.is_tensor(hs[-1]) and hs[-1].dim() == 3:
            return hs[-1]

        if "sequence_hidden_states" in outputs and torch.is_tensor(outputs["sequence_hidden_states"]):
            if outputs["sequence_hidden_states"].dim() == 3:
                return outputs["sequence_hidden_states"]

    t = _first_3d_tensor_from_tuple(outputs)
    if t is not None:
        return t

    return None


# ============================================================
# 3) Dataset
# ============================================================
class ProteinCSVDataset(Dataset):
    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path)
        self.ids = df.iloc[:, 0].astype(str).tolist()
        self.seqs = df.iloc[:, 1].astype(str).tolist()
        self.labels = df.iloc[:, 2].astype(int).tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {"id": self.ids[idx], "seq": self.seqs[idx], "label": int(self.labels[idx])}


# ============================================================
# 4) Cache helpers
# ============================================================
def cache_file_path(cache_dir: str, mode: str, sample_id: str) -> str:
    safe_id = sample_id.replace("/", "_")
    return os.path.join(cache_dir, mode, f"{safe_id}.pt")


def collate_raw_batch(
    batch: List[Dict],
    prott5_tokenizer: T5Tokenizer,
    proprime_tokenizer: AutoTokenizer,
    max_len: int
) -> Dict[str, torch.Tensor]:
    ids = [x["id"] for x in batch]
    seqs = [x["seq"] for x in batch]
    labels = torch.tensor([x["label"] for x in batch], dtype=torch.long)

    t5_text = [" ".join(list(s)) for s in seqs]
    enc_t5 = prott5_tokenizer(
        t5_text,
        max_length=max_len,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )

    enc_pp = proprime_tokenizer(
        seqs,
        max_length=max_len,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )

    return {
        "ids": ids,
        "labels": labels,
        "t5_input_ids": enc_t5["input_ids"],
        "t5_attn_mask": enc_t5["attention_mask"],
        "pp_input_ids": enc_pp["input_ids"],
        "pp_attn_mask": enc_pp["attention_mask"],
    }


def _pad_2d(seqs: List[torch.Tensor], pad_value: float = 0.0) -> torch.Tensor:
    B = len(seqs)
    L_max = max(s.size(0) for s in seqs)
    D = seqs[0].size(1)
    out = torch.full((B, L_max, D), pad_value, dtype=seqs[0].dtype)
    for i, s in enumerate(seqs):
        out[i, : s.size(0)] = s
    return out


def _pad_1d(ms: List[torch.Tensor], pad_value: int = 0) -> torch.Tensor:
    B = len(ms)
    L_max = max(m.size(0) for m in ms)
    out = torch.full((B, L_max), pad_value, dtype=ms[0].dtype)
    for i, m in enumerate(ms):
        out[i, : m.size(0)] = m
    return out


def collate_cache_bigru(batch: List[Dict], cache_dir: str, mode_dirname: str) -> Dict[str, torch.Tensor]:
    ids = []
    h_t5_list, m_t5_list = [], []
    h_pp_list, m_pp_list = [], []
    y_list = []

    for x in batch:
        sid = x["id"]
        ids.append(sid)
        fp = cache_file_path(cache_dir, mode_dirname, sid)
        obj = torch.load(fp, map_location="cpu")
        h_t5_list.append(obj["h_t5"].float())
        m_t5_list.append(obj["m_t5"].long())
        h_pp_list.append(obj["h_pp"].float())
        m_pp_list.append(obj["m_pp"].long())
        y_list.append(int(obj["label"]))

    h_t5 = _pad_2d(h_t5_list, 0.0)
    m_t5 = _pad_1d(m_t5_list, 0)
    h_pp = _pad_2d(h_pp_list, 0.0)
    m_pp = _pad_1d(m_pp_list, 0)

    return {
        "ids": ids,
        "labels": torch.tensor(y_list, dtype=torch.long),
        "h_t5": h_t5,
        "m_t5": m_t5,
        "h_pp": h_pp,
        "m_pp": m_pp,
    }


# ============================================================
# 5) Fusion head
# ============================================================
class MLPClassifier(nn.Module):
    def __init__(self, in_dim: int, num_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.net(x)


class FusionHead(nn.Module):
    def __init__(
        self,
        mode: str,
        dim_t5: int = 1024,
        dim_pp: int = 1280,
        num_classes: int = 3,
        dropout: float = 0.3,
        proj_dim: int = 512,
        gru_hidden: int = 256,
    ):
        super().__init__()
        assert mode == "bigru-fusion"
        self.mode = mode
        self.proj_t5 = nn.Linear(dim_t5, proj_dim)
        self.proj_pp = nn.Linear(dim_pp, proj_dim)
        self.bigru = nn.GRU(
            input_size=2 * proj_dim,
            hidden_size=gru_hidden,
            batch_first=True,
            bidirectional=True
        )
        self.classifier = MLPClassifier(2 * gru_hidden, num_classes, dropout)

    def forward(self, *inputs):
        h_t5, m_t5, h_pp, m_pp = inputs
        m_fuse = ((m_t5 > 0) & (m_pp > 0)).long()

        pt5 = self.proj_t5(h_t5)
        ppp = self.proj_pp(h_pp)
        x = torch.cat([pt5, ppp], dim=-1)

        y, _ = self.bigru(x)
        z = attention_pooling(y, m_fuse)
        return self.classifier(z)


# ============================================================
# 6) Load & freeze encoders
# ============================================================
def load_frozen_encoders(
    prott5_model_path: str,
    proprime_model_path: str,
    device: torch.device
):
    t5_tokenizer = T5Tokenizer.from_pretrained(prott5_model_path, do_lower_case=False)
    t5_encoder = T5EncoderModel.from_pretrained(prott5_model_path).to(device)
    t5_encoder.eval()
    t5_encoder.requires_grad_(False)

    pp_tokenizer = AutoTokenizer.from_pretrained(proprime_model_path, trust_remote_code=True)
    pp_encoder = AutoModel.from_pretrained(proprime_model_path, trust_remote_code=True).to(device)
    pp_encoder.eval()
    pp_encoder.requires_grad_(False)

    return t5_tokenizer, t5_encoder, pp_tokenizer, pp_encoder


# ============================================================
# 7) Build cache (robust forward)
# ============================================================
@torch.no_grad()
def build_cache(dataset: ProteinCSVDataset, indices: List[int], args, device: torch.device):
    ensure_dir(args.cache_dir)
    ensure_dir(os.path.join(args.cache_dir, args.mode))

    t5_tokenizer, t5_encoder, pp_tokenizer, pp_encoder = load_frozen_encoders(
        args.prott5_model_path, args.proprime_model_path, device
    )

    bs = args.cache_batch_size
    id_list = [dataset[i]["id"] for i in indices]
    seq_list = [dataset[i]["seq"] for i in indices]
    y_list = [dataset[i]["label"] for i in indices]

    pbar = tqdm(range(0, len(indices), bs), desc=f"[Cache] mode={args.mode}")
    for s in pbar:
        e = min(s + bs, len(indices))

        todo_ids, todo_seqs, todo_y = [], [], []
        for sid, seq, lab in zip(id_list[s:e], seq_list[s:e], y_list[s:e]):
            fp = cache_file_path(args.cache_dir, args.mode, sid)
            if not os.path.exists(fp) or args.cache_overwrite:
                todo_ids.append(sid)
                todo_seqs.append(seq)
                todo_y.append(lab)

        if len(todo_ids) == 0:
            continue

        t5_text = [" ".join(list(sq)) for sq in todo_seqs]
        enc_t5 = t5_tokenizer(
            t5_text,
            max_length=args.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        enc_t5 = {k: v.to(device) for k, v in enc_t5.items()}

        enc_pp = pp_tokenizer(
            todo_seqs,
            max_length=args.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        enc_pp = {k: v.to(device) for k, v in enc_pp.items()}

        out_t5 = t5_encoder(input_ids=enc_t5["input_ids"], attention_mask=enc_t5["attention_mask"])
        h_t5 = out_t5.last_hidden_state
        m_t5 = enc_t5["attention_mask"]

        try:
            out_pp = pp_encoder(**enc_pp, output_hidden_states=True, return_dict=True)
        except TypeError:
            out_pp = pp_encoder(**enc_pp)

        h_pp = extract_token_hidden(out_pp)
        m_pp = enc_pp.get("attention_mask", None)

        if h_pp is None:
            raise RuntimeError("bigru-fusion requires ProPrime token-level hidden (B,L,D), but got None.")

        for i, sid in enumerate(todo_ids):
            fp = cache_file_path(args.cache_dir, args.mode, sid)

            lt5 = int(m_t5[i].sum().item())
            lpp = int(m_pp[i].sum().item())

            obj = {
                "h_t5": h_t5[i, :lt5].detach().cpu().half(),
                "m_t5": m_t5[i, :lt5].detach().cpu().to(torch.int8),
                "h_pp": h_pp[i, :lpp].detach().cpu().half(),
                "m_pp": m_pp[i, :lpp].detach().cpu().to(torch.int8),
                "label": int(todo_y[i]),
            }
            torch.save(obj, fp)

    del t5_encoder, pp_encoder
    torch.cuda.empty_cache()


# ============================================================
# 8) EarlyStopping
# ============================================================
class EarlyStopping:
    def __init__(self, patience: int = 7, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.bad = 0

    def step(self, metric: float) -> bool:
        # metric 越小越好（这里用 val_loss）
        if self.best is None or metric < self.best - self.min_delta:
            self.best = metric
            self.bad = 0
            return False
        self.bad += 1
        return self.bad >= self.patience


# ============================================================
# 9) AMP helpers
# ============================================================
def get_amp(device: torch.device, fp16: bool):
    enabled = fp16 and device.type == "cuda"
    if hasattr(torch, "amp"):
        autocast = torch.amp.autocast
        GradScaler = torch.amp.GradScaler
        return enabled, autocast, GradScaler
    else:
        autocast = torch.cuda.amp.autocast
        GradScaler = torch.cuda.amp.GradScaler
        return enabled, autocast, GradScaler


# ============================================================
# 10) Metrics (新增：SN/SP/PRE/F1 + AUROC/AUPRC + MCC + per-class)
# ============================================================
def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b != 0 else 0.0


def per_class_from_confusion(cm: np.ndarray):
    """
    cm: (C,C), rows=true, cols=pred
    返回：每类的 TP, FP, TN, FN + SN/SP/PRE/F1
    """
    C = cm.shape[0]
    res = []
    for k in range(C):
        TP = cm[k, k]
        FN = cm[k, :].sum() - TP
        FP = cm[:, k].sum() - TP
        TN = cm.sum() - TP - FN - FP

        SN = _safe_div(TP, TP + FN)          # recall
        SP = _safe_div(TN, TN + FP)          # specificity
        PRE = _safe_div(TP, TP + FP)         # precision
        F1 = _safe_div(2 * PRE * SN, PRE + SN) if (PRE + SN) > 0 else 0.0

        res.append({
            "TP": int(TP), "FP": int(FP), "TN": int(TN), "FN": int(FN),
            "SN": SN, "SP": SP, "PRE": PRE, "F1": F1
        })
    return res


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, num_classes: int = 3):
    """
    y_true: (N,)
    y_pred: (N,)
    y_prob: (N,C) softmax 概率
    """
    out = {}
    out["ACC"] = float(accuracy_score(y_true, y_pred))
    out["MCC"] = float(matthews_corrcoef(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    pcs = per_class_from_confusion(cm)

    # per-class SN/SP/PRE/F1
    for k in range(num_classes):
        out[f"SN_c{k}"] = float(pcs[k]["SN"])
        out[f"SP_c{k}"] = float(pcs[k]["SP"])
        out[f"PRE_c{k}"] = float(pcs[k]["PRE"])
        out[f"F1_c{k}"] = float(pcs[k]["F1"])

    # macro
    out["SN_macro"] = float(np.mean([pcs[k]["SN"] for k in range(num_classes)]))
    out["SP_macro"] = float(np.mean([pcs[k]["SP"] for k in range(num_classes)]))
    out["PRE_macro"] = float(np.mean([pcs[k]["PRE"] for k in range(num_classes)]))
    out["F1_macro"] = float(np.mean([pcs[k]["F1"] for k in range(num_classes)]))

    # AUROC / AUPRC (one-vs-rest)
    y_true_oh = label_binarize(y_true, classes=list(range(num_classes)))  # (N,C)

    # roc_auc_score: macro (OVR), micro (treat as multilabel)
    try:
        out["AUROC_macro"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
    except Exception:
        out["AUROC_macro"] = float("nan")

    try:
        out["AUROC_micro"] = float(roc_auc_score(y_true_oh, y_prob, average="micro"))
    except Exception:
        out["AUROC_micro"] = float("nan")

    # PR AUC
    try:
        out["AUPRC_macro"] = float(average_precision_score(y_true_oh, y_prob, average="macro"))
    except Exception:
        out["AUPRC_macro"] = float("nan")

    try:
        out["AUPRC_micro"] = float(average_precision_score(y_true_oh, y_prob, average="micro"))
    except Exception:
        out["AUPRC_micro"] = float("nan")

    # per-class AUROC/AUPRC
    for k in range(num_classes):
        yk = y_true_oh[:, k]
        pk = y_prob[:, k]
        # AUROC per class
        if len(np.unique(yk)) < 2:
            out[f"AUROC_c{k}"] = float("nan")
        else:
            out[f"AUROC_c{k}"] = float(roc_auc_score(yk, pk))
        # AUPRC per class
        try:
            out[f"AUPRC_c{k}"] = float(average_precision_score(yk, pk))
        except Exception:
            out[f"AUPRC_c{k}"] = float("nan")

    return out


# ============================================================
# 11) Train / Eval loops
# ============================================================
def train_one_epoch(
    *,
    head: FusionHead,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    train_loader: DataLoader,
    device: torch.device,
    mode: str,
    use_cache: bool,
    fp16: bool,
    t5_encoder=None,
    pp_encoder=None,
):
    head.train()
    total_loss = 0.0
    all_y, all_pred = [], []

    amp_enabled, autocast, GradScaler = get_amp(device, fp16)
    scaler = GradScaler(device.type if hasattr(torch, "amp") else None, enabled=amp_enabled) if hasattr(torch, "amp") else GradScaler(enabled=amp_enabled)

    for batch in tqdm(train_loader, desc="Train", leave=False):
        optimizer.zero_grad(set_to_none=True)
        y = batch["labels"].to(device)

        if use_cache:
            h_t5 = batch["h_t5"].to(device)
            m_t5 = batch["m_t5"].to(device)
            h_pp = batch["h_pp"].to(device)
            m_pp = batch["m_pp"].to(device)
            with autocast(device_type=device.type, enabled=amp_enabled) if hasattr(torch, "amp") else autocast(enabled=amp_enabled):
                logits = head(h_t5, m_t5, h_pp, m_pp)
                loss = loss_fn(logits, y)

        else:
            t5_input_ids = batch["t5_input_ids"].to(device)
            t5_attn_mask = batch["t5_attn_mask"].to(device)
            pp_input_ids = batch["pp_input_ids"].to(device)
            pp_attn_mask = batch["pp_attn_mask"].to(device)

            with torch.no_grad():
                out_t5 = t5_encoder(input_ids=t5_input_ids, attention_mask=t5_attn_mask)
                h_t5 = out_t5.last_hidden_state

                try:
                    out_pp = pp_encoder(input_ids=pp_input_ids, attention_mask=pp_attn_mask,
                                        output_hidden_states=True, return_dict=True)
                except TypeError:
                    out_pp = pp_encoder(input_ids=pp_input_ids, attention_mask=pp_attn_mask)

                h_pp = extract_token_hidden(out_pp)

            if h_pp is None:
                raise RuntimeError("bigru-fusion requires ProPrime token-level hidden (B,L,D), but got None.")
            with autocast(device_type=device.type, enabled=amp_enabled) if hasattr(torch, "amp") else autocast(enabled=amp_enabled):
                logits = head(h_t5, t5_attn_mask, h_pp, pp_attn_mask)
                loss = loss_fn(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * y.size(0)
        pred = torch.argmax(logits.detach(), dim=1).cpu().numpy().tolist()
        all_pred.extend(pred)
        all_y.extend(y.detach().cpu().numpy().tolist())

    avg_loss = total_loss / len(train_loader.dataset)
    acc = accuracy_score(all_y, all_pred)
    f1m = f1_score(all_y, all_pred, average="macro")
    return avg_loss, acc, f1m


@torch.no_grad()
def eval_one_epoch(
    *,
    head: FusionHead,
    loss_fn,
    val_loader: DataLoader,
    device: torch.device,
    mode: str,
    use_cache: bool,
    fp16: bool,
    t5_encoder=None,
    pp_encoder=None,
):
    head.eval()
    total_loss = 0.0
    all_y, all_pred = [], []

    amp_enabled, autocast, _ = get_amp(device, fp16)

    for batch in tqdm(val_loader, desc="Val", leave=False):
        y = batch["labels"].to(device)

        if use_cache:
            h_t5 = batch["h_t5"].to(device)
            m_t5 = batch["m_t5"].to(device)
            h_pp = batch["h_pp"].to(device)
            m_pp = batch["m_pp"].to(device)
            with autocast(device_type=device.type, enabled=amp_enabled) if hasattr(torch, "amp") else autocast(enabled=amp_enabled):
                logits = head(h_t5, m_t5, h_pp, m_pp)
                loss = loss_fn(logits, y)

        else:
            t5_input_ids = batch["t5_input_ids"].to(device)
            t5_attn_mask = batch["t5_attn_mask"].to(device)
            pp_input_ids = batch["pp_input_ids"].to(device)
            pp_attn_mask = batch["pp_attn_mask"].to(device)

            out_t5 = t5_encoder(input_ids=t5_input_ids, attention_mask=t5_attn_mask)
            h_t5 = out_t5.last_hidden_state

            try:
                out_pp = pp_encoder(input_ids=pp_input_ids, attention_mask=pp_attn_mask,
                                    output_hidden_states=True, return_dict=True)
            except TypeError:
                out_pp = pp_encoder(input_ids=pp_input_ids, attention_mask=pp_attn_mask)

            h_pp = extract_token_hidden(out_pp)

            if h_pp is None:
                raise RuntimeError("bigru-fusion requires ProPrime token-level hidden (B,L,D), but got None.")
            with autocast(device_type=device.type, enabled=amp_enabled) if hasattr(torch, "amp") else autocast(enabled=amp_enabled):
                logits = head(h_t5, t5_attn_mask, h_pp, pp_attn_mask)
                loss = loss_fn(logits, y)

        total_loss += loss.item() * y.size(0)
        pred = torch.argmax(logits.detach(), dim=1).cpu().numpy().tolist()
        all_pred.extend(pred)
        all_y.extend(y.detach().cpu().numpy().tolist())

    avg_loss = total_loss / len(val_loader.dataset)
    acc = accuracy_score(all_y, all_pred)
    f1m = f1_score(all_y, all_pred, average="macro")
    return avg_loss, acc, f1m


@torch.no_grad()
def predict_val_probs(
    *,
    head: FusionHead,
    val_loader: DataLoader,
    device: torch.device,
    mode: str,
    use_cache: bool,
    fp16: bool,
    t5_encoder=None,
    pp_encoder=None,
    num_classes: int = 3,
):
    """
    返回：y_true, y_pred, y_prob(softmax)
    """
    head.eval()
    amp_enabled, autocast, _ = get_amp(device, fp16)

    ys, preds = [], []
    probs = []

    for batch in tqdm(val_loader, desc="Predict", leave=False):
        y = batch["labels"].to(device)

        if use_cache:
            h_t5 = batch["h_t5"].to(device)
            m_t5 = batch["m_t5"].to(device)
            h_pp = batch["h_pp"].to(device)
            m_pp = batch["m_pp"].to(device)
            with autocast(device_type=device.type, enabled=amp_enabled) if hasattr(torch, "amp") else autocast(enabled=amp_enabled):
                logits = head(h_t5, m_t5, h_pp, m_pp)
        else:
            t5_input_ids = batch["t5_input_ids"].to(device)
            t5_attn_mask = batch["t5_attn_mask"].to(device)
            pp_input_ids = batch["pp_input_ids"].to(device)
            pp_attn_mask = batch["pp_attn_mask"].to(device)

            out_t5 = t5_encoder(input_ids=t5_input_ids, attention_mask=t5_attn_mask)
            h_t5 = out_t5.last_hidden_state

            try:
                out_pp = pp_encoder(input_ids=pp_input_ids, attention_mask=pp_attn_mask,
                                    output_hidden_states=True, return_dict=True)
            except TypeError:
                out_pp = pp_encoder(input_ids=pp_input_ids, attention_mask=pp_attn_mask)

            h_pp = extract_token_hidden(out_pp)

            if h_pp is None:
                raise RuntimeError("bigru-fusion requires ProPrime token-level hidden (B,L,D), but got None.")
            with autocast(device_type=device.type, enabled=amp_enabled) if hasattr(torch, "amp") else autocast(enabled=amp_enabled):
                logits = head(h_t5, t5_attn_mask, h_pp, pp_attn_mask)

        p = torch.softmax(logits.float(), dim=1)  # (B,C)
        pred = torch.argmax(p, dim=1)

        ys.append(y.detach().cpu().numpy())
        preds.append(pred.detach().cpu().numpy())
        probs.append(p.detach().cpu().numpy())

    y_true = np.concatenate(ys, axis=0)
    y_pred = np.concatenate(preds, axis=0)
    y_prob = np.concatenate(probs, axis=0)
    return y_true, y_pred, y_prob


# ============================================================
# 12) Plot helpers (error bars)
# ============================================================
def plot_errorbar_metrics(df_fold: pd.DataFrame, out_png: str, metrics: List[str], title: str, err: str = "std"):
    """
    err: "std" or "sem"
    """
    means = []
    errs = []
    for m in metrics:
        v = df_fold[m].astype(float).values
        means.append(np.nanmean(v))
        if err == "sem":
            errs.append(np.nanstd(v, ddof=1) / np.sqrt(len(v)))
        else:
            errs.append(np.nanstd(v, ddof=1))

    x = np.arange(len(metrics))
    plt.figure(figsize=(max(10, len(metrics) * 0.8), 5))
    plt.bar(x, means, yerr=errs, capsize=6)
    plt.xticks(x, metrics, rotation=30, ha="right")
    plt.ylabel("Score")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_errorbar_per_class(df_fold: pd.DataFrame, out_png: str, class_names: List[str], base_metrics: List[str], err: str = "std"):
    """
    每类：base_metrics（SN/SP/PRE/F1/AUROC/AUPRC）分别画 errorbar
    """
    num_classes = len(class_names)
    plt.figure(figsize=(12, 6))
    # group bar with error
    group_width = 0.8
    bar_width = group_width / num_classes
    x = np.arange(len(base_metrics))

    for k in range(num_classes):
        means = []
        errs = []
        for bm in base_metrics:
            col = f"{bm}_c{k}"
            v = df_fold[col].astype(float).values
            means.append(np.nanmean(v))
            if err == "sem":
                errs.append(np.nanstd(v, ddof=1) / np.sqrt(len(v)))
            else:
                errs.append(np.nanstd(v, ddof=1))
        offset = (k - (num_classes - 1) / 2.0) * bar_width
        plt.bar(x + offset, means, yerr=errs, width=bar_width, capsize=5, label=class_names[k])

    plt.xticks(x, base_metrics)
    plt.ylabel("Score")
    plt.title("Per-class metrics (mean ± error)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# ============================================================
# 13) Main
# ============================================================
def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--train_csv", type=str, default="train.csv")
    ap.add_argument("--prott5_model_path", type=str, default="prot_t5_xl_uniref50")
    ap.add_argument("--proprime_model_path", type=str, default="ProPrime_650M_OGT_Prediction")

    ap.add_argument("--mode", type=str, default="bigru-fusion",
                    choices=["bigru-fusion"])

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--k_folds", type=int, default=5)

    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=1024)

    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--min_delta", type=float, default=0.0)

    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.3)

    ap.add_argument("--label_smoothing", type=float, default=0.0)

    ap.add_argument("--plateau_factor", type=float, default=0.5)
    ap.add_argument("--plateau_patience", type=int, default=2)
    ap.add_argument("--plateau_min_lr", type=float, default=1e-6)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--deterministic", action="store_true")

    ap.add_argument("--out_dir", type=str, default="fusion_out")

    # cache
    ap.add_argument("--use_cache", action="store_true")
    ap.add_argument("--cache_dir", type=str, default="fusion_cache")
    ap.add_argument("--cache_batch_size", type=int, default=4)
    ap.add_argument("--cache_overwrite", action="store_true")

    # bigru
    ap.add_argument("--proj_dim", type=int, default=512)
    ap.add_argument("--gru_hidden", type=int, default=256)

    # dataloader
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--pin_memory", action="store_true")

    # monitor
    ap.add_argument("--tensorboard", action="store_true")
    ap.add_argument("--fp16", action="store_true")

    # NEW: class names
    ap.add_argument("--class_names", type=str, default="嗜温,嗜热,嗜冷",
                    help="Comma-separated class names aligned with labels 0,1,2")

    # NEW: errorbar type
    ap.add_argument("--errorbar", type=str, default="std", choices=["std", "sem"],
                    help="errorbar type for plots")

    return ap.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.out_dir)
    set_seed(args.seed, deterministic=args.deterministic)

    class_names = [x.strip() for x in args.class_names.split(",")]
    assert len(class_names) == 3, f"--class_names must have 3 names, got: {class_names}"
    num_classes = 3

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    print(f"[INFO] device={device} | mode={args.mode} | use_cache={args.use_cache}")

    writer = None
    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        tb_dir = os.path.join(args.out_dir, f"runs_{args.mode}_{now_ts()}")
        writer = SummaryWriter(tb_dir)
        print(f"[TB] logdir: {tb_dir}")

    full_ds = ProteinCSVDataset(args.train_csv)
    labels = np.array([full_ds[i]["label"] for i in range(len(full_ds))], dtype=np.int64)

    # Cache build once (whole dataset)
    if args.use_cache:
        print("[INFO] Building/Checking cache ...")
        all_idx = list(range(len(full_ds)))
        build_cache(full_ds, all_idx, args, device)

    # For no-cache, load encoders once
    t5_tokenizer = t5_encoder = pp_tokenizer = pp_encoder = None
    if not args.use_cache:
        t5_tokenizer, t5_encoder, pp_tokenizer, pp_encoder = load_frozen_encoders(
            args.prott5_model_path, args.proprime_model_path, device
        )

    loss_fn = nn.CrossEntropyLoss(label_smoothing=float(args.label_smoothing))

    skf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)

    fold_train_stats = []
    fold_val_metrics = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(labels)), labels), start=1):
        print(f"\n========== Fold {fold}/{args.k_folds} ==========")

        head = FusionHead(
            mode=args.mode,
            dim_t5=1024,
            dim_pp=1280,   # 如果你的 ProPrime dim 不同，手动改这里
            num_classes=num_classes,
            dropout=args.dropout,
            proj_dim=args.proj_dim,
            gru_hidden=args.gru_hidden,
        ).to(device)

        optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min",
            factor=args.plateau_factor,
            patience=args.plateau_patience,
            min_lr=args.plateau_min_lr
        )
        stopper = EarlyStopping(patience=args.patience, min_delta=args.min_delta)

        tr_subset = torch.utils.data.Subset(full_ds, tr_idx.tolist())
        va_subset = torch.utils.data.Subset(full_ds, va_idx.tolist())

        from functools import partial
        if args.use_cache:
            collate_fn = partial(collate_cache_bigru, cache_dir=args.cache_dir, mode_dirname=args.mode)
        else:
            collate_fn = partial(collate_raw_batch, prott5_tokenizer=t5_tokenizer, proprime_tokenizer=pp_tokenizer, max_len=args.max_len)

        train_loader = DataLoader(
            tr_subset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=args.pin_memory,
            collate_fn=collate_fn
        )
        val_loader = DataLoader(
            va_subset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=args.pin_memory,
            collate_fn=collate_fn
        )

        best_val_loss = float("inf")
        best_path = os.path.join(args.out_dir, f"best_head_fold_{fold}.pt")

        for epoch in range(1, args.epochs + 1):
            lr_now = optimizer.param_groups[0]["lr"]

            tr_loss, tr_acc, tr_f1 = train_one_epoch(
                head=head, optimizer=optimizer, loss_fn=loss_fn,
                train_loader=train_loader, device=device,
                mode=args.mode, use_cache=args.use_cache, fp16=args.fp16,
                t5_encoder=t5_encoder, pp_encoder=pp_encoder
            )

            va_loss, va_acc, va_f1 = eval_one_epoch(
                head=head, loss_fn=loss_fn,
                val_loader=val_loader, device=device,
                mode=args.mode, use_cache=args.use_cache, fp16=args.fp16,
                t5_encoder=t5_encoder, pp_encoder=pp_encoder
            )

            scheduler.step(va_loss)

            print(
                f"[Fold {fold}][Epoch {epoch:02d}] lr={lr_now:.2e} | "
                f"train_loss={tr_loss:.4f} acc={tr_acc:.4f} f1m={tr_f1:.4f} | "
                f"val_loss={va_loss:.4f} acc={va_acc:.4f} f1m={va_f1:.4f}"
            )

            if writer is not None:
                step = (fold - 1) * args.epochs + epoch
                writer.add_scalar(f"fold{fold}/lr", lr_now, step)
                writer.add_scalar(f"fold{fold}/train_loss", tr_loss, step)
                writer.add_scalar(f"fold{fold}/train_acc", tr_acc, step)
                writer.add_scalar(f"fold{fold}/train_f1m", tr_f1, step)
                writer.add_scalar(f"fold{fold}/val_loss", va_loss, step)
                writer.add_scalar(f"fold{fold}/val_acc", va_acc, step)
                writer.add_scalar(f"fold{fold}/val_f1m", va_f1, step)

            if va_loss < best_val_loss:
                best_val_loss = va_loss
                torch.save({"head": head.state_dict(), "args": vars(args)}, best_path)

            if stopper.step(va_loss):
                print(f"[Fold {fold}] Early stopping at epoch {epoch}. Best val_loss={best_val_loss:.4f}")
                break

        # load best
        ckpt = torch.load(best_path, map_location=device)
        head.load_state_dict(ckpt["head"])

        # 重新做一次 val loss/acc/f1
        va_loss, va_acc, va_f1 = eval_one_epoch(
            head=head, loss_fn=loss_fn,
            val_loader=val_loader, device=device,
            mode=args.mode, use_cache=args.use_cache, fp16=args.fp16,
            t5_encoder=t5_encoder, pp_encoder=pp_encoder
        )
        fold_train_stats.append({"fold": fold, "val_loss": va_loss, "val_acc": va_acc, "val_f1_macro": va_f1})

        # NEW: 计算完整指标（需要概率）
        y_true, y_pred, y_prob = predict_val_probs(
            head=head, val_loader=val_loader, device=device,
            mode=args.mode, use_cache=args.use_cache, fp16=args.fp16,
            t5_encoder=t5_encoder, pp_encoder=pp_encoder,
            num_classes=num_classes
        )
        m = compute_all_metrics(y_true, y_pred, y_prob, num_classes=num_classes)
        m["fold"] = fold
        m["val_loss"] = float(va_loss)

        # 额外：把每类指标复制成带类名的列（方便看）
        for k, cname in enumerate(class_names):
            m[f"{cname}_SN"] = m[f"SN_c{k}"]
            m[f"{cname}_SP"] = m[f"SP_c{k}"]
            m[f"{cname}_PRE"] = m[f"PRE_c{k}"]
            m[f"{cname}_F1"] = m[f"F1_c{k}"]
            m[f"{cname}_AUROC"] = m[f"AUROC_c{k}"]
            m[f"{cname}_AUPRC"] = m[f"AUPRC_c{k}"]

        fold_val_metrics.append(m)

        print(
            f"[Fold {fold}] BEST val_loss={va_loss:.4f} | "
            f"ACC={m['ACC']:.4f} F1_macro={m['F1_macro']:.4f} MCC={m['MCC']:.4f} "
            f"AUROC_macro={m['AUROC_macro']:.4f} AUPRC_macro={m['AUPRC_macro']:.4f}"
        )

        with open(os.path.join(args.out_dir, f"fold_{fold}_stats.json"), "w", encoding="utf-8") as f:
            json.dump({"train_stats": fold_train_stats[-1], "val_metrics": m}, f, indent=2, ensure_ascii=False)

    # ============================================================
    # 汇总：CSV + errorbar plots
    # ============================================================
    df_fold = pd.DataFrame(fold_val_metrics).sort_values("fold").reset_index(drop=True)
    fold_csv = os.path.join(args.out_dir, "fold_val_metrics.csv")
    df_fold.to_csv(fold_csv, index=False, encoding="utf-8-sig")
    print(f"[SAVE] {fold_csv}")

    # 选择你关心的总体指标列
    overall_metrics = [
        "ACC",
        "SN_macro", "SP_macro", "PRE_macro", "F1_macro",
        "AUROC_macro", "AUROC_micro",
        "AUPRC_macro", "AUPRC_micro",
        "MCC"
    ]

    # summary: mean/std/sem
    summary_rows = []
    for mname in overall_metrics + [f"{bm}_c{k}" for bm in ["SN", "SP", "PRE", "F1", "AUROC", "AUPRC"] for k in range(num_classes)]:
        vals = df_fold[mname].astype(float).values if mname in df_fold.columns else np.array([np.nan]*len(df_fold))
        mean = float(np.nanmean(vals))
        std = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
        sem = float(std / np.sqrt(len(vals))) if len(vals) > 0 else 0.0
        summary_rows.append({"metric": mname, "mean": mean, "std": std, "sem": sem})

    df_sum = pd.DataFrame(summary_rows)
    sum_csv = os.path.join(args.out_dir, "cv_metrics_summary.csv")
    df_sum.to_csv(sum_csv, index=False, encoding="utf-8-sig")
    print(f"[SAVE] {sum_csv}")

    # plots
    plot1 = os.path.join(args.out_dir, "errorbar_overall_metrics.png")
    plot_errorbar_metrics(df_fold, plot1, overall_metrics, title=f"Overall metrics (mean ± {args.errorbar})", err=args.errorbar)
    print(f"[SAVE] {plot1}")

    plot2 = os.path.join(args.out_dir, "errorbar_per_class_metrics.png")
    base_metrics = ["SN", "SP", "PRE", "F1", "AUROC", "AUPRC"]
    plot_errorbar_per_class(df_fold, plot2, class_names, base_metrics, err=args.errorbar)
    print(f"[SAVE] {plot2}")

    # 兼容你原来的 cv_summary.json（保留）
    accs = df_fold["ACC"].astype(float).values
    f1s = df_fold["F1_macro"].astype(float).values
    losses = df_fold["val_loss"].astype(float).values
    summary = {
        "mode": args.mode,
        "use_cache": args.use_cache,
        "k_folds": args.k_folds,
        "val_loss_mean": float(np.mean(losses)),
        "val_loss_std": float(np.std(losses, ddof=1)),
        "val_acc_mean": float(np.mean(accs)),
        "val_acc_std": float(np.std(accs, ddof=1)),
        "val_f1_macro_mean": float(np.mean(f1s)),
        "val_f1_macro_std": float(np.std(f1s, ddof=1)),
        "folds": fold_train_stats
    }

    print("\n========== CV Summary (Train stats) ==========")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    with open(os.path.join(args.out_dir, "cv_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
