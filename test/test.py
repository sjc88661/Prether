#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import json
import argparse
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import T5Tokenizer, T5EncoderModel
from transformers import AutoTokenizer, AutoModel

from sklearn.metrics import (
    accuracy_score, f1_score,
    confusion_matrix, matthews_corrcoef,
    roc_curve, auc,
    precision_recall_curve,
    roc_auc_score, average_precision_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

# Optional UMAP
try:
    import umap
    _HAVE_UMAP = True
except Exception:
    _HAVE_UMAP = False


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


# ============================================================
# 1) Attention pooling
# ============================================================
def attention_pooling(hidden: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    """
    hidden:    (B, L, D)
    attn_mask: (B, L)
    """
    weights = hidden.mean(dim=-1)  # (B, L)
    weights = weights.masked_fill(attn_mask == 0, -1e9)
    weights = torch.softmax(weights, dim=1)
    pooled = torch.sum(hidden * weights.unsqueeze(-1), dim=1)  # (B, D)
    return pooled


# ============================================================
# 2) Robust hidden extraction (ProPrime-compatible)
# ============================================================
def _first_3d_tensor_from_tuple(obj) -> Optional[torch.Tensor]:
    if not isinstance(obj, (tuple, list)):
        return None
    for x in obj:
        if torch.is_tensor(x) and x.dim() == 3:
            return x
    if len(obj) > 0 and torch.is_tensor(obj[0]) and obj[0].dim() == 3:
        return obj[0]
    return None


def extract_token_hidden(outputs) -> Optional[torch.Tensor]:
    if hasattr(outputs, "last_hidden_state") and torch.is_tensor(outputs.last_hidden_state):
        if outputs.last_hidden_state.dim() == 3:
            return outputs.last_hidden_state

    if hasattr(outputs, "hidden_states"):
        hs = getattr(outputs, "hidden_states")
        if isinstance(hs, (list, tuple)) and len(hs) > 0 and torch.is_tensor(hs[-1]) and hs[-1].dim() == 3:
            return hs[-1]

    if isinstance(outputs, dict):
        if "last_hidden_state" in outputs and torch.is_tensor(outputs["last_hidden_state"]) and outputs["last_hidden_state"].dim() == 3:
            return outputs["last_hidden_state"]
        hs = outputs.get("hidden_states", None)
        if isinstance(hs, (list, tuple)) and len(hs) > 0 and torch.is_tensor(hs[-1]) and hs[-1].dim() == 3:
            return hs[-1]

    t = _first_3d_tensor_from_tuple(outputs)
    if t is not None:
        return t

    return None


def extract_pooled_if_available(outputs) -> Optional[torch.Tensor]:
    if hasattr(outputs, "sequence_hidden_states") and torch.is_tensor(outputs.sequence_hidden_states):
        if outputs.sequence_hidden_states.dim() == 2:
            return outputs.sequence_hidden_states
    if isinstance(outputs, dict) and "sequence_hidden_states" in outputs and torch.is_tensor(outputs["sequence_hidden_states"]):
        if outputs["sequence_hidden_states"].dim() == 2:
            return outputs["sequence_hidden_states"]
    return None


# ============================================================
# 3) Dataset & collate
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


def collate_raw_batch(batch, prott5_tokenizer, proprime_tokenizer, max_len: int):
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


# ============================================================
# 4) Fusion head (same as training)
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
            nn.Linear(256, num_classes),
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
        assert mode in ["late-fusion", "bigru-fusion"]
        self.mode = mode

        if mode == "late-fusion":
            self.classifier = MLPClassifier(dim_t5 + dim_pp, num_classes, dropout)
        else:
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
        if self.mode == "late-fusion":
            e_t5, e_pp = inputs
            x = torch.cat([e_t5, e_pp], dim=-1)
            return self.classifier(x)

        h_t5, m_t5, h_pp, m_pp = inputs
        m_fuse = ((m_t5 > 0) & (m_pp > 0)).long()
        pt5 = self.proj_t5(h_t5)
        ppp = self.proj_pp(h_pp)
        x = torch.cat([pt5, ppp], dim=-1)
        y, _ = self.bigru(x)
        z = attention_pooling(y, m_fuse)
        return self.classifier(z)

    @torch.no_grad()
    def forward_features(self, *inputs) -> torch.Tensor:
        if self.mode == "late-fusion":
            e_t5, e_pp = inputs
            return torch.cat([e_t5, e_pp], dim=-1)

        h_t5, m_t5, h_pp, m_pp = inputs
        m_fuse = ((m_t5 > 0) & (m_pp > 0)).long()
        pt5 = self.proj_t5(h_t5)
        ppp = self.proj_pp(h_pp)
        x = torch.cat([pt5, ppp], dim=-1)
        y, _ = self.bigru(x)
        z = attention_pooling(y, m_fuse)
        return z


# ============================================================
# 5) Load frozen encoders
# ============================================================
def load_frozen_encoders(prott5_model_path: str, proprime_model_path: str, device: torch.device):
    t5_tokenizer = T5Tokenizer.from_pretrained(prott5_model_path, do_lower_case=False, legacy=True)
    t5_encoder = T5EncoderModel.from_pretrained(prott5_model_path).to(device)
    t5_encoder.eval()
    t5_encoder.requires_grad_(False)

    pp_tokenizer = AutoTokenizer.from_pretrained(proprime_model_path, trust_remote_code=True)
    pp_encoder = AutoModel.from_pretrained(proprime_model_path, trust_remote_code=True).to(device)
    pp_encoder.eval()
    pp_encoder.requires_grad_(False)

    return t5_tokenizer, t5_encoder, pp_tokenizer, pp_encoder


# ============================================================
# 6) Metrics helpers
# ============================================================
def safe_div(a, b):
    return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=(b != 0))


def one_vs_rest_stats(cm: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    TP = np.diag(cm)
    FN = cm.sum(axis=1) - TP
    FP = cm.sum(axis=0) - TP
    TN = cm.sum() - (TP + FN + FP)
    return TP, FN, FP, TN


def compute_overall_and_perclass_from_cm(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> Dict[str, float]:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    TP, FN, FP, TN = one_vs_rest_stats(cm)

    sn = safe_div(TP, TP + FN)    # recall
    sp = safe_div(TN, TN + FP)    # specificity
    pre = safe_div(TP, TP + FP)   # precision
    f1 = safe_div(2 * pre * sn, pre + sn)

    out = {
        "SN_macro": float(np.mean(sn)),
        "SP_macro": float(np.mean(sp)),
        "PRE_macro": float(np.mean(pre)),
        "confusion_matrix": cm,
    }
    for k in range(num_classes):
        out[f"SN_c{k}"] = float(sn[k])
        out[f"SP_c{k}"] = float(sp[k])
        out[f"PRE_c{k}"] = float(pre[k])
        out[f"F1_c{k}"] = float(f1[k])
    return out


def compute_auc_metrics_overall_and_perclass(y_true: np.ndarray, y_prob: np.ndarray, num_classes: int) -> Dict[str, float]:
    Y = np.eye(num_classes)[y_true]  # (N,C)

    # AUROC macro/micro
    try:
        auroc_macro = roc_auc_score(Y, y_prob, average="macro")
    except Exception:
        auroc_macro = float("nan")
    try:
        auroc_micro = roc_auc_score(Y, y_prob, average="micro")
    except Exception:
        auroc_micro = float("nan")

    # AUPRC macro/micro
    try:
        auprc_macro = average_precision_score(Y, y_prob, average="macro")
    except Exception:
        auprc_macro = float("nan")
    try:
        auprc_micro = average_precision_score(Y, y_prob, average="micro")
    except Exception:
        auprc_micro = float("nan")

    out = {
        "AUROC_macro": float(auroc_macro),
        "AUROC_micro": float(auroc_micro),
        "AUPRC_macro": float(auprc_macro),
        "AUPRC_micro": float(auprc_micro),
    }

    # per-class (OvR)
    for k in range(num_classes):
        yk = (y_true == k).astype(int)
        pk = y_prob[:, k]
        if len(np.unique(yk)) < 2:
            out[f"AUROC_c{k}"] = float("nan")
        else:
            out[f"AUROC_c{k}"] = float(roc_auc_score(yk, pk))
        try:
            out[f"AUPRC_c{k}"] = float(average_precision_score(yk, pk))
        except Exception:
            out[f"AUPRC_c{k}"] = float("nan")

    return out


def roc_pr_micro_curve(y_true: np.ndarray, y_prob: np.ndarray, num_classes: int):
    Y = np.eye(num_classes)[y_true]
    y_true_micro = Y.ravel()
    y_score_micro = y_prob.ravel()

    fpr, tpr, _ = roc_curve(y_true_micro, y_score_micro)
    roc_auc = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(y_true_micro, y_score_micro)
    ap = average_precision_score(y_true_micro, y_score_micro)

    return (fpr, tpr, roc_auc), (precision, recall, ap)


def roc_pr_per_class_curve(y_true: np.ndarray, y_prob: np.ndarray, class_idx: int):
    y_bin = (y_true == class_idx).astype(int)
    y_score = y_prob[:, class_idx]

    # ROC
    if len(np.unique(y_bin)) < 2:
        fpr = np.array([0.0, 1.0])
        tpr = np.array([0.0, 1.0])
        roc_auc = float("nan")
    else:
        fpr, tpr, _ = roc_curve(y_bin, y_score)
        roc_auc = auc(fpr, tpr)

    # PR
    precision, recall, _ = precision_recall_curve(y_bin, y_score)
    try:
        ap = average_precision_score(y_bin, y_score)
    except Exception:
        ap = float("nan")

    return (fpr, tpr, roc_auc), (precision, recall, ap)


# ============================================================
# 7) Inference (per fold)
# ============================================================
@torch.no_grad()
def infer_one_fold(
    fold_ckpt_path: str,
    head: FusionHead,
    mode: str,
    dataloader: DataLoader,
    device: torch.device,
    t5_encoder,
    pp_encoder,
    num_classes: int
):
    ckpt = torch.load(fold_ckpt_path, map_location=device)
    head.load_state_dict(ckpt["head"], strict=True)
    head.eval()

    all_ids, all_y, all_prob, all_pred, all_feat = [], [], [], [], []

    for batch in tqdm(dataloader, desc=f"Infer {os.path.basename(fold_ckpt_path)}", leave=False):
        ids = batch["ids"]
        y = batch["labels"].to(device)

        t5_input_ids = batch["t5_input_ids"].to(device)
        t5_attn_mask = batch["t5_attn_mask"].to(device)
        pp_input_ids = batch["pp_input_ids"].to(device)
        pp_attn_mask = batch["pp_attn_mask"].to(device)

        out_t5 = t5_encoder(input_ids=t5_input_ids, attention_mask=t5_attn_mask)
        h_t5 = out_t5.last_hidden_state

        try:
            out_pp = pp_encoder(
                input_ids=pp_input_ids,
                attention_mask=pp_attn_mask,
                output_hidden_states=True,
                return_dict=True
            )
        except TypeError:
            out_pp = pp_encoder(input_ids=pp_input_ids, attention_mask=pp_attn_mask)

        h_pp = extract_token_hidden(out_pp)

        if mode == "late-fusion":
            e_t5 = attention_pooling(h_t5, t5_attn_mask)
            if h_pp is None:
                pooled_pp = extract_pooled_if_available(out_pp)
                if pooled_pp is None:
                    raise RuntimeError("ProPrime output has no token hidden nor pooled embedding.")
                e_pp = pooled_pp
            else:
                e_pp = attention_pooling(h_pp, pp_attn_mask)

            logits = head(e_t5, e_pp)
            feat = head.forward_features(e_t5, e_pp)

        else:
            if h_pp is None:
                raise RuntimeError("bigru-fusion requires ProPrime token-level hidden (B,L,D), got None.")
            logits = head(h_t5, t5_attn_mask, h_pp, pp_attn_mask)
            feat = head.forward_features(h_t5, t5_attn_mask, h_pp, pp_attn_mask)

        prob = torch.softmax(logits, dim=1)
        pred = torch.argmax(prob, dim=1)

        all_ids.extend(ids)
        all_y.append(y.detach().cpu().numpy())
        all_prob.append(prob.detach().cpu().numpy())
        all_pred.append(pred.detach().cpu().numpy())
        all_feat.append(feat.detach().cpu().numpy())

    ids_out = np.array(all_ids, dtype=str)
    y_true = np.concatenate(all_y, axis=0)
    y_prob = np.concatenate(all_prob, axis=0)
    y_pred = np.concatenate(all_pred, axis=0)
    feats = np.concatenate(all_feat, axis=0)

    return ids_out, y_true, y_pred, y_prob, feats


# ============================================================
# 8) Plot helpers
# ============================================================
def plot_roc_micro(per_fold_curves, ensemble_curve, out_path: str):
    plt.figure()
    for i, (fpr, tpr, roc_auc) in enumerate(per_fold_curves, start=1):
        plt.plot(fpr, tpr, alpha=0.5, label=f"fold{i} AUC={roc_auc:.3f}")
    efpr, etpr, eauc = ensemble_curve
    plt.plot(efpr, etpr, linewidth=2.5, label=f"mean-ensemble AUC={eauc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("Micro-averaged ROC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_pr_micro(per_fold_curves, ensemble_curve, out_path: str):
    plt.figure()
    for i, (prec, rec, ap) in enumerate(per_fold_curves, start=1):
        plt.plot(rec, prec, alpha=0.5, label=f"fold{i} AP={ap:.3f}")
    eprec, erec, eap = ensemble_curve
    plt.plot(erec, eprec, linewidth=2.5, label=f"mean-ensemble AP={eap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Micro-averaged Precision-Recall")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_roc_pr_per_class(per_fold_curves, ens_curve, class_name: str, out_roc: str, out_pr: str):
    # ROC
    plt.figure()
    for i, (fpr, tpr, roc_auc) in enumerate(per_fold_curves["roc"], start=1):
        plt.plot(fpr, tpr, alpha=0.5, label=f"fold{i} AUC={roc_auc:.3f}")
    efpr, etpr, eauc = ens_curve["roc"]
    plt.plot(efpr, etpr, linewidth=2.5, label=f"mean-ensemble AUC={eauc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title(f"ROC (OvR) - {class_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_roc, dpi=300)
    plt.close()

    # PR
    plt.figure()
    for i, (prec, rec, ap) in enumerate(per_fold_curves["pr"], start=1):
        plt.plot(rec, prec, alpha=0.5, label=f"fold{i} AP={ap:.3f}")
    eprec, erec, eap = ens_curve["pr"]
    plt.plot(erec, eprec, linewidth=2.5, label=f"mean-ensemble AP={eap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR (OvR) - {class_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_pr, dpi=300)
    plt.close()


def plot_confusion(cm: np.ndarray, labels: List[str], out_path: str, normalize: bool = False, title: str = ""):
    if normalize:
        cm = cm.astype(float)
        row_sum = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sum, out=np.zeros_like(cm), where=(row_sum != 0))

    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(title if title else ("Confusion Matrix (normalized)" if normalize else "Confusion Matrix"))
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0 if cm.size > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_tsne_umap(features: np.ndarray, y_true: np.ndarray, out_dir: str, title_prefix: str = ""):
    ensure_dir(out_dir)

    X = StandardScaler().fit_transform(features)

    tsne = TSNE(n_components=2, perplexity=30, learning_rate="auto", init="pca", random_state=42)
    X_tsne = tsne.fit_transform(X)
    plt.figure()
    for c in np.unique(y_true):
        idx = (y_true == c)
        plt.scatter(X_tsne[idx, 0], X_tsne[idx, 1], s=8, alpha=0.8, label=f"class {c}")
    plt.title(f"{title_prefix} t-SNE (features)".strip())
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "tsne.png"), dpi=300)
    plt.close()

    if _HAVE_UMAP:
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        X_umap = reducer.fit_transform(X)
        plt.figure()
        for c in np.unique(y_true):
            idx = (y_true == c)
            plt.scatter(X_umap[idx, 0], X_umap[idx, 1], s=8, alpha=0.8, label=f"class {c}")
        plt.title(f"{title_prefix} UMAP (features)".strip())
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "umap.png"), dpi=300)
        plt.close()
    else:
        with open(os.path.join(out_dir, "umap_missing.txt"), "w", encoding="utf-8") as f:
            f.write("umap-learn not installed. Install: pip install umap-learn\n")


# ---------------- NEW: 3-class ROC in one figure (mean±std across folds) ----------------
def _interp_roc_at_grid(fpr: np.ndarray, tpr: np.ndarray, fpr_grid: np.ndarray) -> np.ndarray:
    # ensure increasing fpr
    order = np.argsort(fpr)
    fpr = fpr[order]
    tpr = tpr[order]
    # np.interp requires ascending x
    tpr_i = np.interp(fpr_grid, fpr, tpr)
    tpr_i[0] = 0.0
    tpr_i[-1] = 1.0
    return tpr_i


def plot_roc_3class_meanstd(
    per_class_fold_curves: Dict[int, Dict[str, List[Tuple[np.ndarray, np.ndarray, float]]]],
    class_names: List[str],
    out_path: str,
    title: str = "ROC Curves (OvR) - 3 Classes",
    fpr_points: int = 1000
):
    """
    One plot includes 3 classes (OvR), each drawn as:
      - mean ROC curve across folds
      - shaded band = ±1 std across folds (after interpolating to common FPR grid)
      - legend shows AUC mean ± std (computed from fold AUCs)
    """
    fpr_grid = np.linspace(0.0, 1.0, fpr_points)

    plt.figure(figsize=(7.6, 5.6))
    ax = plt.gca()
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)

    for k, cname in enumerate(class_names):
        fold_rocs = per_class_fold_curves[k]["roc"]  # list of (fpr,tpr,auc)
        if len(fold_rocs) == 0:
            continue

        tprs = []
        aucs = []
        for fpr, tpr, roc_auc in fold_rocs:
            if fpr is None or tpr is None or len(fpr) < 2 or len(tpr) < 2:
                continue
            tprs.append(_interp_roc_at_grid(np.asarray(fpr), np.asarray(tpr), fpr_grid))
            aucs.append(float(roc_auc) if roc_auc == roc_auc else np.nan)

        if len(tprs) == 0:
            continue

        tprs = np.stack(tprs, axis=0)
        tpr_mean = np.mean(tprs, axis=0)
        tpr_std = np.std(tprs, axis=0, ddof=0)

        aucs = np.asarray(aucs, dtype=float)
        auc_mean = np.nanmean(aucs)
        auc_std = np.nanstd(aucs, ddof=0)

        ax.plot(
            fpr_grid, tpr_mean, linewidth=2.2,
            label=f"{cname} (AUC = {auc_mean:.3f} \u00b1 {auc_std:.3f})"
        )
        ax.fill_between(
            fpr_grid,
            np.clip(tpr_mean - tpr_std, 0.0, 1.0),
            np.clip(tpr_mean + tpr_std, 0.0, 1.0),
            alpha=0.18
        )

    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0, alpha=0.8)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    ax.set_title(title)

    # legend outside to mimic your example
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# ============================================================
# 9) Main
# ============================================================
def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--test_csv", type=str, default="test.csv")
    ap.add_argument("--train_csv", type=str, default="train.csv", help="Optional: also evaluate train set")

    ap.add_argument("--prott5_model_path", type=str, default="prot_t5_xl_uniref50")
    ap.add_argument("--proprime_model_path", type=str, default="ProPrime_650M_OGT_Prediction")

    ap.add_argument("--mode", type=str, default="bigru-fusion", choices=["late-fusion", "bigru-fusion"])

    ap.add_argument("--ckpt_dir", type=str, default="../train-T5+Prime/fusion_out",
                    help="Directory containing best_head_fold_{i}.pt")
    ap.add_argument("--folds", type=int, default=5)

    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=1024)

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--pin_memory", action="store_true")

    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--proj_dim", type=int, default=512)
    ap.add_argument("--gru_hidden", type=int, default=256)

    ap.add_argument("--num_classes", type=int, default=3)
    ap.add_argument(
        "--class_names",
        type=str,
        default="Mesophile,Thermophilic,Psychrophilic",
        help="Comma-separated names aligned with labels 0,1,2"
    )

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--deterministic", action="store_true")

    ap.add_argument("--out_dir", type=str, default="test_out")

    return ap.parse_args()


def eval_dataset_with_kfold_heads(
    *,
    name: str,
    dataloader: DataLoader,
    head: FusionHead,
    device: torch.device,
    t5_encoder,
    pp_encoder,
    args,
    class_names: List[str],
):
    num_classes = args.num_classes

    per_fold_rows = []
    per_fold_roc_micro = []
    per_fold_pr_micro = []
    cm_sum = np.zeros((num_classes, num_classes), dtype=np.int64)

    per_class_fold_curves = {k: {"roc": [], "pr": []} for k in range(num_classes)}

    fold_probs = []
    feats_for_ensemble = []
    y_true_ref = None
    ids_ref = None

    for fold in range(1, args.folds + 1):
        ckpt_path = os.path.join(args.ckpt_dir, f"best_head_fold_{fold}.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

        ids, y_true, y_pred, y_prob, feats = infer_one_fold(
            ckpt_path, head, args.mode, dataloader, device,
            t5_encoder, pp_encoder, num_classes
        )

        if y_true_ref is None:
            y_true_ref = y_true
            ids_ref = ids
        else:
            if not np.array_equal(y_true_ref, y_true):
                raise RuntimeError(f"[{name}] y_true mismatch across folds (should be identical).")
            if not np.array_equal(ids_ref, ids):
                raise RuntimeError(f"[{name}] ids mismatch across folds (dataloader order changed).")

        fold_probs.append(y_prob)
        feats_for_ensemble.append(feats)

        acc = float(accuracy_score(y_true, y_pred))
        f1m = float(f1_score(y_true, y_pred, average="macro"))
        mcc = float(matthews_corrcoef(y_true, y_pred))

        cm_stats = compute_overall_and_perclass_from_cm(y_true, y_pred, num_classes)
        cm = cm_stats["confusion_matrix"]
        cm_sum += cm

        aucs = compute_auc_metrics_overall_and_perclass(y_true, y_prob, num_classes)

        row = {
            "dataset": name,
            "fold": fold,
            "ACC": acc,
            "MCC": mcc,
            "SN_macro": cm_stats["SN_macro"],
            "SP_macro": cm_stats["SP_macro"],
            "PRE_macro": cm_stats["PRE_macro"],
            "F1_macro": f1m,
            "AUROC_macro": aucs["AUROC_macro"],
            "AUROC_micro": aucs["AUROC_micro"],
            "AUPRC_macro": aucs["AUPRC_macro"],
            "AUPRC_micro": aucs["AUPRC_micro"],
        }

        for k, cname in enumerate(class_names):
            row[f"{cname}_SN"] = cm_stats[f"SN_c{k}"]
            row[f"{cname}_SP"] = cm_stats[f"SP_c{k}"]
            row[f"{cname}_PRE"] = cm_stats[f"PRE_c{k}"]
            row[f"{cname}_F1"] = cm_stats[f"F1_c{k}"]
            row[f"{cname}_AUROC"] = aucs[f"AUROC_c{k}"]
            row[f"{cname}_AUPRC"] = aucs[f"AUPRC_c{k}"]

        per_fold_rows.append(row)

        roc_c, pr_c = roc_pr_micro_curve(y_true, y_prob, num_classes)
        per_fold_roc_micro.append(roc_c)
        per_fold_pr_micro.append(pr_c)

        for k in range(num_classes):
            roc_k, pr_k = roc_pr_per_class_curve(y_true, y_prob, k)
            per_class_fold_curves[k]["roc"].append(roc_k)
            per_class_fold_curves[k]["pr"].append(pr_k)

        print(
            f"[{name}][Fold {fold}] ACC={acc:.4f} F1m={f1m:.4f} MCC={mcc:.4f} "
            f"AUROC_micro={aucs['AUROC_micro']:.4f} AUPRC_micro={aucs['AUPRC_micro']:.4f}"
        )

    probs_ens = np.mean(np.stack(fold_probs, axis=0), axis=0)  # (N,C)
    y_true = y_true_ref
    y_pred_ens = probs_ens.argmax(axis=1)

    acc_ens = float(accuracy_score(y_true, y_pred_ens))
    f1m_ens = float(f1_score(y_true, y_pred_ens, average="macro"))
    mcc_ens = float(matthews_corrcoef(y_true, y_pred_ens))
    cm_stats_ens = compute_overall_and_perclass_from_cm(y_true, y_pred_ens, num_classes)
    aucs_ens = compute_auc_metrics_overall_and_perclass(y_true, probs_ens, num_classes)

    roc_ens_micro, pr_ens_micro = roc_pr_micro_curve(y_true, probs_ens, num_classes)

    per_class_ens_curves = {}
    for k in range(num_classes):
        roc_k, pr_k = roc_pr_per_class_curve(y_true, probs_ens, k)
        per_class_ens_curves[k] = {"roc": roc_k, "pr": pr_k}

    feats_mean = np.mean(np.stack(feats_for_ensemble, axis=0), axis=0)

    ensemble_metrics = {
        "dataset": name,
        "ACC": acc_ens,
        "F1_macro": f1m_ens,
        "MCC": mcc_ens,
        "SN_macro": cm_stats_ens["SN_macro"],
        "SP_macro": cm_stats_ens["SP_macro"],
        "PRE_macro": cm_stats_ens["PRE_macro"],
        "AUROC_macro": aucs_ens["AUROC_macro"],
        "AUROC_micro": aucs_ens["AUROC_micro"],
        "AUPRC_macro": aucs_ens["AUPRC_macro"],
        "AUPRC_micro": aucs_ens["AUPRC_micro"],
    }
    for k, cname in enumerate(class_names):
        ensemble_metrics[f"{cname}_SN"] = cm_stats_ens[f"SN_c{k}"]
        ensemble_metrics[f"{cname}_SP"] = cm_stats_ens[f"SP_c{k}"]
        ensemble_metrics[f"{cname}_PRE"] = cm_stats_ens[f"PRE_c{k}"]
        ensemble_metrics[f"{cname}_F1"] = cm_stats_ens[f"F1_c{k}"]
        ensemble_metrics[f"{cname}_AUROC"] = aucs_ens[f"AUROC_c{k}"]
        ensemble_metrics[f"{cname}_AUPRC"] = aucs_ens[f"AUPRC_c{k}"]

    return {
        "per_fold_rows": per_fold_rows,
        "per_fold_roc_micro": per_fold_roc_micro,
        "per_fold_pr_micro": per_fold_pr_micro,
        "cm_sum": cm_sum,
        "probs_ens": probs_ens,
        "feats_mean": feats_mean,
        "y_true": y_true,
        "ids": ids_ref,
        "roc_ens_micro": roc_ens_micro,
        "pr_ens_micro": pr_ens_micro,
        "per_class_fold_curves": per_class_fold_curves,
        "per_class_ens_curves": per_class_ens_curves,
        "ensemble_metrics": ensemble_metrics
    }


def save_curve_points_csv(
    out_csv: str,
    dataset_name: str,
    class_names: List[str],
    per_class_fold_curves: Dict[int, Dict[str, List[Tuple[np.ndarray, np.ndarray, float]]]],
    per_class_ens_curves: Dict[int, Dict[str, Tuple[np.ndarray, np.ndarray, float]]],
    curve_type: str = "roc"
):
    rows = []
    for k, cname in enumerate(class_names):
        fold_list = per_class_fold_curves[k][curve_type]
        for fold_idx, curve in enumerate(fold_list, start=1):
            x, y, score = curve
            for i in range(len(x)):
                rows.append({
                    "dataset": dataset_name,
                    "class_idx": k,
                    "class_name": cname,
                    "source": f"fold{fold_idx}",
                    "score": float(score) if score == score else np.nan,
                    "x": float(x[i]),
                    "y": float(y[i]),
                })
        x, y, score = per_class_ens_curves[k][curve_type]
        for i in range(len(x)):
            rows.append({
                "dataset": dataset_name,
                "class_idx": k,
                "class_name": cname,
                "source": "ensemble",
                "score": float(score) if score == score else np.nan,
                "x": float(x[i]),
                "y": float(y[i]),
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[SAVE] {out_csv}")


def save_prob_dist_csv(out_csv: str, ids: np.ndarray, y_true: np.ndarray, probs: np.ndarray, class_names: List[str]):
    y_pred = probs.argmax(axis=1)

    df = pd.DataFrame({
        "id": ids.astype(str),
        "y_true": y_true.astype(int),
        "y_pred": y_pred.astype(int),
    })
    for k, cname in enumerate(class_names):
        df[f"prob_{cname}"] = probs[:, k]

    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[SAVE] {out_csv}")


def main():
    args = parse_args()
    ensure_dir(args.out_dir)
    set_seed(args.seed, deterministic=args.deterministic)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    print(f"[INFO] device={device} | mode={args.mode}")

    class_names = [x.strip() for x in args.class_names.split(",")]
    if len(class_names) != args.num_classes:
        class_names = [str(i) for i in range(args.num_classes)]

    # Load frozen encoders
    t5_tokenizer, t5_encoder, pp_tokenizer, pp_encoder = load_frozen_encoders(
        args.prott5_model_path, args.proprime_model_path, device
    )

    # Build head skeleton (weights loaded per fold)
    head = FusionHead(
        mode=args.mode,
        dim_t5=1024,
        dim_pp=1280,  # if your ProPrime hidden dim differs, change here
        num_classes=args.num_classes,
        dropout=args.dropout,
        proj_dim=args.proj_dim,
        gru_hidden=args.gru_hidden,
    ).to(device)

    # --------------------------
    # Test dataloader
    # --------------------------
    ds_test = ProteinCSVDataset(args.test_csv)
    from functools import partial
    dl_test = DataLoader(
        ds_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        collate_fn=partial(collate_raw_batch, prott5_tokenizer=t5_tokenizer, proprime_tokenizer=pp_tokenizer, max_len=args.max_len)
    )

    # Evaluate test
    res_test = eval_dataset_with_kfold_heads(
        name="test",
        dataloader=dl_test,
        head=head,
        device=device,
        t5_encoder=t5_encoder,
        pp_encoder=pp_encoder,
        args=args,
        class_names=class_names
    )

    # Save per-fold metrics (test)
    df_test = pd.DataFrame(res_test["per_fold_rows"])
    df_test_path = os.path.join(args.out_dir, "metrics_per_fold_test.csv")
    df_test.to_csv(df_test_path, index=False, encoding="utf-8-sig")
    print(f"[SAVE] {df_test_path}")

    # Save per-sample prob dist CSV (test)
    save_prob_dist_csv(
        out_csv=os.path.join(args.out_dir, "prob_dist_test.csv"),
        ids=res_test["ids"],
        y_true=res_test["y_true"],
        probs=res_test["probs_ens"],
        class_names=class_names
    )

    # micro plots (test)
    plot_roc_micro(res_test["per_fold_roc_micro"], res_test["roc_ens_micro"], os.path.join(args.out_dir, "roc_micro_test.png"))
    plot_pr_micro(res_test["per_fold_pr_micro"], res_test["pr_ens_micro"], os.path.join(args.out_dir, "pr_micro_test.png"))

    # per-class plots (test)
    for cname in class_names:
        out_roc = os.path.join(args.out_dir, f"roc_class_{cname}_test.png")
        out_pr = os.path.join(args.out_dir, f"pr_class_{cname}_test.png")
        k = class_names.index(cname)
        plot_roc_pr_per_class(
            res_test["per_class_fold_curves"][k],
            res_test["per_class_ens_curves"][k],
            cname, out_roc, out_pr
        )

    # NEW: 3-class ROC in one figure (test)
    plot_roc_3class_meanstd(
        per_class_fold_curves=res_test["per_class_fold_curves"],
        class_names=class_names,
        out_path=os.path.join(args.out_dir, "roc_3class_test.png"),
        title="ROC Curves (OvR) - Mesophile / Thermophilic / Psychrophilic"
    )

    # Curve points CSV (test)
    save_curve_points_csv(
        out_csv=os.path.join(args.out_dir, "roc_points_per_class_test.csv"),
        dataset_name="test",
        class_names=class_names,
        per_class_fold_curves=res_test["per_class_fold_curves"],
        per_class_ens_curves=res_test["per_class_ens_curves"],
        curve_type="roc"
    )
    save_curve_points_csv(
        out_csv=os.path.join(args.out_dir, "pr_points_per_class_test.csv"),
        dataset_name="test",
        class_names=class_names,
        per_class_fold_curves=res_test["per_class_fold_curves"],
        per_class_ens_curves=res_test["per_class_ens_curves"],
        curve_type="pr"
    )

    # Confusion matrix (sum over folds) for test
    plot_confusion(
        res_test["cm_sum"], class_names,
        os.path.join(args.out_dir, "confusion_matrix_sum_test.png"),
        normalize=False, title="Confusion Matrix (sum over folds) - TEST"
    )
    plot_confusion(
        res_test["cm_sum"], class_names,
        os.path.join(args.out_dir, "confusion_matrix_sum_norm_test.png"),
        normalize=True, title="Confusion Matrix (row-normalized) - TEST"
    )

    # Save arrays + dimred (test)
    np.save(os.path.join(args.out_dir, "features_mean_test.npy"), res_test["feats_mean"])
    np.save(os.path.join(args.out_dir, "probs_ensemble_test.npy"), res_test["probs_ens"])
    np.save(os.path.join(args.out_dir, "y_true_test.npy"), res_test["y_true"])
    plot_tsne_umap(res_test["feats_mean"], res_test["y_true"], os.path.join(args.out_dir, "dimred_test"), title_prefix="TEST")

    # --------------------------
    # Optional: Train evaluation + dimred + prob CSV + 3class ROC
    # --------------------------
    have_train = bool(args.train_csv.strip())
    res_train = None
    if have_train:
        ds_train = ProteinCSVDataset(args.train_csv)
        dl_train = DataLoader(
            ds_train,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            collate_fn=partial(collate_raw_batch, prott5_tokenizer=t5_tokenizer, proprime_tokenizer=pp_tokenizer, max_len=args.max_len)
        )

        res_train = eval_dataset_with_kfold_heads(
            name="train",
            dataloader=dl_train,
            head=head,
            device=device,
            t5_encoder=t5_encoder,
            pp_encoder=pp_encoder,
            args=args,
            class_names=class_names
        )

        df_train = pd.DataFrame(res_train["per_fold_rows"])
        df_train_path = os.path.join(args.out_dir, "metrics_per_fold_train.csv")
        df_train.to_csv(df_train_path, index=False, encoding="utf-8-sig")
        print(f"[SAVE] {df_train_path}")

        save_prob_dist_csv(
            out_csv=os.path.join(args.out_dir, "prob_dist_train.csv"),
            ids=res_train["ids"],
            y_true=res_train["y_true"],
            probs=res_train["probs_ens"],
            class_names=class_names
        )

        # NEW: 3-class ROC in one figure (train)
        plot_roc_3class_meanstd(
            per_class_fold_curves=res_train["per_class_fold_curves"],
            class_names=class_names,
            out_path=os.path.join(args.out_dir, "roc_3class_train.png"),
            title="ROC Curves (OvR) - Mesophile / Thermophilic / Psychrophilic (TRAIN)"
        )

        np.save(os.path.join(args.out_dir, "features_mean_train.npy"), res_train["feats_mean"])
        np.save(os.path.join(args.out_dir, "probs_ensemble_train.npy"), res_train["probs_ens"])
        np.save(os.path.join(args.out_dir, "y_true_train.npy"), res_train["y_true"])
        plot_tsne_umap(res_train["feats_mean"], res_train["y_true"], os.path.join(args.out_dir, "dimred_train"), title_prefix="TRAIN")

    # --------------------------
    # Summary JSON (test + train)
    # --------------------------
    summary = {"test": {}, "train": {}}

    for col in df_test.columns:
        if col in ["dataset", "fold"]:
            continue
        if pd.api.types.is_numeric_dtype(df_test[col]):
            summary["test"][f"{col}_mean"] = float(df_test[col].mean())
            summary["test"][f"{col}_std"] = float(df_test[col].std(ddof=0))
    summary["test"]["ensemble"] = res_test["ensemble_metrics"]

    if have_train and res_train is not None:
        df_train_loaded = pd.read_csv(os.path.join(args.out_dir, "metrics_per_fold_train.csv"))
        for col in df_train_loaded.columns:
            if col in ["dataset", "fold"]:
                continue
            if pd.api.types.is_numeric_dtype(df_train_loaded[col]):
                summary["train"][f"{col}_mean"] = float(df_train_loaded[col].mean())
                summary["train"][f"{col}_std"] = float(df_train_loaded[col].std(ddof=0))
        summary["train"]["ensemble"] = res_train["ensemble_metrics"]

    with open(os.path.join(args.out_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n[DONE] Outputs saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
