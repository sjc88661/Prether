#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import time
import random
import argparse
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import T5Tokenizer, T5EncoderModel
from transformers import AutoTokenizer, AutoModel


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
# 1) Attention pooling (same as training)
# ============================================================
def attention_pooling(hidden: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    weights = hidden.mean(dim=-1)  # (B, L)
    weights = weights.masked_fill(attn_mask == 0, -1e9)
    weights = torch.softmax(weights, dim=1)
    pooled = torch.sum(hidden * weights.unsqueeze(-1), dim=1)  # (B, D)
    return pooled


# ============================================================
# 2) Robust hidden extraction (same as training)
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


def extract_pooled_if_available(outputs) -> Optional[torch.Tensor]:
    if hasattr(outputs, "sequence_hidden_states") and torch.is_tensor(outputs.sequence_hidden_states):
        if outputs.sequence_hidden_states.dim() == 2:
            return outputs.sequence_hidden_states
    if isinstance(outputs, dict) and "sequence_hidden_states" in outputs and torch.is_tensor(outputs["sequence_hidden_states"]):
        if outputs["sequence_hidden_states"].dim() == 2:
            return outputs["sequence_hidden_states"]
    return None


# ============================================================
# 3) Data: CSV or FASTA (label ignored)
# ============================================================
def read_fasta(path: str) -> Tuple[List[str], List[str]]:
    ids, seqs = [], []
    cur_id = None
    cur_seq = []

    def flush():
        nonlocal cur_id, cur_seq
        if cur_id is None:
            return
        ids.append(cur_id)
        seqs.append("".join(cur_seq).strip())
        cur_id, cur_seq = None, []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                header = line[1:].strip()
                # if ends with |label, strip it from id (optional)
                parts = header.split("|")
                if len(parts) >= 2 and parts[-1].isdigit():
                    cur_id = "|".join(parts[:-1])
                else:
                    cur_id = header
            else:
                cur_seq.append(line)
    flush()
    return ids, seqs


class ProteinDataset(Dataset):
    def __init__(self, path: str, input_type: str = "csv"):
        if input_type == "csv":
            df = pd.read_csv(path)
            self.ids = df.iloc[:, 0].astype(str).tolist()
            self.seqs = df.iloc[:, 1].astype(str).tolist()
        else:
            self.ids, self.seqs = read_fasta(path)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return {"id": self.ids[idx], "seq": self.seqs[idx]}


# ============================================================
# 4) Cache helpers (same layout as training; label not needed)
# ============================================================
def cache_file_path(cache_dir: str, mode: str, sample_id: str) -> str:
    safe_id = sample_id.replace("/", "_")
    return os.path.join(cache_dir, mode, f"{safe_id}.pt")


def collate_raw_batch(batch, prott5_tokenizer, proprime_tokenizer, max_len: int):
    ids = [x["id"] for x in batch]
    seqs = [x["seq"] for x in batch]

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
        "t5_input_ids": enc_t5["input_ids"],
        "t5_attn_mask": enc_t5["attention_mask"],
        "pp_input_ids": enc_pp["input_ids"],
        "pp_attn_mask": enc_pp["attention_mask"],
    }


def collate_cache_late(batch, cache_dir: str, mode_dirname: str):
    e_t5_list, e_pp_list, ids = [], [], []
    for x in batch:
        sid = x["id"]
        ids.append(sid)
        fp = cache_file_path(cache_dir, mode_dirname, sid)
        obj = torch.load(fp, map_location="cpu")
        e_t5_list.append(obj["e_t5"].float())
        e_pp_list.append(obj["e_pp"].float())
    return {
        "ids": ids,
        "e_t5": torch.stack(e_t5_list, dim=0),
        "e_pp": torch.stack(e_pp_list, dim=0),
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


def collate_cache_bigru(batch, cache_dir: str, mode_dirname: str):
    ids = []
    h_t5_list, m_t5_list = [], []
    h_pp_list, m_pp_list = [], []

    for x in batch:
        sid = x["id"]
        ids.append(sid)
        fp = cache_file_path(cache_dir, mode_dirname, sid)
        obj = torch.load(fp, map_location="cpu")
        h_t5_list.append(obj["h_t5"].float())
        m_t5_list.append(obj["m_t5"].long())
        h_pp_list.append(obj["h_pp"].float())
        m_pp_list.append(obj["m_pp"].long())

    h_t5 = _pad_2d(h_t5_list, 0.0)
    m_t5 = _pad_1d(m_t5_list, 0)
    h_pp = _pad_2d(h_pp_list, 0.0)
    m_pp = _pad_1d(m_pp_list, 0)

    return {
        "ids": ids,
        "h_t5": h_t5,
        "m_t5": m_t5,
        "h_pp": h_pp,
        "m_pp": m_pp,
    }


# ============================================================
# 5) Model (same as training)
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


# ============================================================
# 6) Encoders
# ============================================================
def load_frozen_encoders(prott5_model_path: str, proprime_model_path: str, device: torch.device):
    t5_tokenizer = T5Tokenizer.from_pretrained(prott5_model_path, do_lower_case=False)
    t5_encoder = T5EncoderModel.from_pretrained(prott5_model_path).to(device)
    t5_encoder.eval()
    t5_encoder.requires_grad_(False)

    pp_tokenizer = AutoTokenizer.from_pretrained(proprime_model_path, trust_remote_code=True)
    pp_encoder = AutoModel.from_pretrained(proprime_model_path, trust_remote_code=True).to(device)
    pp_encoder.eval()
    pp_encoder.requires_grad_(False)

    return t5_tokenizer, t5_encoder, pp_tokenizer, pp_encoder


@torch.no_grad()
def build_cache(dataset: ProteinDataset, args, device: torch.device):
    ensure_dir(args.cache_dir)
    ensure_dir(os.path.join(args.cache_dir, args.mode))

    t5_tokenizer, t5_encoder, pp_tokenizer, pp_encoder = load_frozen_encoders(
        args.prott5_model_path, args.proprime_model_path, device
    )

    bs = args.cache_batch_size
    pbar = tqdm(range(0, len(dataset), bs), desc=f"[Cache] mode={args.mode}")
    for s in pbar:
        e = min(s + bs, len(dataset))
        batch_items = [dataset[i] for i in range(s, e)]

        todo = []
        for x in batch_items:
            fp = cache_file_path(args.cache_dir, args.mode, x["id"])
            if (not os.path.exists(fp)) or args.cache_overwrite:
                todo.append(x)
        if not todo:
            continue

        todo_ids = [x["id"] for x in todo]
        todo_seqs = [x["seq"] for x in todo]

        t5_text = [" ".join(list(sq)) for sq in todo_seqs]
        enc_t5 = t5_tokenizer(
            t5_text, max_length=args.max_len, truncation=True,
            padding="max_length", return_tensors="pt"
        )
        enc_t5 = {k: v.to(device) for k, v in enc_t5.items()}

        enc_pp = pp_tokenizer(
            todo_seqs, max_length=args.max_len, truncation=True,
            padding="max_length", return_tensors="pt"
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

        if args.mode == "late-fusion":
            e_t5 = attention_pooling(h_t5, m_t5)
            if h_pp is None:
                pooled_pp = extract_pooled_if_available(out_pp)
                if pooled_pp is None:
                    raise RuntimeError("ProPrime cache failed: cannot extract token hidden or pooled embeddings.")
                e_pp = pooled_pp
            else:
                e_pp = attention_pooling(h_pp, m_pp)

            for i, sid in enumerate(todo_ids):
                fp = cache_file_path(args.cache_dir, args.mode, sid)
                torch.save({"e_t5": e_t5[i].detach().cpu().half(),
                            "e_pp": e_pp[i].detach().cpu().half()}, fp)
        else:
            if h_pp is None:
                raise RuntimeError("bigru-fusion requires ProPrime token-level hidden (B,L,D), but got None.")
            for i, sid in enumerate(todo_ids):
                fp = cache_file_path(args.cache_dir, args.mode, sid)
                lt5 = int(m_t5[i].sum().item())
                lpp = int(m_pp[i].sum().item())
                torch.save({
                    "h_t5": h_t5[i, :lt5].detach().cpu().half(),
                    "m_t5": m_t5[i, :lt5].detach().cpu().to(torch.int8),
                    "h_pp": h_pp[i, :lpp].detach().cpu().half(),
                    "m_pp": m_pp[i, :lpp].detach().cpu().to(torch.int8),
                }, fp)

    del t5_encoder, pp_encoder
    torch.cuda.empty_cache()


# ============================================================
# 7) AMP
# ============================================================
def get_amp(device: torch.device, fp16: bool):
    enabled = fp16 and device.type == "cuda"
    if hasattr(torch, "amp"):
        return enabled, torch.amp.autocast
    return enabled, torch.cuda.amp.autocast


# ============================================================
# 8) Prediction
# ============================================================
@torch.no_grad()
def predict_one_fold(
    *,
    head: FusionHead,
    loader: DataLoader,
    device: torch.device,
    mode: str,
    use_cache: bool,
    fp16: bool,
    t5_encoder=None,
    pp_encoder=None,
):
    head.eval()
    amp_enabled, autocast = get_amp(device, fp16)

    all_ids = []
    all_prob = []

    for batch in tqdm(loader, desc="Predict", leave=False):
        ids = batch["ids"]

        if use_cache:
            if mode == "late-fusion":
                e_t5 = batch["e_t5"].to(device)
                e_pp = batch["e_pp"].to(device)
                with autocast(device_type=device.type, enabled=amp_enabled) if hasattr(torch, "amp") else autocast(enabled=amp_enabled):
                    logits = head(e_t5, e_pp)
            else:
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

            if mode == "late-fusion":
                e_t5 = attention_pooling(h_t5, t5_attn_mask)
                if h_pp is None:
                    pooled_pp = extract_pooled_if_available(out_pp)
                    if pooled_pp is None:
                        raise RuntimeError("ProPrime output has no token hidden nor pooled embedding.")
                    e_pp = pooled_pp
                else:
                    e_pp = attention_pooling(h_pp, pp_attn_mask)

                with autocast(device_type=device.type, enabled=amp_enabled) if hasattr(torch, "amp") else autocast(enabled=amp_enabled):
                    logits = head(e_t5, e_pp)
            else:
                if h_pp is None:
                    raise RuntimeError("bigru-fusion requires ProPrime token-level hidden (B,L,D), but got None.")
                with autocast(device_type=device.type, enabled=amp_enabled) if hasattr(torch, "amp") else autocast(enabled=amp_enabled):
                    logits = head(h_t5, t5_attn_mask, h_pp, pp_attn_mask)

        prob = torch.softmax(logits.float(), dim=1).detach().cpu().numpy()
        all_prob.append(prob)
        all_ids.extend(ids)

    prob_all = np.concatenate(all_prob, axis=0)
    return all_ids, prob_all


# ============================================================
# 9) CLI
# ============================================================
def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--input_path", type=str, required=True, help="test.csv or test.fasta")
    ap.add_argument("--input_type", type=str, default="fasta", choices=["csv", "fasta"])

    ap.add_argument("--ckpt_dir", type=str, default="../train-T5+Prime/fusion_out",
                    help="dir containing best_head_fold_{i}.pt")
    ap.add_argument("--k_folds", type=int, default=5)

    ap.add_argument("--prott5_model_path", type=str, default="prot_t5_xl_uniref50")
    ap.add_argument("--proprime_model_path", type=str, default="ProPrime_650M_OGT_Prediction")

    # override if needed; otherwise inferred from fold1 ckpt['args']
    ap.add_argument("--mode", type=str, default="bigru-fusion", choices=["", "late-fusion", "bigru-fusion"])
    ap.add_argument("--dim_pp", type=int, default=1280, help="ProPrime hidden size if different")
    ap.add_argument("--proj_dim", type=int, default=512)
    ap.add_argument("--gru_hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.3)

    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--pin_memory", action="store_true")

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--fp16", action="store_true")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--deterministic", action="store_true")

    ap.add_argument("--out_dir", type=str, default="pred_out")

    # cache
    ap.add_argument("--use_cache", action="store_true")
    ap.add_argument("--cache_dir", type=str, default="fusion_cache")
    ap.add_argument("--cache_batch_size", type=int, default=4)
    ap.add_argument("--cache_overwrite", action="store_true")

    ap.add_argument("--save_class_name", action="store_true",
                    help="Also save pred_name using --class_names")
    ap.add_argument("--class_names", type=str, default="嗜温,嗜热,嗜冷")

    return ap.parse_args()


# ============================================================
# 10) Main
# ============================================================
def main():
    args = parse_args()
    ensure_dir(args.out_dir)
    set_seed(args.seed, deterministic=args.deterministic)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    print(f"[INFO] device={device} | fp16={args.fp16}")

    # infer hyperparams from fold1 checkpoint args (to match training)
    fold1 = os.path.join(args.ckpt_dir, "best_head_fold_1.pt")
    if not os.path.exists(fold1):
        raise FileNotFoundError(f"Missing {fold1}")
    ck1 = torch.load(fold1, map_location="cpu")
    ck_args = ck1.get("args", {}) if isinstance(ck1, dict) else {}

    if args.mode == "":
        args.mode = ck_args.get("mode", "bigru-fusion")
    if "proj_dim" in ck_args:
        args.proj_dim = int(ck_args["proj_dim"])
    if "gru_hidden" in ck_args:
        args.gru_hidden = int(ck_args["gru_hidden"])
    if "dropout" in ck_args:
        args.dropout = float(ck_args["dropout"])

    print(f"[INFO] mode={args.mode} | proj_dim={args.proj_dim} | gru_hidden={args.gru_hidden} | dropout={args.dropout} | dim_pp={args.dim_pp}")

    # dataset
    ds = ProteinDataset(args.input_path, input_type=args.input_type)

    # cache build
    if args.use_cache:
        print("[INFO] Building/Checking cache ...")
        build_cache(ds, args, device)

    # load encoders once (only used if not use_cache)
    t5_tok, t5_enc, pp_tok, pp_enc = load_frozen_encoders(args.prott5_model_path, args.proprime_model_path, device)

    # dataloader
    from functools import partial
    if args.use_cache:
        if args.mode == "late-fusion":
            collate_fn = partial(collate_cache_late, cache_dir=args.cache_dir, mode_dirname=args.mode)
        else:
            collate_fn = partial(collate_cache_bigru, cache_dir=args.cache_dir, mode_dirname=args.mode)
    else:
        collate_fn = partial(collate_raw_batch, prott5_tokenizer=t5_tok, proprime_tokenizer=pp_tok, max_len=args.max_len)

    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=args.pin_memory,
        collate_fn=collate_fn
    )

    # predict folds
    fold_probs = []
    ids_order = None

    for fold in range(1, args.k_folds + 1):
        ckpt_path = os.path.join(args.ckpt_dir, f"best_head_fold_{fold}.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Missing {ckpt_path}")

        ck = torch.load(ckpt_path, map_location=device)
        state = ck["head"] if isinstance(ck, dict) and "head" in ck else ck

        head = FusionHead(
            mode=args.mode,
            dim_t5=1024,
            dim_pp=args.dim_pp,
            num_classes=3,
            dropout=args.dropout,
            proj_dim=args.proj_dim,
            gru_hidden=args.gru_hidden,
        ).to(device)
        head.load_state_dict(state, strict=True)
        head.eval()

        ids, prob = predict_one_fold(
            head=head, loader=loader, device=device,
            mode=args.mode, use_cache=args.use_cache, fp16=args.fp16,
            t5_encoder=t5_enc, pp_encoder=pp_enc
        )

        if ids_order is None:
            ids_order = ids
        else:
            if ids != ids_order:
                raise RuntimeError("Sample order mismatch across folds (should not happen).")

        fold_probs.append(prob)
        print(f"[Fold {fold}] done. prob shape={prob.shape}")

    # ensemble
    probs_stack = np.stack(fold_probs, axis=0)  # (K, N, 3)
    prob_mean = probs_stack.mean(axis=0)        # (N, 3)
    pred_ens = np.argmax(prob_mean, axis=1)

    # save per-fold
    df_pf = pd.DataFrame({"id": ids_order})
    df_pf["pred_ens"] = pred_ens
    for c in range(3):
        df_pf[f"prob_c{c}_ens"] = prob_mean[:, c]

    for f in range(args.k_folds):
        df_pf[f"pred_fold{f+1}"] = np.argmax(fold_probs[f], axis=1)
        for c in range(3):
            df_pf[f"prob_c{c}_fold{f+1}"] = fold_probs[f][:, c]

    if args.save_class_name:
        names = [x.strip() for x in args.class_names.split(",")]
        if len(names) != 3:
            raise ValueError(f"--class_names must have 3 names, got {names}")
        df_pf["pred_name"] = [names[i] for i in pred_ens]

    out_pf = os.path.join(args.out_dir, "predictions_per_fold.csv")
    df_pf.to_csv(out_pf, index=False, encoding="utf-8-sig")
    print(f"[SAVE] {out_pf}")

    # save ensemble only
    df_ens = pd.DataFrame({"id": ids_order})
    for c in range(3):
        df_ens[f"prob_c{c}"] = prob_mean[:, c]
    df_ens["pred"] = pred_ens
    if args.save_class_name:
        df_ens["pred_name"] = df_pf["pred_name"].values

    out_ens = os.path.join(args.out_dir, "predictions_ensemble.csv")
    df_ens.to_csv(out_ens, index=False, encoding="utf-8-sig")
    print(f"[SAVE] {out_ens}")

    print("[DONE]")


if __name__ == "__main__":
    main()
