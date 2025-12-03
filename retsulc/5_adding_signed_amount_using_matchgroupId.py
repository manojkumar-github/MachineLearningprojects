"""
End-to-end optimized pipeline (Option A: use MatchGroupId for contrastive positives)

Features:
- Hybrid character encoder (Per-field CNN -> Transformer over fields)
- SignedAmount included as a learnable feature inside the encoder
- Contrastive pretraining (InfoNCE / NT-Xent) using MatchGroupId to form positive pairs
- HDBSCAN clustering with automatic hyperparameter grid search (silhouette on non-noise)
- FAST greedy matching (O(k log k) per cluster) to enforce sum(SignedAmount) ~= 0
- Optional pairwise evaluation (precision/recall/F1) using MatchGroupId

Usage:
- Replace example `df` creation with pd.read_csv("transactions.csv") for your dataset.
- Tune hyperparams at the top: BATCH_SIZE, EPOCHS, PROJ_DIM, HDBSCAN grids, tolerances.
"""

import os
import sys
import random
import math
import numpy as np
import pandas as pd
from itertools import combinations
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

# Try import hdbscan (install if necessary)
try:
    import hdbscan
except Exception:
    os.system(f"{sys.executable} -m pip install hdbscan --quiet")
    import hdbscan

from scipy.optimize import linear_sum_assignment  # kept for completeness if needed

# -----------------------
# CONFIG (tune for your data)
# -----------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# contrastive training hyperparams
BATCH_SIZE = 256
EPOCHS = 40
LR = 1e-3
TEMPERATURE = 0.5
PROJ_DIM = 128      # final embedding dim used for clustering
CNN_CHAR_EMBED = 16
CNN_OUT_CH = 64

# Sequence/fields (confirmed by user)
FIELDS = [
    "TransactionRefNo",
    "MerchantRefNum",
    "AcquireRefNumber",
    "WebOrderNumber",
    "PONumber",
    "CardNo"
]
SEQ_LEN = 64   # characters per field

# HDBSCAN hyperparameter search grid
HDBSCAN_MIN_CLUSTER_SIZE = [3, 5, 10]
HDBSCAN_MIN_SAMPLES = [1, 3, 5]

# greedy matching tolerances
AMOUNT_ABS_TOL = 0.01   # absolute tolerance (e.g., currency units)
AMOUNT_REL_TOL = 0.01   # relative tolerance (fraction)
MAX_MULTIWAY = 4        # brute-force multi-way up to this size

# -----------------------
# 0) Load data
# -----------------------
# Replace with your CSV in production
# df = pd.read_csv("transactions.csv", dtype=str)  # then convert numeric columns appropriately
data = {
    "DocumentDate": ["02/01/2025", "02/01/2025"],
    "DocType": ["unknown", "unknown"],
    "TransactionType": ["SAP", "SAP"],
    "BankTrfRef": ["unknown", "unknown"],
    "Amount": [7800,  -7800],
    "TransactionRefNo": ["W1dsfsafjdjfb", "W1dsfsafjdjfb"],
    "MerchantRefNum": ["777741344598", "777741344598"],
    "CR_DR": ["CR", "DR"],
    "GLRecordID": ["unknown", "unknown"],
    "OID": ["unknown", "unknown"],
    "PONumber": ["13u350u5u05", "13u350u5u05"],
    "CardNo": ["13415555531535", "13415555531535"],
    "AccountingDocNum": ["KD0nfdkk", "KD0nfdkk"],
    "AcquireRefNumber": ["unknown", "unknown"],
    "WebOrderNumber": ["W1342421414", "W1342421414"],
    "MatchGroupId": ["14443553", "14443553"],
    "Source": ["SAP", "SAP"],
    "SourceType": ["Internal", "Internal"]
}
df = pd.DataFrame(data)

# Ensure numeric types
df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0.0)
df["DocumentDate"] = pd.to_datetime(df["DocumentDate"], errors="coerce", dayfirst=True)

# -----------------------
# 1) Preprocessing
# -----------------------
# Drop columns that are uninformative if present (user requested earlier)
drop_cols = ["ReceiptNumber", "RefDocument", "Assignment", "StoreNumber", "AuthCode"]
for c in drop_cols:
    if c in df.columns:
        df = df.drop(columns=c)

# SignedAmount: CR -> +, DR -> -
df["CR_DR"] = df["CR_DR"].fillna("CR")
df["SignedAmount"] = df["Amount"].astype(float) * df["CR_DR"].map({"CR": 1.0, "DR": -1.0}).fillna(1.0)

# label-encode small categorical columns (optional features)
label_cols = [c for c in ["DocType", "TransactionType", "Source", "SourceType"] if c in df.columns]
for col in label_cols:
    le = LabelEncoder()
    df[col + "_enc"] = le.fit_transform(df[col].astype(str))

# -----------------------
# 2) Build char vocabulary from chosen fields
# -----------------------
all_text = ""
for f in FIELDS:
    if f in df.columns:
        all_text += " ".join(df[f].astype(str).tolist()) + " "
chars = sorted(set([c for c in all_text]))
if len(chars) == 0:
    chars = list("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_-/.")
char_to_idx = {c: i+1 for i, c in enumerate(chars)}  # reserve 0 for padding
IDX_PAD = 0
VOCAB_SIZE = len(char_to_idx) + 1

def encode_text(s, max_len=SEQ_LEN):
    s = "" if s is None else str(s)
    idxs = [char_to_idx.get(ch, 0) for ch in s[:max_len]]
    if len(idxs) < max_len:
        idxs += [IDX_PAD] * (max_len - len(idxs))
    return idxs

def build_row_seq(row):
    seq = []
    for f in FIELDS:
        if f in df.columns:
            seq += encode_text(row.get(f, ""), SEQ_LEN)
        else:
            seq += [IDX_PAD] * SEQ_LEN
    return seq

SEQ_TOTAL = SEQ_LEN * len(FIELDS)
sequences = np.array([build_row_seq(r) for _, r in df.iterrows()], dtype=np.int64)

# -----------------------
# 3) Augmentations (small edits) - used optionally for positives (we rely on MatchGroupId positives)
# -----------------------
def augment_string(s):
    s = "" if s is None else str(s)
    if len(s) > 2 and random.random() < 0.2:
        i = random.randrange(len(s)); s = s[:i] + s[i+1:]
    if len(s) > 2 and random.random() < 0.2:
        i = random.randrange(len(s)-1); lst = list(s); lst[i], lst[i+1] = lst[i+1], lst[i]; s = "".join(lst)
    if len(s) > 0 and random.random() < 0.15:
        i = random.randrange(len(s)); s = s[:i] + "#" + s[i+1:]
    if len(s) > 6 and random.random() < 0.15:
        a = random.randrange(0, len(s)//2); b = a + random.randrange(3, min(6, len(s)-a)); s = s[:a] + s[b:]
    return s

def build_aug_seq(row):
    seq = []
    for f in FIELDS:
        if f in df.columns:
            aug = augment_string(row.get(f, ""))
            seq += encode_text(aug, SEQ_LEN)
        else:
            seq += [IDX_PAD] * SEQ_LEN
    return seq

# -----------------------
# 4) Contrastive Dataset using MatchGroupId for positives (Option A)
# Produce batches where each anchor has a positive from same MatchGroupId when available.
# -----------------------
class MatchGroupContrastiveDataset(Dataset):
    def __init__(self, df, max_pairs_per_group=1000):
        self.df = df.reset_index(drop=True)
        # group indices by MatchGroupId
        self.groups = defaultdict(list)
        for idx, mg in enumerate(self.df["MatchGroupId"].astype(str)):
            self.groups[mg].append(idx)
        # create list of anchors (rows that have another same-group partner)
        self.anchors = [i for mg, idxs in self.groups.items() for i in idxs if len(idxs) >= 2]
    def __len__(self):
        return len(self.anchors)
    def __getitem__(self, idx):
        anchor_idx = self.anchors[idx]
        mg = str(self.df.loc[anchor_idx, "MatchGroupId"])
        members = self.groups[mg]
        # pick positive (different from anchor)
        pos = anchor_idx
        while pos == anchor_idx:
            pos = random.choice(members)
        # build raw sequence for anchor and positive (no augmentation needed; can augment if desired)
        x1 = np.array(build_row_seq(self.df.loc[anchor_idx]), dtype=np.int64)
        x2 = np.array(build_row_seq(self.df.loc[pos]), dtype=np.int64)
        # include amount information (we will feed amounts separately into model via DataLoader collate)
        amt1 = float(self.df.loc[anchor_idx, "SignedAmount"])
        amt2 = float(self.df.loc[pos, "SignedAmount"])
        return x1, x2, amt1, amt2

dataset = MatchGroupContrastiveDataset(df)
loader = DataLoader(dataset, batch_size=min(BATCH_SIZE, len(dataset)), shuffle=True, drop_last=False)

# -----------------------
# 5) Hybrid encoder with signed amount embedding included
#  - per-field CNN -> Transformer over fields -> combine with amount embedding -> projection head
# -----------------------
class PerFieldCNN(nn.Module):
    def __init__(self, vocab_size, char_emb= CNN_CHAR_EMBED, conv_out=CNN_OUT_CH, k=5):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, char_emb, padding_idx=IDX_PAD)
        self.conv = nn.Conv1d(char_emb, conv_out, kernel_size=k, padding=k//2)
        self.pool = nn.AdaptiveMaxPool1d(1)
    def forward(self, seq):  # seq: (batch, seq_len)
        x = self.embed(seq)            # (batch, seq_len, char_emb)
        x = x.transpose(1,2)           # (batch, char_emb, seq_len)
        x = F.relu(self.conv(x))       # (batch, conv_out, seq_len)
        x = self.pool(x).squeeze(-1)   # (batch, conv_out)
        return x

class HybridEncoderWithAmount(nn.Module):
    def __init__(self, vocab_size, seq_per_field=SEQ_LEN, num_fields=len(FIELDS),
                 char_emb=CNN_CHAR_EMBED, conv_out=CNN_OUT_CH, trans_dim=128,
                 nhead=4, num_layers=2, proj_dim=PROJ_DIM):
        super().__init__()
        self.num_fields = num_fields
        self.seq_per_field = seq_per_field
        self.field_encoder = PerFieldCNN(vocab_size, char_emb=char_emb, conv_out=conv_out)
        encoder_layer = nn.TransformerEncoderLayer(d_model=conv_out, nhead=nhead, dim_feedforward=trans_dim, activation='relu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # amount embedding: map scalar signed amount -> conv_out dimension
        self.amount_project = nn.Sequential(
            nn.Linear(1, conv_out),
            nn.ReLU()
        )
        # final projection head to embedding space
        self.proj = nn.Sequential(
            nn.Linear(conv_out, conv_out),
            nn.ReLU(),
            nn.Linear(conv_out, proj_dim)
        )
    def forward(self, seq_batch, amount_batch):
        """
        seq_batch: (batch, SEQ_TOTAL) long tensor
        amount_batch: (batch,) float tensor (signed amounts, normalized externally or here)
        """
        b = seq_batch.size(0)
        x = seq_batch.view(b, self.num_fields, self.seq_per_field)  # (batch, num_fields, seq_len)
        field_vecs = []
        for f in range(self.num_fields):
            seq_f = x[:, f, :].long()
            v = self.field_encoder(seq_f)  # (batch, conv_out)
            field_vecs.append(v.unsqueeze(1))
        field_stack = torch.cat(field_vecs, dim=1)  # (batch, num_fields, conv_out)
        tr_in = field_stack.transpose(0,1)  # (num_fields, batch, conv_out)
        tr_out = self.transformer(tr_in)    # (num_fields, batch, conv_out)
        pooled = tr_out.mean(dim=0)         # (batch, conv_out)
        # amount embedding
        amt = amount_batch.view(-1, 1)      # (batch,1)
        amt_proj = self.amount_project(amt)  # (batch, conv_out)
        # combine pooled + amount embedding (element-wise add)
        combined = pooled + amt_proj
        z = self.proj(combined)             # (batch, proj_dim)
        z = F.normalize(z, p=2, dim=1)
        return z

# instantiate model
model = HybridEncoderWithAmount(VOCAB_SIZE, seq_per_field=SEQ_LEN, num_fields=len(FIELDS)).to(DEVICE)

# -----------------------
# 6) NT-Xent / InfoNCE loss (batch of positive pairs)
# Implementation: for a batch of size B pairs -> produce 2B embeddings, positives at (i, i+B)
# -----------------------
def nt_xent_loss_from_pair_embeddings(z1, z2, temperature=TEMPERATURE):
    """
    z1, z2: tensors (B, D) normalized
    returns scalar loss (InfoNCE)
    """
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  # (2B, D)
    sim = torch.matmul(z, z.T) / temperature  # (2B,2B)
    # mask self-similarity
    mask = (~torch.eye(2*B, 2*B, dtype=torch.bool, device=DEVICE)).float()
    positives = torch.cat([torch.diag(sim, B), torch.diag(sim, -B)], dim=0)  # (2B,)
    nom = torch.exp(positives)
    denom = (mask * torch.exp(sim)).sum(dim=1)
    loss = -torch.log(nom / denom)
    return loss.mean()

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Optionally normalize amounts before training: use log + sign or simple scaling.
# We'll compute a robust scaler on SignedAmount and use that scaling in training/inference.
amounts = df["SignedAmount"].astype(float).values.reshape(-1,1)
amt_scaler = StandardScaler()
if len(amounts) > 1:
    amt_scaler.fit(amounts)
else:
    amt_scaler.mean_ = np.array([0.0]); amt_scaler.scale_ = np.array([1.0])
def scale_amounts(x):
    x = np.array(x).reshape(-1,1).astype(float)
    return (x - amt_scaler.mean_) / (amt_scaler.scale_ + 1e-9)

# -----------------------
# 7) Train contrastive encoder using MatchGroupId pairs
# -----------------------
model.train()
for epoch in range(EPOCHS):
    epoch_loss = 0.0
    n = 0
    for x1_np, x2_np, a1, a2 in loader:
        # x1_np, x2_np: numpy arrays (B, SEQ_TOTAL)
        x1 = torch.tensor(x1_np, dtype=torch.long, device=DEVICE)
        x2 = torch.tensor(x2_np, dtype=torch.long, device=DEVICE)
        a1_scaled = torch.tensor(scale_amounts(a1), dtype=torch.float32, device=DEVICE).squeeze(1)
        a2_scaled = torch.tensor(scale_amounts(a2), dtype=torch.float32, device=DEVICE).squeeze(1)
        optimizer.zero_grad()
        z1 = model(x1, a1_scaled)
        z2 = model(x2, a2_scaled)
        loss = nt_xent_loss_from_pair_embeddings(z1, z2, TEMPERATURE)
        loss.backward()
        optimizer.step()
        batch_n = x1.size(0)
        epoch_loss += loss.item() * batch_n
        n += batch_n
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"[Contrastive] Epoch {epoch+1}/{EPOCHS}, avg_loss={epoch_loss / max(1,n):.6f}")

# -----------------------
# 8) Produce embeddings for all rows (use model in eval)
# -----------------------
model.eval()
with torch.no_grad():
    seq_tensor = torch.tensor(sequences, dtype=torch.long, device=DEVICE)
    amt_scaled = torch.tensor(scale_amounts(df["SignedAmount"].values), dtype=torch.float32, device=DEVICE).squeeze(1)
    embeddings = model(seq_tensor, amt_scaled).cpu().numpy()  # (n_rows, PROJ_DIM)

# -----------------------
# 9) Build final feature matrix: embeddings + scaled amount + categorical encodings (optional)
# -----------------------
num_feats = df[["SignedAmount"]].values.astype(float)
num_feats_scaled = (num_feats - amt_scaler.mean_) / (amt_scaler.scale_ + 1e-9)

cat_feat_cols = [c + "_enc" for c in label_cols] if len(label_cols) > 0 else []
cat_feats = df[cat_feat_cols].values if len(cat_feat_cols) > 0 else np.zeros((len(df), 0))

X = np.hstack([embeddings, num_feats_scaled, cat_feats])

# -----------------------
# 10) HDBSCAN hyperparameter grid search (silhouette on non-noise)
# -----------------------
best_score = -1.0
best_labels = None
best_params = None

if len(X) < 2:
    best_labels = np.array([-1] * len(X))
    best_params = (None, None)
else:
    for mcs in HDBSCAN_MIN_CLUSTER_SIZE:
        for ms in HDBSCAN_MIN_SAMPLES:
            try:
                clusterer = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms, metric='euclidean', cluster_selection_method='eom')
                labels = clusterer.fit_predict(X)
                mask = labels != -1
                if mask.sum() < 2:
                    score = -1.0
                else:
                    score = silhouette_score(X[mask], labels[mask])
                if score > best_score:
                    best_score = score
                    best_labels = labels
                    best_params = (mcs, ms)
            except Exception:
                continue

if best_labels is None:
    # fallback
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=3, metric='euclidean', cluster_selection_method='eom')
    best_labels = clusterer.fit_predict(X)
    best_params = ("fallback", "fallback")

print("HDBSCAN best params:", best_params, "best silhouette:", best_score)
df["raw_cluster"] = best_labels

# -----------------------
# 11) FAST Greedy Matching per cluster (no heavy graph matching)
# - Sort pos/neg by absolute amount (desc)
# - For each larger-side item find best opposite-side candidate within small window
# - Accept if abs or relative tolerance satisfied
# - This is O(k * topK) per cluster; very fast in practice
# -----------------------
n = len(df)
final_cluster_map = {i: -1 for i in range(n)}
next_cid = 0

def greedy_pairing(pos_list, neg_list):
    """pos_list & neg_list are lists of (idx, amount). returns pairs and leftover lists."""
    pos_sorted = sorted(pos_list, key=lambda x: -abs(x[1]))
    neg_sorted = sorted(neg_list, key=lambda x: -abs(x[1]))
    neg_available = neg_sorted.copy()
    pairs = []
    # for speed, limit search for each pos to top-K candidates
    TOP_K = 50
    for i_idx, i_amt in pos_sorted:
        if not neg_available:
            break
        best_j = None
        best_diff = float("inf")
        limit = min(TOP_K, len(neg_available))
        for j_idx, j_amt in neg_available[:limit]:
            diff = abs(i_amt + j_amt)
            if diff < best_diff:
                best_diff = diff
                best_j = (j_idx, j_amt)
                if best_diff <= AMOUNT_ABS_TOL:
                    break
        if best_j is None:
            continue
        rel_tol = AMOUNT_REL_TOL * max(abs(i_amt), abs(best_j[1]), 1.0)
        if best_diff <= max(AMOUNT_ABS_TOL, rel_tol):
            pairs.append((i_idx, best_j[0]))
            neg_available.remove(best_j)
    paired_pos = {p for p, q in pairs}
    paired_neg = {q for p, q in pairs}
    leftovers_pos = [i for i,a in pos_sorted if i not in paired_pos]
    leftovers_neg = [j for j,a in neg_sorted if j not in paired_neg]
    return pairs, leftovers_pos, leftovers_neg

def process_cluster_members(members):
    global next_cid
    pos = [(i, df.loc[i,"SignedAmount"]) for i in members if df.loc[i,"SignedAmount"] > 0]
    neg = [(i, df.loc[i,"SignedAmount"]) for i in members if df.loc[i,"SignedAmount"] < 0]
    if not pos or not neg:
        return
    pairs, left_pos, left_neg = greedy_pairing(pos, neg)
    for i,j in pairs:
        final_cluster_map[i] = next_cid
        final_cluster_map[j] = next_cid
        next_cid += 1
    leftovers = left_pos + left_neg
    # try brute-force multi-way up to MAX_MULTIWAY
    if 2 <= len(leftovers) <= MAX_MULTIWAY:
        assigned = set()
        for r in range(min(MAX_MULTIWAY, len(leftovers)), 2, -1):
            for combo in combinations(leftovers, r):
                if any(c in assigned for c in combo):
                    continue
                s = sum(df.loc[c, "SignedAmount"] for c in combo)
                if abs(s) <= AMOUNT_ABS_TOL:
                    for c in combo:
                        final_cluster_map[c] = next_cid
                        assigned.add(c)
                    next_cid += 1

# run per raw cluster (excluding noise)
unique_labels = sorted(set(best_labels))
raw_clusters = {lbl: [i for i, lab in enumerate(best_labels) if lab == lbl] for lbl in unique_labels}
for lbl, members in raw_clusters.items():
    if lbl == -1:
        continue
    process_cluster_members(members)

# global greedy pass for remaining unassigned (including noise)
unassigned = [i for i in range(n) if final_cluster_map[i] == -1]
if unassigned:
    pos_un = [(i, df.loc[i,"SignedAmount"]) for i in unassigned if df.loc[i,"SignedAmount"] > 0]
    neg_un = [(i, df.loc[i,"SignedAmount"]) for i in unassigned if df.loc[i,"SignedAmount"] < 0]
    pairs, left_pos, left_neg = greedy_pairing(pos_un, neg_un)
    for i,j in pairs:
        final_cluster_map[i] = next_cid
        final_cluster_map[j] = next_cid
        next_cid += 1
    remaining = [i for i in unassigned if final_cluster_map[i] == -1]
    if 2 <= len(remaining) <= MAX_MULTIWAY:
        assigned = set()
        for r in range(min(MAX_MULTIWAY, len(remaining)), 2, -1):
            for combo in combinations(remaining, r):
                if any(c in assigned for c in combo):
                    continue
                s = sum(df.loc[c, "SignedAmount"] for c in combo)
                if abs(s) <= AMOUNT_ABS_TOL:
                    for c in combo:
                        final_cluster_map[c] = next_cid
                        assigned.add(c)
                    next_cid += 1

# write results
df["FinalCluster"] = [final_cluster_map.get(i, -1) for i in range(n)]

# -----------------------
# 12) Optional evaluation vs MatchGroupId (pairwise metrics)
# -----------------------
if "MatchGroupId" in df.columns:
    true_pairs = set()
    pred_pairs = set()
    for i, j in combinations(range(n), 2):
        if str(df.loc[i,"MatchGroupId"]) == str(df.loc[j,"MatchGroupId"]):
            true_pairs.add((i,j))
        if df.loc[i,"FinalCluster"] != -1 and df.loc[i,"FinalCluster"] == df.loc[j,"FinalCluster"]:
            pred_pairs.add((i,j))
    tp = len(true_pairs & pred_pairs)
    fp = len(pred_pairs - true_pairs)
    fn = len(true_pairs - pred_pairs)
    prec = tp / (tp + fp) if (tp+fp)>0 else 0.0
    rec = tp / (tp + fn) if (tp+fn)>0 else 0.0
    f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
    print(f"Evaluation -> precision: {prec:.4f}, recall: {rec:.4f}, f1: {f1:.4f}")

# -----------------------
# 13) Output / save
# -----------------------
print("HDBSCAN best params:", best_params)
print("Sample output:")
show_cols = ["SignedAmount", "raw_cluster", "FinalCluster"] + ([c + "_enc" for c in label_cols] if label_cols else [])
print(df[show_cols].head(50))
df.to_csv("clusters_hybrid_contrastive_hdbscan_greedy.csv", index=False)
print("Done.")
