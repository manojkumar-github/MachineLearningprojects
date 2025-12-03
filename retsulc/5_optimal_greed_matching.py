"""
Optimized end-to-end pipeline for transaction reconciliation

Features:
- Hybrid character-level encoder (Per-field CNN -> Transformer) producing embeddings
- Self-supervised contrastive pretraining (NT-Xent / SimCLR-style) with ID-specific augmentations
- HDBSCAN clustering with automatic hyperparameter grid search (silhouette on non-noise)
- FAST greedy matching within clusters to enforce sum(SignedAmount) ~ 0
  (no heavy O(n^3) matching). Optional small multi-way brute-force repair for tiny leftovers.
- Optional pairwise evaluation vs MatchGroupId if available

Notes:
- Replace the example data loading with `pd.read_csv(...)` for real dataset.
- Tune hyperparameters (EPOCHS, BATCH_SIZE, HDBSCAN grid, tolerances) for best results.
- Requires: torch, sklearn, scipy, hdbscan, numpy, pandas
"""

import os
import sys
import random
import math
import numpy as np
import pandas as pd
from itertools import combinations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

# attempt to import hdbscan; install if missing
try:
    import hdbscan
except Exception:
    os.system(f"{sys.executable} -m pip install hdbscan --quiet")
    import hdbscan

from scipy.optimize import linear_sum_assignment  # not used for greedy but kept if needed

# -----------------------
# CONFIG (tune these)
# -----------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Contrastive training hyperparams
BATCH_SIZE = 128
EPOCHS = 20
LR = 1e-3
TEMPERATURE = 0.5
PROJ_DIM = 128      # final embedding dim used for clustering
CNN_CHAR_EMBED = 16
CNN_OUT_CH = 64

# Sequence/fields
SEQ_LEN = 64   # characters per field
FIELDS = [
    "MerchantRefNum",
    "WebOrderNumber",
    "AcquireRefNumber",
    "PONumber",
    "TransactionRefNo",
    "CardNo",
    "AccountingDocNum"
]

# HDBSCAN hyperparameter grid (auto-search)
HDBSCAN_MIN_CLUSTER_SIZE = [3, 5, 10]
HDBSCAN_MIN_SAMPLES = [1, 3, 5]

# Greedy matching settings
AMOUNT_ABS_TOL = 0.01   # absolute tolerance in currency units (e.g. 1 cent = 0.01)
AMOUNT_REL_TOL = 0.01   # relative tolerance (1% of max amount)
MAX_MULTIWAY = 4        # attempt multi-way brute force up to this size for leftovers

# -----------------------
# 0) Load your data
# -----------------------
# Replace this with real read:
# df = pd.read_csv("transactions.csv")
# Example single-row (provided earlier) used as fallback/demo:
data = {
    "DocumentDate": ["02/01/2025"],
    "DocType": ["unknown"],
    "TransactionType": ["SAP"],
    "BankTrfRef": ["unknown"],
    "Amount": [7800],
    "TransactionRefNo": ["W1dsfsafjdjfb"],
    "MerchantRefNum": ["777741344598"],
    "CR_DR": ["CR"],
    "GLRecordID": ["unknown"],
    "OID": ["unknown"],
    "PONumber": ["13u350u5u05"],
    "CardNo": ["13415555531535"],
    "ReceiptNumber": ["unknown"],
    "AccountingDocNum": ["KD0nfdkk"],
    "AuthCode": ["unknown"],
    "RefDocument": ["unknown"],
    "Assignment": ["unknown"],
    "StoreNumber": ["unknown"],
    "AcquireRefNumber": ["unknown"],
    "WebOrderNumber": ["W1342421414"],
    "MatchGroupId": ["14443553"],
    "Source": ["SAP"],
    "SourceType": ["Internal"]
}
df = pd.DataFrame(data)

# -----------------------
# 1) Preprocessing
# -----------------------
# Drop columns that are all unknown (user requested)
drop_cols = ["ReceiptNumber", "RefDocument", "Assignment", "StoreNumber", "AuthCode"]
for c in drop_cols:
    if c in df.columns:
        df = df.drop(columns=c)

df["DocumentDate"] = pd.to_datetime(df["DocumentDate"], errors="coerce", dayfirst=True)

# Signed amount: CR => +, DR => -
df["SignedAmount"] = df["Amount"].astype(float) * df["CR_DR"].map({"CR": 1.0, "DR": -1.0}).fillna(1.0)

# label-encode small categorical fields (optional features)
label_cols = [c for c in ["DocType", "TransactionType", "Source", "SourceType"] if c in df.columns]
for col in label_cols:
    le = LabelEncoder()
    df[col + "_enc"] = le.fit_transform(df[col].astype(str))

# -----------------------
# 2) Build character vocabulary from fields
# -----------------------
all_text = ""
for f in FIELDS:
    if f in df.columns:
        all_text += " ".join(df[f].astype(str).tolist()) + " "
chars = sorted(set([c for c in all_text]))
if len(chars) == 0:
    # fallback
    chars = list("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_-/.")
char_to_idx = {c: i+1 for i, c in enumerate(chars)}
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
# 3) augmentation for contrastive training (small ID edits)
# -----------------------
def augment_string(s):
    s = "" if s is None else str(s)
    if len(s) > 2 and random.random() < 0.2:
        i = random.randrange(len(s))
        s = s[:i] + s[i+1:]
    if len(s) > 2 and random.random() < 0.2:
        i = random.randrange(len(s)-1)
        lst = list(s); lst[i], lst[i+1] = lst[i+1], lst[i]; s = "".join(lst)
    if len(s) > 0 and random.random() < 0.15:
        i = random.randrange(len(s))
        s = s[:i] + "#" + s[i+1:]
    if len(s) > 6 and random.random() < 0.15:
        a = random.randrange(0, len(s)//2)
        b = a + random.randrange(3, min(6, len(s)-a))
        s = s[:a] + s[b:]
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
# 4) Dataset & DataLoader for contrastive training
# -----------------------
class ContrastiveDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x1 = np.array(build_row_seq(row), dtype=np.int64)
        x2 = np.array(build_aug_seq(row), dtype=np.int64)
        return x1, x2

dataset = ContrastiveDataset(df)
loader = DataLoader(dataset, batch_size=min(BATCH_SIZE, len(dataset)), shuffle=True)

# -----------------------
# 5) Hybrid encoder: per-field CNN -> Transformer over fields -> projection head
# -----------------------
class PerFieldCNN(nn.Module):
    def __init__(self, vocab_size, char_emb= CNN_CHAR_EMBED, conv_out=CNN_OUT_CH, k=5):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, char_emb, padding_idx=IDX_PAD)
        self.conv = nn.Conv1d(char_emb, conv_out, kernel_size=k, padding=k//2)
        self.pool = nn.AdaptiveMaxPool1d(1)
    def forward(self, seq):  # seq: (batch, seq_len)
        x = self.embed(seq)           # (batch, seq_len, char_emb)
        x = x.transpose(1,2)          # (batch, char_emb, seq_len)
        x = F.relu(self.conv(x))      # (batch, conv_out, seq_len)
        x = self.pool(x).squeeze(-1)  # (batch, conv_out)
        return x

class HybridEncoder(nn.Module):
    def __init__(self, vocab_size, seq_per_field=SEQ_LEN, num_fields=len(FIELDS),
                 char_emb= CNN_CHAR_EMBED, conv_out=CNN_OUT_CH, trans_dim=128, nhead=4, num_layers=2, proj_dim=PROJ_DIM):
        super().__init__()
        self.num_fields = num_fields
        self.seq_per_field = seq_per_field
        self.field_encoder = PerFieldCNN(vocab_size, char_emb=char_emb, conv_out=conv_out)
        encoder_layer = nn.TransformerEncoderLayer(d_model=conv_out, nhead=nhead, dim_feedforward=trans_dim, activation='relu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # projection head
        self.proj = nn.Sequential(
            nn.Linear(conv_out, conv_out),
            nn.ReLU(),
            nn.Linear(conv_out, proj_dim)
        )
    def forward(self, x):
        b = x.size(0)
        x = x.view(b, self.num_fields, self.seq_per_field)  # (batch, num_fields, seq_len)
        field_vecs = []
        for f in range(self.num_fields):
            seq_f = x[:, f, :].long()
            v = self.field_encoder(seq_f)
            field_vecs.append(v.unsqueeze(1))
        field_stack = torch.cat(field_vecs, dim=1)  # (batch, num_fields, conv_out)
        tr_in = field_stack.transpose(0,1)  # (num_fields, batch, conv_out)
        tr_out = self.transformer(tr_in)    # (num_fields, batch, conv_out)
        pooled = tr_out.mean(dim=0)         # (batch, conv_out)
        z = self.proj(pooled)               # (batch, proj_dim)
        z = F.normalize(z, p=2, dim=1)
        return z

# instantiate
model = HybridEncoder(VOCAB_SIZE, seq_per_field=SEQ_LEN, num_fields=len(FIELDS)).to(DEVICE)

# -----------------------
# 6) NT-Xent / SimCLR-style loss
# -----------------------
def nt_xent_loss(z1, z2, temperature=TEMPERATURE):
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.T) / temperature
    # mask self
    mask = (~torch.eye(2*B, 2*B, dtype=torch.bool, device=DEVICE)).float()
    positives = torch.cat([torch.diag(sim, B), torch.diag(sim, -B)], dim=0)
    nom = torch.exp(positives)
    denom = (mask * torch.exp(sim)).sum(dim=1)
    loss = -torch.log(nom / denom)
    return loss.mean()

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# -----------------------
# 7) Contrastive training loop
# -----------------------
model.train()
for epoch in range(EPOCHS):
    epoch_loss = 0.0
    n = 0
    for x1_np, x2_np in loader:
        x1 = torch.tensor(x1_np, dtype=torch.long, device=DEVICE)
        x2 = torch.tensor(x2_np, dtype=torch.long, device=DEVICE)
        optimizer.zero_grad()
        z1 = model(x1)
        z2 = model(x2)
        loss = nt_xent_loss(z1, z2)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * x1.size(0)
        n += x1.size(0)
    if (epoch+1) % 5 == 0 or epoch == 0:
        print(f"[Contrastive] Epoch {epoch+1}/{EPOCHS}, avg_loss={epoch_loss / max(1,n):.4f}")

# -----------------------
# 8) Produce embeddings for all rows (deterministic)
# -----------------------
model.eval()
with torch.no_grad():
    seq_tensor = torch.tensor(sequences, dtype=torch.long, device=DEVICE)
    embeddings = model(seq_tensor).cpu().numpy()  # (n_rows, PROJ_DIM)

# -----------------------
# 9) Build final feature matrix: embeddings + signed amount (scaled) + categorical encodings
# -----------------------
num_feats = df[["SignedAmount"]].values.astype(float)
scaler = StandardScaler()
num_feats_scaled = scaler.fit_transform(num_feats)

cat_cols_enc = [c + "_enc" for c in label_cols] if len(label_cols) > 0 else []
cat_feats = df[cat_cols_enc].values if len(cat_cols_enc) > 0 else np.zeros((len(df), 0))

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
                # compute silhouette on non-noise points only
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
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=3, metric='euclidean', cluster_selection_method='eom')
    best_labels = clusterer.fit_predict(X)
    best_params = ("fallback", "fallback")

print("HDBSCAN best params:", best_params, "best silhouette:", best_score)
df["raw_cluster"] = best_labels

# -----------------------
# 11) FAST Greedy Matching per cluster (O(k log k) per cluster)
# Strategy:
#  - For each non-noise cluster: separate positives and negatives by sign
#  - Sort both lists by absolute amount descending
#  - For each item in larger side, find closest opposite-side amount (binary search on sorted list)
#  - Pair if amounts cancel within absolute OR relative tolerance
#  - Remove paired items and continue
# -----------------------
n = len(df)
final_cluster_map = {i: -1 for i in range(n)}
next_cid = 0

def greedy_pairing(pos_list, neg_list, df_local):
    """
    pos_list: list of (idx, amount) where amount > 0
    neg_list: list of (idx, amount) where amount < 0
    Returns paired list of (i,j) and leftovers lists
    """
    # sort descending by abs(amount)
    pos_sorted = sorted(pos_list, key=lambda x: -abs(x[1]))
    neg_sorted = sorted(neg_list, key=lambda x: -abs(x[1]))
    # use lists for removals
    neg_available = neg_sorted.copy()
    pairs = []
    used_neg = set()
    for i_idx, i_amt in pos_sorted:
        if not neg_available:
            break
        # binary search for best neg in neg_available: find min abs(i_amt + neg_amt)
        # linear scan over small top-K portion (fast in practice)
        best_j = None
        best_diff = float("inf")
        best_pos = None
        # limit search to top 50 candidates for speed
        limit = min(50, len(neg_available))
        for j_idx, j_amt in neg_available[:limit]:
            diff = abs(i_amt + j_amt)
            if diff < best_diff:
                best_diff = diff
                best_j = (j_idx, j_amt)
                best_pos = (i_idx, i_amt)
                if best_diff <= AMOUNT_ABS_TOL:
                    break
        if best_j is None:
            continue
        # acceptance test: absolute or relative tolerance
        rel_tol = AMOUNT_REL_TOL * max(abs(i_amt), abs(best_j[1]), 1.0)
        if best_diff <= max(AMOUNT_ABS_TOL, rel_tol):
            pairs.append((best_pos[0], best_j[0]))
            # remove matched neg
            neg_available.remove(best_j)
    # leftovers:
    paired_pos = {p for p, _ in pairs}
    paired_neg = {q for _, q in pairs}
    leftovers_pos = [i for i, a in pos_sorted if i not in paired_pos]
    leftovers_neg = [j for j, a in neg_sorted if j not in paired_neg]
    return pairs, leftovers_pos, leftovers_neg

def process_cluster_members(member_indices):
    global next_cid
    pos = [(i, df.loc[i, "SignedAmount"]) for i in member_indices if df.loc[i, "SignedAmount"] > 0]
    neg = [(i, df.loc[i, "SignedAmount"]) for i in member_indices if df.loc[i, "SignedAmount"] < 0]
    if not pos or not neg:
        return  # nothing to pair here
    pairs, leftover_pos, leftover_neg = greedy_pairing(pos, neg, df)
    # assign cluster ids for pairs
    for i, j in pairs:
        final_cluster_map[i] = next_cid
        final_cluster_map[j] = next_cid
        next_cid += 1
    # attempt multi-way balancing for leftovers if small
    leftovers = leftover_pos + leftover_neg
    if len(leftovers) > 1 and len(leftovers) <= MAX_MULTIWAY + 1:
        # brute-force subsets up to MAX_MULTIWAY
        assigned = set()
        for size in range(min(MAX_MULTIWAY, len(leftovers)), 2, -1):
            for combo in combinations(leftovers, size):
                if any(c in assigned for c in combo):
                    continue
                s = sum(df.loc[c, "SignedAmount"] for c in combo)
                if abs(s) <= AMOUNT_ABS_TOL:
                    for c in combo:
                        final_cluster_map[c] = next_cid
                        assigned.add(c)
                    next_cid += 1
        # remaining leftovers left unassigned (-1)
    return

# Run greedy pairing per raw cluster (excluding noise)
unique_raw = sorted(set(best_labels))
for lbl in unique_raw:
    if lbl == -1:
        continue
    members = [i for i, lab in enumerate(best_labels) if lab == lbl]
    process_cluster_members(members)

# Global greedy pass for remaining unassigned rows (include noise)
unassigned = [i for i in range(n) if final_cluster_map[i] == -1]
if unassigned:
    # build pos/neg among unassigned
    pos_un = [(i, df.loc[i,"SignedAmount"]) for i in unassigned if df.loc[i,"SignedAmount"] > 0]
    neg_un = [(i, df.loc[i,"SignedAmount"]) for i in unassigned if df.loc[i,"SignedAmount"] < 0]
    pairs, leftover_pos, leftover_neg = greedy_pairing(pos_un, neg_un, df)
    for i, j in pairs:
        final_cluster_map[i] = next_cid
        final_cluster_map[j] = next_cid
        next_cid += 1
    # attempt small multi-way on remaining unassigned
    remaining = [i for i in unassigned if final_cluster_map[i] == -1]
    if len(remaining) <= MAX_MULTIWAY + 1 and len(remaining) > 1:
        assigned = set()
        for size in range(min(MAX_MULTIWAY, len(remaining)), 2, -1):
            for combo in combinations(remaining, size):
                if any(c in assigned for c in combo):
                    continue
                s = sum(df.loc[c, "SignedAmount"] for c in combo)
                if abs(s) <= AMOUNT_ABS_TOL:
                    for c in combo:
                        final_cluster_map[c] = next_cid
                        assigned.add(c)
                    next_cid += 1

# Final: write final cluster assignments
df["FinalCluster"] = [final_cluster_map.get(i, -1) for i in range(n)]

# -----------------------
# 12) Optional evaluation vs MatchGroupId
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
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0
    print(f"Evaluation -> precision: {prec:.4f}, recall: {rec:.4f}, f1: {f1:.4f}")

# -----------------------
# 13) Save and show results
# -----------------------
print("HDBSCAN best params (from search):", best_params)
print("Sample output columns: SignedAmount, raw_cluster, FinalCluster")
cols_show = ["SignedAmount", "raw_cluster", "FinalCluster"] + cat_cols_enc
print(df[cols_show].head(20))
df.to_csv("optimized_clusters_greedy.csv", index=False)
print("Done.")
