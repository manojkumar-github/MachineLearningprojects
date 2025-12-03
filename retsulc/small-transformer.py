"""
End-to-end: Hybrid char-level encoder (CNN+Transformer) + Contrastive pretraining + HDBSCAN hypersearch + Graph matching
"""

import os
import sys
import random
import math
import numpy as np
import pandas as pd
from itertools import combinations
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
from scipy.optimize import linear_sum_assignment

# Try to import hdbscan (install if necessary)
try:
    import hdbscan
except Exception:
    print("hdbscan not found, attempting to install...")
    os.system(f"{sys.executable} -m pip install hdbscan --quiet")
    import hdbscan

# -----------------------
# CONFIG / HYPERPARAMS (tune these)
# -----------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# Contrastive training hyperparams
BATCH_SIZE = 128
EPOCHS = 30
LR = 1e-3
TEMPERATURE = 0.5
PROJ_DIM = 128     # projection head dim (embedding used for clustering)
CNN_EMBED_DIM = 32
CNN_OUT_CH = 64

# Sequence/fields
SEQ_LEN = 64       # char length per field
FIELDS = [
    "MerchantRefNum",
    "WebOrderNumber",
    "AcquireRefNumber",
    "PONumber",
    "TransactionRefNo",
    "CardNo",
    "AccountingDocNum"
]

# HDBSCAN hyperparameter search grid
HDBSCAN_MIN_CLUSTER_SIZE = [3, 5, 10]   # try these values
HDBSCAN_MIN_SAMPLES = [1, 3, 5]

# Matching & balancing
MAX_SUBSET_SIZE = 4
AMOUNT_TOL = 1e-6

# -----------------------
# 0) Load data (replace with pd.read_csv if you have CSV)
# -----------------------
# Example single-row dataset provided earlier
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
# 1) Basic cleaning & drop known-all-unknown columns
# -----------------------
drop_cols = ["ReceiptNumber", "RefDocument", "Assignment", "StoreNumber", "AuthCode"]
for c in drop_cols:
    if c in df.columns:
        df = df.drop(columns=c)

# Parse date (we're not using date-window constraint anymore)
df["DocumentDate"] = pd.to_datetime(df["DocumentDate"], errors="coerce", dayfirst=True)

# SignedAmount: CR -> +, DR -> -
df["SignedAmount"] = df["Amount"].astype(float) * df["CR_DR"].map({"CR": 1.0, "DR": -1.0}).fillna(1.0)

# Label-encode small categorical fields to include later
label_cols = [c for c in ["DocType", "TransactionType", "Source", "SourceType"] if c in df.columns]
for col in label_cols:
    le = LabelEncoder()
    df[col + "_enc"] = le.fit_transform(df[col].astype(str))

# -----------------------
# 2) Build character vocabulary from FIELDS
# -----------------------
all_text = ""
for f in FIELDS:
    if f in df.columns:
        all_text += " ".join(df[f].astype(str).tolist()) + " "
chars = sorted(set([c for c in all_text]))
if len(chars) == 0:
    chars = list("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_-/.")
char_to_idx = {c: i+1 for i, c in enumerate(chars)}  # 0 used for padding
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
# 3) Augmentation for ID strings (ID-specific augmentations)
# -----------------------
def augment_string(s):
    s = "" if s is None else str(s)
    # Only small edits to generate positives
    # random deletion
    if len(s) > 2 and random.random() < 0.2:
        i = random.randrange(len(s))
        s = s[:i] + s[i+1:]
    # adjacent swap
    if len(s) > 2 and random.random() < 0.2:
        i = random.randrange(len(s)-1)
        lst = list(s); lst[i], lst[i+1] = lst[i+1], lst[i]; s = "".join(lst)
    # mask char
    if len(s) > 0 and random.random() < 0.15:
        i = random.randrange(len(s))
        s = s[:i] + "#" + s[i+1:]
    # substring crop
    if len(s) > 6 and random.random() < 0.15:
        a = random.randrange(0, len(s)//2)
        b = a + random.randrange(3, min(6, len(s)-a))
        s = s[:a] + s[b:]
    return s

def build_augmented_seq(row):
    seq = []
    for f in FIELDS:
        if f in df.columns:
            aug = augment_string(row.get(f, ""))
            seq += encode_text(aug, SEQ_LEN)
        else:
            seq += [IDX_PAD] * SEQ_LEN
    return seq

# -----------------------
# 4) PyTorch Dataset for contrastive training
# -----------------------
class ContrastiveDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x1 = np.array(build_row_seq(row), dtype=np.int64)
        x2 = np.array(build_augmented_seq(row), dtype=np.int64)
        return x1, x2

dataset = ContrastiveDataset(df)
loader = DataLoader(dataset, batch_size=min(BATCH_SIZE, len(dataset)), shuffle=True)

# -----------------------
# 5) Hybrid Encoder model: per-field CNN -> field vectors -> small Transformer -> projection head
# -----------------------
class PerFieldCNN(nn.Module):
    def __init__(self, vocab_size, char_embed_dim=16, conv_out= CNN_OUT_CH, kernel_size=5):
        super().__init__()
        self.char_embed = nn.Embedding(vocab_size, char_embed_dim, padding_idx=IDX_PAD)
        self.conv = nn.Conv1d(char_embed_dim, conv_out, kernel_size=kernel_size, padding=kernel_size//2)
        # pool to get fixed-size vector
        self.pool = nn.AdaptiveMaxPool1d(1)
    def forward(self, seq):  # seq: (batch, seq_len)
        x = self.char_embed(seq)            # (batch, seq_len, char_embed_dim)
        x = x.transpose(1,2)                # (batch, char_embed_dim, seq_len)
        x = F.relu(self.conv(x))            # (batch, conv_out, seq_len)
        x = self.pool(x).squeeze(-1)        # (batch, conv_out)
        return x

class HybridEncoder(nn.Module):
    def __init__(self, vocab_size, seq_per_field=SEQ_LEN, num_fields=len(FIELDS),
                 char_embed_dim=16, conv_out=CNN_OUT_CH, trans_dim=128, nhead=4, num_layers=2, proj_dim=PROJ_DIM):
        super().__init__()
        self.num_fields = num_fields
        self.seq_per_field = seq_per_field
        self.field_encoder = PerFieldCNN(vocab_size, char_embed_dim=char_embed_dim, conv_out=conv_out)
        encoder_layer = nn.TransformerEncoderLayer(d_model=conv_out, nhead=nhead, dim_feedforward=trans_dim, activation='relu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # projection head
        self.proj = nn.Sequential(
            nn.Linear(conv_out, conv_out),
            nn.ReLU(),
            nn.Linear(conv_out, proj_dim)
        )
    def forward(self, x):  # x: (batch, seq_total) where seq_total = seq_per_field * num_fields
        b = x.size(0)
        # split into fields
        x = x.view(b, self.num_fields, self.seq_per_field)  # (batch, num_fields, seq_len)
        field_vecs = []
        for f in range(self.num_fields):
            seq_f = x[:, f, :]  # (batch, seq_len)
            v = self.field_encoder(seq_f)  # (batch, conv_out)
            field_vecs.append(v.unsqueeze(1))
        # concat field vectors -> (batch, num_fields, conv_out)
        field_stack = torch.cat(field_vecs, dim=1)
        # transformer expects (seq_len=num_fields, batch, dim)
        tr_in = field_stack.transpose(0,1)  # (num_fields, batch, conv_out)
        tr_out = self.transformer(tr_in)    # (num_fields, batch, conv_out)
        # pool across fields (mean)
        pooled = tr_out.mean(dim=0)         # (batch, conv_out)
        z = self.proj(pooled)               # (batch, proj_dim)
        z = F.normalize(z, p=2, dim=1)
        return z

# instantiate model
model = HybridEncoder(VOCAB_SIZE, seq_per_field=SEQ_LEN, num_fields=len(FIELDS),
                      char_embed_dim=16, conv_out=CNN_OUT_CH, trans_dim=128,
                      nhead=4, num_layers=2, proj_dim=PROJ_DIM).to(DEVICE)

# -----------------------
# 6) NT-Xent (contrastive) loss implementation
# -----------------------
def nt_xent_loss(z1, z2, temperature=TEMPERATURE):
    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  # 2B x D
    sim = torch.matmul(z, z.T)      # 2B x 2B
    sim = sim / temperature
    # mask out self-similarity
    mask = (~torch.eye(2*batch_size, 2*batch_size, dtype=torch.bool, device=DEVICE)).float()
    positives = torch.cat([torch.diag(sim, batch_size), torch.diag(sim, -batch_size)], dim=0)
    nom = torch.exp(positives)
    denom = mask * torch.exp(sim)
    denom = denom.sum(dim=1)
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
        loss = nt_xent_loss(z1, z2, TEMPERATURE)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * x1.size(0)
        n += x1.size(0)
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"[Contrastive] Epoch {epoch+1}/{EPOCHS}, avg_loss={epoch_loss / max(1,n):.4f}")

# -----------------------
# 8) Get embeddings for all rows
# -----------------------
model.eval()
with torch.no_grad():
    seq_tensor = torch.tensor(sequences, dtype=torch.long, device=DEVICE)
    embeddings = model(seq_tensor).cpu().numpy()  # (n_rows, PROJ_DIM)

# -----------------------
# 9) Build final feature matrix (embeddings + numeric + categorical)
# -----------------------
num_feats = df[["SignedAmount"]].values.astype(float)
scaler = StandardScaler()
num_feats_scaled = scaler.fit_transform(num_feats)

cat_feat_cols = [col + "_enc" for col in label_cols] if len(label_cols) > 0 else []
cat_feats = df[cat_feat_cols].values if len(cat_feat_cols) > 0 else np.zeros((len(df), 0))

X = np.hstack([embeddings, num_feats_scaled, cat_feats])

# -----------------------
# 10) HDBSCAN hyperparameter grid search (silhouette scoring)
# If silhouette cannot be computed (too few clusters), fallback to default params
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
                unique = set(labels)
                n_clusters = len([l for l in unique if l != -1])
                if n_clusters < 2:
                    score = -1.0
                else:
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
            except Exception as e:
                # skip failing hyperparams quietly
                # print("HDBSCAN params failed:", mcs, ms, e)
                continue

if best_labels is None:
    # fallback default
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=3, metric='euclidean', cluster_selection_method='eom')
    best_labels = clusterer.fit_predict(X)
    best_params = ("fallback", "fallback")

print("HDBSCAN best params:", best_params, "best silhouette:", best_score)
df["raw_cluster"] = best_labels

# -----------------------
# 11) Graph + max-weight matching per cluster (use embedding similarity + amount compatibility)
# Steps:
#  - For each non-noise raw cluster: split pos/neg, compute similarity matrix (cosine)
#  - Solve linear_sum_assignment on negative similarity (maximize sim)
#  - Keep matches only if amount cancel within tolerance (or relative tolerance)
#  - Assign matched pairs to final clusters
#  - For leftovers attempt multi-way balancing up to MAX_SUBSET_SIZE
#  - After processing clusters, attempt global matching on unassigned rows
# -----------------------
n = len(df)
final_cluster_map = {i: -1 for i in range(n)}
next_cid = 0

def pairwise_match_in_cluster(member_indices):
    global next_cid
    pos = [i for i in member_indices if df.loc[i, "SignedAmount"] > 0]
    neg = [i for i in member_indices if df.loc[i, "SignedAmount"] < 0]
    if not pos or not neg:
        return member_indices  # nothing matched
    emb_pos = embeddings[pos]
    emb_neg = embeddings[neg]
    sim = cosine_similarity(emb_pos, emb_neg)  # pos x neg
    n1, n2 = sim.shape
    N = max(n1, n2)
    BIG = 1e6
    cost = np.ones((N, N)) * BIG
    cost[:n1, :n2] = -sim  # maximize similarity via minimizing negative sim
    row_ind, col_ind = linear_sum_assignment(cost)
    matches = []
    for r, c in zip(row_ind, col_ind):
        if r < n1 and c < n2:
            i = pos[r]; j = neg[c]
            tol = max(AMOUNT_TOL, 0.005 * max(abs(df.loc[i,"SignedAmount"]), abs(df.loc[j,"SignedAmount"]), 1.0))  # 0.5% tolerance heuristic
            if abs(df.loc[i, "SignedAmount"] + df.loc[j, "SignedAmount"]) <= tol:
                matches.append((i, j, sim[r, c]))
    # pick non-overlapping matches by descending sim
    matches = sorted(matches, key=lambda x: -x[2])
    used = set()
    chosen = []
    for i, j, s in matches:
        if i in used or j in used:
            continue
        chosen.append((i, j))
        used.add(i); used.add(j)
    for i, j in chosen:
        final_cluster_map[i] = next_cid
        final_cluster_map[j] = next_cid
        next_cid += 1
    leftovers = [m for m in member_indices if final_cluster_map[m] == -1]
    return leftovers

def find_multiway_balanced(member_indices):
    global next_cid
    assigned = set()
    amounts = {i: df.loc[i, "SignedAmount"] for i in member_indices}
    # prefer larger subsets first
    for r in range(min(MAX_SUBSET_SIZE, len(member_indices)), 2, -1):
        for combo in combinations(member_indices, r):
            if any(c in assigned for c in combo):
                continue
            s = sum(amounts[c] for c in combo)
            if abs(s) <= AMOUNT_TOL:
                for c in combo:
                    final_cluster_map[c] = next_cid
                    assigned.add(c)
                next_cid += 1
    leftovers = [m for m in member_indices if final_cluster_map[m] == -1]
    return leftovers

# Process each raw cluster (excluding noise)
unique_labels = sorted(set(best_labels))
raw_clusters = {lbl: [i for i, lab in enumerate(best_labels) if lab == lbl] for lbl in unique_labels}
for lbl, members in raw_clusters.items():
    if lbl == -1:
        continue
    leftovers = pairwise_match_in_cluster(members)
    if len(leftovers) > 0:
        _ = find_multiway_balanced(leftovers)

# Global matching across unassigned rows (including noise)
unassigned = [i for i in range(n) if final_cluster_map[i] == -1]
pos_un = [i for i in unassigned if df.loc[i, "SignedAmount"] > 0]
neg_un = [i for i in unassigned if df.loc[i, "SignedAmount"] < 0]
if pos_un and neg_un:
    emb_pos = embeddings[pos_un]; emb_neg = embeddings[neg_un]
    sim = cosine_similarity(emb_pos, emb_neg)
    n1, n2 = sim.shape
    N = max(n1, n2)
    BIG = 1e6
    cost = np.ones((N, N)) * BIG
    cost[:n1, :n2] = -sim
    row_ind, col_ind = linear_sum_assignment(cost)
    matches = []
    for r, c in zip(row_ind, col_ind):
        if r < n1 and c < n2:
            i = pos_un[r]; j = neg_un[c]
            tol = max(AMOUNT_TOL, 0.005 * max(abs(df.loc[i,"SignedAmount"]), abs(df.loc[j,"SignedAmount"]), 1.0))
            if abs(df.loc[i,"SignedAmount"] + df.loc[j,"SignedAmount"]) <= tol:
                matches.append((i, j, sim[r, c]))
    matches = sorted(matches, key=lambda x: -x[2])
    used = set()
    for i, j, s in matches:
        if i in used or j in used:
            continue
        final_cluster_map[i] = next_cid
        final_cluster_map[j] = next_cid
        next_cid += 1
        used.add(i); used.add(j)

# Final multi-way pass for any remaining
remaining = [i for i in range(n) if final_cluster_map[i] == -1]
if remaining:
    _ = find_multiway_balanced(remaining)

# write results
df["FinalCluster"] = [final_cluster_map.get(i, -1) for i in range(n)]

# -----------------------
# 12) Optional evaluation: pairwise precision/recall/f1 if MatchGroupId present
# -----------------------
if "MatchGroupId" in df.columns:
    true_pairs = set()
    pred_pairs = set()
    for i, j in combinations(range(n), 2):
        if str(df.loc[i, "MatchGroupId"]) == str(df.loc[j, "MatchGroupId"]):
            true_pairs.add((i, j))
        if df.loc[i, "FinalCluster"] != -1 and df.loc[i, "FinalCluster"] == df.loc[j, "FinalCluster"]:
            pred_pairs.add((i, j))
    tp = len(true_pairs & pred_pairs)
    fp = len(pred_pairs - true_pairs)
    fn = len(true_pairs - pred_pairs)
    prec = tp / (tp + fp) if (tp+fp)>0 else 0.0
    rec = tp / (tp + fn) if (tp+fn)>0 else 0.0
    f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
    print(f"Evaluation -> precision: {prec:.4f}, recall: {rec:.4f}, f1: {f1:.4f}")

# -----------------------
# 13) Save and sample output
# -----------------------
print("HDBSCAN best params:", best_params)
print("Sample outputs:")
show_cols = ["SignedAmount", "raw_cluster", "FinalCluster"] + cat_feat_cols
print(df[show_cols].head(50))
df.to_csv("clusters_hybrid_contrastive_hdbscan.csv", index=False)

print("Done.")
