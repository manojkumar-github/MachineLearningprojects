"""
Optimized end-to-end pipeline (Option 1: model learns amount-balancing implicitly)
- Field-specific small CNN encoders (one encoder per field)
- Transformer over field vectors
- SignedAmount encoded INSIDE the neural encoder (learnable projection)
- Contrastive pretraining (InfoNCE / NT-Xent) using MatchGroupId to form positive pairs
- DBSCAN clustering with simple hyperparameter search (eps, min_samples)
- NO greedy matching / NO post-processing — rely on the model + clustering to produce balanced clusters
- Optional pairwise evaluation (precision/recall/F1) using MatchGroupId
"""

import os
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
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

# -------- CONFIG (tune these) ----------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

BATCH_SIZE = 256
EPOCHS = 30
LR = 1e-3
TEMPERATURE = 0.5
PROJ_DIM = 128           # final embedding dim
CHAR_EMB = 16
CONV_OUT = 48
SEQ_LEN = 64

# Fields (confirmed)
FIELDS = [
    "TransactionRefNo",
    "MerchantRefNum",
    "AcquireRefNumber",
    "WebOrderNumber",
    "PONumber",
    "CardNo"
]

# DBSCAN hyper-search grid
DBSCAN_EPS = [0.2, 0.5, 0.8, 1.0]
DBSCAN_MIN_SAMPLES = [2, 3, 5]

# -----------------------------
# 0) Load data (replace with pd.read_csv)
# -----------------------------
# Minimal example (replace with your full dataset)
data = {
    "DocumentDate": ["02/01/2025","02/01/2025"],
    "DocType": ["unknown","unknown"],
    "TransactionType": ["SAP","SAP"],
    "Amount": [7800, -7800],
    "TransactionRefNo": ["W1dsfsafjdjfb","W1dsfsafjdjfb"],
    "MerchantRefNum": ["777741344598","777741344598"],
    "CR_DR": ["CR","DR"],
    "PONumber": ["13u350u5u05","13u350u5u05"],
    "CardNo": ["13415555531535","13415555531535"],
    "AcquireRefNumber": ["unknown","unknown"],
    "WebOrderNumber": ["W1342421414","W1342421414"],
    "MatchGroupId": ["14443553","14443553"],
    "Source": ["SAP","SAP"],
    "SourceType": ["Internal","Internal"]
}
df = pd.DataFrame(data)
df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0.0)
df["DocumentDate"] = pd.to_datetime(df["DocumentDate"], errors="coerce", dayfirst=True)
df["CR_DR"] = df["CR_DR"].fillna("CR")
df["SignedAmount"] = df["Amount"].astype(float) * df["CR_DR"].map({"CR":1.0,"DR":-1.0}).fillna(1.0)

# optional small categorical encodings (kept for final features)
label_cols = [c for c in ["DocType","TransactionType","Source","SourceType"] if c in df.columns]
for col in label_cols:
    le = LabelEncoder()
    df[col + "_enc"] = le.fit_transform(df[col].astype(str))

# -----------------------------
# 1) Build character vocabulary (from selected fields)
# -----------------------------
all_text = ""
for f in FIELDS:
    if f in df.columns:
        all_text += " ".join(df[f].astype(str).tolist()) + " "
chars = sorted(set([c for c in all_text]))
if len(chars) == 0:
    chars = list("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_-/.")
char_to_idx = {c:i+1 for i,c in enumerate(chars)}  # 0 reserved for PAD
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

# -----------------------------
# 2) Dataset for contrastive training using MatchGroupId positives (Option A)
# -----------------------------
class MatchGroupContrastiveDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.groups = defaultdict(list)
        for idx, mg in enumerate(self.df["MatchGroupId"].astype(str)):
            self.groups[mg].append(idx)
        # anchors are rows belonging to groups with at least 2 members
        self.anchors = [i for mg, idxs in self.groups.items() for i in idxs if len(idxs) >= 2]
    def __len__(self):
        return len(self.anchors)
    def __getitem__(self, idx):
        anchor_idx = self.anchors[idx]
        mg = str(self.df.loc[anchor_idx, "MatchGroupId"])
        members = self.groups[mg]
        pos = anchor_idx
        # choose a positive different from anchor
        while pos == anchor_idx:
            pos = random.choice(members)
        x1 = np.array(build_row_seq(self.df.loc[anchor_idx]), dtype=np.int64)
        x2 = np.array(build_row_seq(self.df.loc[pos]), dtype=np.int64)
        a1 = float(self.df.loc[anchor_idx, "SignedAmount"])
        a2 = float(self.df.loc[pos, "SignedAmount"])
        return x1, x2, a1, a2

dataset = MatchGroupContrastiveDataset(df)
loader = DataLoader(dataset, batch_size=min(BATCH_SIZE, len(dataset)), shuffle=True, drop_last=False)

# -----------------------------
# 3) Model: field-specific small CNN encoders -> Transformer over field vectors
#    SignedAmount is embedded inside the model (learnable projection) and concatenated
# -----------------------------
class SmallFieldCNN(nn.Module):
    def __init__(self, vocab_size, char_emb=CHAR_EMB, conv_out=CONV_OUT, kernel=5):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, char_emb, padding_idx=IDX_PAD)
        self.conv = nn.Conv1d(char_emb, conv_out, kernel_size=kernel, padding=kernel//2)
        self.pool = nn.AdaptiveMaxPool1d(1)
    def forward(self, seq):  # seq: (batch, seq_len)
        x = self.embed(seq)            # (batch, seq_len, char_emb)
        x = x.transpose(1,2)           # (batch, char_emb, seq_len)
        x = F.relu(self.conv(x))       # (batch, conv_out, seq_len)
        x = self.pool(x).squeeze(-1)   # (batch, conv_out)
        return x

class FieldSpecificHybridEncoder(nn.Module):
    def __init__(self, vocab_size, seq_per_field=SEQ_LEN, num_fields=len(FIELDS),
                 conv_out=CONV_OUT, trans_dim=128, nhead=4, num_layers=2, proj_dim=PROJ_DIM):
        super().__init__()
        self.num_fields = num_fields
        self.seq_per_field = seq_per_field
        # create a distinct small encoder per field
        self.field_encoders = nn.ModuleList([SmallFieldCNN(vocab_size, conv_out=conv_out) for _ in range(num_fields)])
        encoder_layer = nn.TransformerEncoderLayer(d_model=conv_out, nhead=nhead, dim_feedforward=trans_dim, activation='relu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # signed amount projection (map scalar -> conv_out)
        self.amount_proj = nn.Sequential(nn.Linear(1, conv_out), nn.ReLU())
        # final projection head (maps conv_out -> proj_dim)
        self.proj = nn.Sequential(nn.Linear(conv_out, conv_out), nn.ReLU(), nn.Linear(conv_out, proj_dim))
    def forward(self, seq_batch, amount_batch):
        """
        seq_batch: (batch, SEQ_TOTAL) long tensor
        amount_batch: (batch,) float tensor (scaled)
        """
        b = seq_batch.size(0)
        x = seq_batch.view(b, self.num_fields, self.seq_per_field)  # (batch, num_fields, seq_len)
        field_vecs = []
        for i in range(self.num_fields):
            seq_f = x[:, i, :].long()           # (batch, seq_len)
            v = self.field_encoders[i](seq_f)   # (batch, conv_out)
            field_vecs.append(v.unsqueeze(1))
        field_stack = torch.cat(field_vecs, dim=1)  # (batch, num_fields, conv_out)
        # transformer expects (seq_len=num_fields, batch, dim)
        tr_in = field_stack.transpose(0,1)          # (num_fields, batch, conv_out)
        tr_out = self.transformer(tr_in)            # (num_fields, batch, conv_out)
        pooled = tr_out.mean(dim=0)                 # (batch, conv_out)
        amt_proj = self.amount_proj(amount_batch.view(-1,1))  # (batch, conv_out)
        combined = pooled + amt_proj                # incorporate amount signal (elementwise)
        z = self.proj(combined)                     # (batch, proj_dim)
        z = F.normalize(z, p=2, dim=1)
        return z

model = FieldSpecificHybridEncoder(VOCAB_SIZE, seq_per_field=SEQ_LEN, num_fields=len(FIELDS)).to(DEVICE)

# -----------------------------
# 4) InfoNCE / NT-Xent loss for pairs (batch of positive pairs)
# -----------------------------
def nt_xent_pair_loss(z1, z2, temperature=TEMPERATURE):
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)             # (2B, D)
    sim = torch.matmul(z, z.T) / temperature   # (2B, 2B)
    mask = (~torch.eye(2*B, 2*B, dtype=torch.bool, device=DEVICE)).float()
    positives = torch.cat([torch.diag(sim, B), torch.diag(sim, -B)], dim=0)  # (2B,)
    nom = torch.exp(positives)
    denom = (mask * torch.exp(sim)).sum(dim=1)
    loss = -torch.log(nom / denom)
    return loss.mean()

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# -----------------------------
# 5) Scale amounts for stable training (robust standardization)
# -----------------------------
amounts = df["SignedAmount"].values.reshape(-1,1)
amt_scaler = StandardScaler()
if len(amounts) > 1:
    amt_scaler.fit(amounts)
else:
    amt_scaler.mean_ = np.array([0.0]); amt_scaler.scale_ = np.array([1.0])
def scale_amounts(x):
    x = np.array(x).reshape(-1,1).astype(float)
    return (x - amt_scaler.mean_) / (amt_scaler.scale_ + 1e-9)

# -----------------------------
# 6) Contrastive training loop (using MatchGroupId positives)
# -----------------------------
model.train()
for epoch in range(EPOCHS):
    epoch_loss = 0.0
    n = 0
    for x1_np, x2_np, a1, a2 in loader:
        x1 = torch.tensor(x1_np, dtype=torch.long, device=DEVICE)
        x2 = torch.tensor(x2_np, dtype=torch.long, device=DEVICE)
        a1_scaled = torch.tensor(scale_amounts(a1), dtype=torch.float32, device=DEVICE).squeeze(1)
        a2_scaled = torch.tensor(scale_amounts(a2), dtype=torch.float32, device=DEVICE).squeeze(1)
        optimizer.zero_grad()
        z1 = model(x1, a1_scaled)
        z2 = model(x2, a2_scaled)
        loss = nt_xent_pair_loss(z1, z2, TEMPERATURE)
        loss.backward()
        optimizer.step()
        b = x1.size(0)
        epoch_loss += loss.item() * b
        n += b
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"[Contrastive] Epoch {epoch+1}/{EPOCHS}, avg_loss={epoch_loss / max(1,n):.6f}")

# -----------------------------
# 7) Produce final embeddings for all rows (inference)
# -----------------------------
model.eval()
with torch.no_grad():
    seq_tensor = torch.tensor(sequences, dtype=torch.long, device=DEVICE)
    amt_scaled_all = torch.tensor(scale_amounts(df["SignedAmount"].values), dtype=torch.float32, device=DEVICE).squeeze(1)
    embeddings = model(seq_tensor, amt_scaled_all).cpu().numpy()  # (n_rows, PROJ_DIM)

# -----------------------------
# 8) Build final feature matrix for clustering
# Here we use the learned embedding alone (optionally append categorical encodings)
# -----------------------------
cat_cols_enc = [c + "_enc" for c in label_cols] if len(label_cols) > 0 else []
cat_feats = df[cat_cols_enc].values if len(cat_cols_enc) > 0 else np.zeros((len(df), 0))
X = np.hstack([embeddings, cat_feats])

# -----------------------------
# 9) DBSCAN hyperparameter search (eps, min_samples) using silhouette on non-noise
# -----------------------------
best_score = -1.0
best_labels = None
best_params = None

if len(X) < 2:
    best_labels = np.array([-1] * len(X))
    best_params = (None, None)
else:
    for eps in DBSCAN_EPS:
        for ms in DBSCAN_MIN_SAMPLES:
            try:
                db = DBSCAN(eps=eps, min_samples=ms, metric='euclidean')
                labels = db.fit_predict(X)
                mask = labels != -1
                if mask.sum() < 2:
                    score = -1.0
                else:
                    score = silhouette_score(X[mask], labels[mask])
                if score > best_score:
                    best_score = score
                    best_labels = labels
                    best_params = (eps, ms)
            except Exception:
                continue

if best_labels is None:
    # fallback
    db = DBSCAN(eps=0.5, min_samples=2, metric='euclidean')
    best_labels = db.fit_predict(X)
    best_params = ("fallback", "fallback")

df["PredCluster"] = best_labels
print("DBSCAN best params:", best_params, "best silhouette:", best_score)

# -----------------------------
# 10) Optional evaluation vs MatchGroupId (pairwise precision/recall/F1)
# -----------------------------
if "MatchGroupId" in df.columns:
    true_pairs = set()
    pred_pairs = set()
    n = len(df)
    for i, j in combinations(range(n), 2):
        same_true = (str(df.loc[i, "MatchGroupId"]) == str(df.loc[j, "MatchGroupId"]))
        same_pred = (df.loc[i, "PredCluster"] != -1) and (df.loc[i, "PredCluster"] == df.loc[j, "PredCluster"])
        if same_true:
            true_pairs.add((i, j))
        if same_pred:
            pred_pairs.add((i, j))
    tp = len(true_pairs & pred_pairs)
    fp = len(pred_pairs - true_pairs)
    fn = len(true_pairs - pred_pairs)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    print(f"Pairwise evaluation -> precision: {prec:.4f}, recall: {rec:.4f}, f1: {f1:.4f}")

# -----------------------------
# 11) Save / Output
# -----------------------------
print("Sample output (first rows):")
print(df[["SignedAmount", "PredCluster"] + cat_cols_enc].head())
df.to_csv("clusters_field_specific_contrastive_dbscan.csv", index=False)
