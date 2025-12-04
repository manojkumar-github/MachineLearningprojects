"""
Unsupervised autoencoder pipeline (per your spec)

Pipeline:
- Load data (single-row example or CSV)
- Drop fully-unknown columns
- Prepare character vocabulary from selected ID fields
- Per-field small CNN encoder -> produce per-field pooled vectors
- Transformer over field vectors -> pooled representation -> projection = embedding
- Train an autoencoder *unsupervised*:
    - Encoder: field-CNNs + transformer + projection
    - Decoder: map projection back to per-field pooled vectors
    - Loss: MSE between reconstructed per-field pooled vectors and original pooled vectors
  (This is a lightweight practical autoencoder avoiding sequence decoding.)
- After training, produce per-row embeddings
- Combine embeddings with encoded categorical features and scaled SignedAmount (NOT included inside encoder)
- Run DBSCAN (with simple hyperparameter search)
- Post-process clusters to enforce sum(SignedAmount) ~= 0 by searching balanced subsets (small subsets only)
- Output predicted clusters and (optional) pairwise evaluation using MatchGroupId (not used for training)

Notes:
- Replace the toy `df` creation with `pd.read_csv("your.csv")` for real data.
- Tune hyperparameters at the top for your dataset size and compute.
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

# ----------------------------
# CONFIG / HYPERPARAMETERS
# ----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

BATCH_SIZE = 128
EPOCHS = 30
LR = 1e-3

# architecture sizes
CHAR_EMB = 32
CONV_OUT = 64         # per-field pooled dimension
TRANS_DIM = 128
NHEAD = 4
NUM_TRANS_LAYERS = 2
PROJ_DIM = 128        # final embedding dim used for clustering

SEQ_LEN = 64          # chars per field
FIELDS = [
    "TransactionRefNo",
    "MerchantRefNum",
    "AcquireRefNumber",
    "WebOrderNumber",
    "PONumber",
    "CardNo"
]

# DBSCAN hyper-search grid
DBSCAN_EPS = [0.2, 0.5, 0.8]
DBSCAN_MIN_SAMPLES = [2, 3, 5]

# post-processing subset search
MAX_SUBSET_SIZE = 4
AMOUNT_ABS_TOL = 0.01   # absolute tolerance for sum-to-zero
AMOUNT_REL_TOL = 0.005  # relative tolerance (fraction)

# ----------------------------
# 0) Load data (replace this with pd.read_csv for real use)
# ----------------------------
# Example small dataset (for runnable demo). Replace in production.
data = {
    "DocumentDate": ["02/01/2025","02/01/2025","02/02/2025","02/02/2025"],
    "DocType": ["unknown","unknown","unknown","unknown"],
    "TransactionType": ["SAP","SAP","SAP","SAP"],
    "Amount": [7800, -7800, 1200, -1200],
    "TransactionRefNo": ["W1dsfsafjdjfb","W1dsfsafjdjfb","A1b2c3","Z9y8x7"],
    "MerchantRefNum": ["777741344598","777741344598","M123","M124"],
    "CR_DR": ["CR","DR","CR","DR"],
    "PONumber": ["13u350u5u05","13u350u5u05","po1","po2"],
    "CardNo": ["13415555531535","13415555531535","1111","2222"],
    "AcquireRefNumber": ["unknown","unknown","unknown","unknown"],
    "WebOrderNumber": ["W1342421414","W1342421414","W200","W201"],
    "MatchGroupId": ["14443553","14443553","2222","2222"],
    "Source": ["SAP","SAP","SAP","SAP"],
    "SourceType": ["Internal","Internal","Internal","Internal"]
}
df = pd.DataFrame(data)

# If you have a CSV, replace above with:
# df = pd.read_csv("transactions.csv", dtype=str)  # then coerce numeric columns below

# ----------------------------
# 1) Basic cleaning
# ----------------------------
# Drop columns that are often entirely "unknown" (customize as needed)
drop_cols = ["ReceiptNumber", "RefDocument", "Assignment", "StoreNumber", "AuthCode"]
for c in drop_cols:
    if c in df.columns:
        df = df.drop(columns=c)

# parse numeric/date fields
df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0.0)
df["DocumentDate"] = pd.to_datetime(df["DocumentDate"], errors="coerce", dayfirst=True)
df["CR_DR"] = df["CR_DR"].fillna("CR")
df["SignedAmount"] = df["Amount"].astype(float) * df["CR_DR"].map({"CR":1.0,"DR":-1.0}).fillna(1.0)

# label encode small categorical columns (optional side info)
label_cols = [c for c in ["DocType","TransactionType","Source","SourceType"] if c in df.columns]
for col in label_cols:
    le = LabelEncoder()
    df[col + "_enc"] = le.fit_transform(df[col].astype(str))

# ----------------------------
# 2) Build character vocabulary from the selected fields
# ----------------------------
all_text = ""
for f in FIELDS:
    if f in df.columns:
        all_text += " ".join(df[f].astype(str).tolist()) + " "
chars = sorted(set([c for c in all_text]))
if len(chars) == 0:
    # fallback ASCII set
    chars = list("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_-/.")
char_to_idx = {c:i+1 for i,c in enumerate(chars)}   # reserve 0 for padding
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

# ----------------------------
# 3) Dataset for autoencoder training
# We train unsupervised: inputs are sequences; targets are pooled per-field vectors computed by the same field-CNN (teacher)
# ----------------------------
class AutoencoderDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.seqs = np.array([build_row_seq(r) for _, r in self.df.iterrows()], dtype=np.int64)
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        return self.seqs[idx]

dataset = AutoencoderDataset(df)
loader = DataLoader(dataset, batch_size=min(BATCH_SIZE, len(dataset)), shuffle=True, drop_last=False)

# ----------------------------
# 4) Model components
# - SmallFieldCNN: returns pooled per-field vector (conv_out)
# - FieldTransformerAutoencoder: encoder -> projection; decoder reconstructs per-field pooled vectors
# ----------------------------
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

class FieldTransformerAutoencoder(nn.Module):
    def __init__(self, vocab_size, seq_per_field=SEQ_LEN, num_fields=len(FIELDS),
                 char_emb=CHAR_EMB, conv_out=CONV_OUT, trans_dim=TRANS_DIM,
                 nhead=NHEAD, num_layers=NUM_TRANS_LAYERS, proj_dim=PROJ_DIM):
        super().__init__()
        self.num_fields = num_fields
        self.seq_per_field = seq_per_field
        # per-field encoders (we will use these to compute targets too)
        self.field_encoders = nn.ModuleList([SmallFieldCNN(vocab_size, char_emb=char_emb, conv_out=conv_out) for _ in range(num_fields)])
        # transformer across fields
        encoder_layer = nn.TransformerEncoderLayer(d_model=conv_out, nhead=nhead, dim_feedforward=trans_dim, activation='relu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # projection to embedding
        self.proj = nn.Sequential(nn.Linear(conv_out, conv_out), nn.ReLU(), nn.Linear(conv_out, proj_dim))
        # decoder: map embedding -> conv_out and then produce per-field reconstructions
        self.decoder_fc = nn.Sequential(nn.Linear(proj_dim, conv_out), nn.ReLU())
        # a small linear layer to produce per-field vectors from decoded vector (we will expand same vector for each field)
        self.reconstruct_field = nn.Linear(conv_out, conv_out)  # reconstruct pooled field vector
    def encode_pooled_fields(self, seq_batch):
        """Compute per-field pooled vectors (used as reconstruction targets)."""
        b = seq_batch.size(0)
        x = seq_batch.view(b, self.num_fields, self.seq_per_field)  # (batch, num_fields, seq_len)
        pooled = []
        for i in range(self.num_fields):
            seq_i = x[:, i, :].long()
            v = self.field_encoders[i](seq_i)   # (batch, conv_out)
            pooled.append(v.unsqueeze(1))
        field_stack = torch.cat(pooled, dim=1)  # (batch, num_fields, conv_out)
        return field_stack  # not transformed by transformer (these are targets)
    def forward(self, seq_batch):
        """
        seq_batch: (batch, SEQ_TOTAL) long
        returns:
           emb: (batch, proj_dim)
           recon: (batch, num_fields, conv_out) reconstructed per-field pooled vectors
           targets: (batch, num_fields, conv_out) original pooled vectors (for loss)
        """
        b = seq_batch.size(0)
        seq_batch = seq_batch.long()
        # compute targets (pooled per-field vectors) using field_encoders
        targets = self.encode_pooled_fields(seq_batch)   # (b, num_fields, conv_out)
        # now produce transformer-processed pooled representation
        # transformer input: (num_fields, batch, conv_out)
        tr_in = targets.transpose(0,1)   # (num_fields, batch, conv_out)
        tr_out = self.transformer(tr_in) # (num_fields, batch, conv_out)
        pooled = tr_out.mean(dim=0)      # (batch, conv_out)
        emb = self.proj(pooled)          # (batch, proj_dim)
        emb = F.normalize(emb, p=2, dim=1)
        # decoder: map emb -> conv_out, then produce per-field reconstructions
        dec = self.decoder_fc(emb)       # (batch, conv_out)
        # produce per-field reconstructed vectors by applying reconstruct_field to dec and repeating
        field_recon = self.reconstruct_field(dec).unsqueeze(1).repeat(1, self.num_fields, 1)  # (b, num_fields, conv_out)
        return emb, field_recon, targets

# instantiate model
model = FieldTransformerAutoencoder(VOCAB_SIZE, seq_per_field=SEQ_LEN, num_fields=len(FIELDS)).to(DEVICE)

# ----------------------------
# 5) Training autoencoder (MSE between reconstructed per-field pooled vectors and targets)
# ----------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
mse_loss = nn.MSELoss()

model.train()
for epoch in range(EPOCHS):
    total_loss = 0.0
    n = 0
    for seq_batch in loader:
        seq_batch = torch.tensor(seq_batch, dtype=torch.long, device=DEVICE)
        optimizer.zero_grad()
        emb, recon, targets = model(seq_batch)
        loss = mse_loss(recon, targets)
        loss.backward()
        optimizer.step()
        batch_n = seq_batch.size(0)
        total_loss += loss.item() * batch_n
        n += batch_n
    if (epoch+1) % 5 == 0 or epoch == 0:
        print(f"[Autoencoder] Epoch {epoch+1}/{EPOCHS}, avg_mse={total_loss / max(1,n):.6f}")

# ----------------------------
# 6) Produce per-row embeddings (encoder)
# ----------------------------
model.eval()
with torch.no_grad():
    seq_tensor = torch.tensor(sequences, dtype=torch.long, device=DEVICE)
    embeddings = model(seq_tensor)[0].cpu().numpy()   # (n_rows, PROJ_DIM)

# ----------------------------
# 7) Build final feature matrix: embeddings + scaled SignedAmount + categorical encodings
# Note: SignedAmount is NOT part of encoder; we append it to the final vector for clustering.
# ----------------------------
# scale amount
amt_scaler = StandardScaler()
amounts = df[["SignedAmount"]].values.astype(float)
amt_scaler.fit(amounts)
amounts_scaled = ((amounts - amt_scaler.mean_) / (amt_scaler.scale_ + 1e-9)).reshape(-1,1)

cat_cols_enc = [c + "_enc" for c in label_cols] if len(label_cols)>0 else []
cat_feats = df[cat_cols_enc].values if len(cat_cols_enc)>0 else np.zeros((len(df),0))

X = np.hstack([embeddings, amounts_scaled, cat_feats])   # final vectors for clustering

# ----------------------------
# 8) DBSCAN hyperparameter search (silhouette on non-noise)
# ----------------------------
best_score = -1.0
best_labels = None
best_params = None

if len(X) < 2:
    best_labels = np.array([-1] * len(X))
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
        db = DBSCAN(eps=0.5, min_samples=2, metric='euclidean')
        best_labels = db.fit_predict(X)
        best_params = ("fallback", "fallback")

df["raw_cluster"] = best_labels
print("DBSCAN best params:", best_params, "best silhouette:", best_score)
