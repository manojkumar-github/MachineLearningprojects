"""
Supervised classification pipeline using field-wise hybrid Transformer encoder
and CrossEntropyLoss(ignore_index=0) as requested.

Pipeline:
- Field-specific small CNN encoders -> Transformer over field vectors
- SignedAmount included as a learnable feature inside the encoder
- Supervised training with nn.CrossEntropyLoss(ignore_index=0) where:
    * class indices 1..K correspond to MatchGroupIds with >=2 members (training classes)
    * label 0 = ignore (singletons / groups of size 1 or noise)
- After supervised training, extract fixed embeddings from the trained encoder
  (the layer before the classifier) and run DBSCAN clustering (auto hyper-search)
- No greedy matching / no post-processing
- Optional pairwise evaluation (precision/recall/F1) using MatchGroupId

USAGE:
- Replace the example `df` with your pd.read_csv of the full dataset.
- Tune hyperparameters at the top (EPOCHS, BATCH_SIZE, PROJ_DIM, DBSCAN grid).
"""

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
# CONFIG
# ----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

BATCH_SIZE = 256
EPOCHS = 25
LR = 1e-3
TEMPERATURE = 0.5
PROJ_DIM = 128      # embedding dim from encoder (used for clustering)
CHAR_EMB = 32
CONV_OUT = 64
KERNEL = 5
SEQ_LEN = 64

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

# ----------------------------
# 0) Load data - replace this with your pd.read_csv in production
# ----------------------------
# Small illustrative example; replace with full dataset
data = {
    "DocumentDate": ["02/01/2025","02/01/2025","02/01/2025"],
    "DocType": ["unknown","unknown","unknown"],
    "TransactionType": ["SAP","SAP","SAP"],
    "Amount": [7800, -7800, 500],
    "TransactionRefNo": ["W1dsfsafjdjfb","W1dsfsafjdjfb","X9aa12"],
    "MerchantRefNum": ["777741344598","777741344598","555123"],
    "CR_DR": ["CR","DR","CR"],
    "PONumber": ["13u350u5u05","13u350u5u05","po123"],
    "CardNo": ["13415555531535","13415555531535","0000"],
    "AcquireRefNumber": ["unknown","unknown","unknown"],
    "WebOrderNumber": ["W1342421414","W1342421414","W0001"],
    "MatchGroupId": ["14443553","14443553","999999"],
    "Source": ["SAP","SAP","SAP"],
    "SourceType": ["Internal","Internal","Internal"]
}
df = pd.DataFrame(data)

# ensure numeric types
df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0.0)
df["CR_DR"] = df["CR_DR"].fillna("CR")
df["SignedAmount"] = df["Amount"].astype(float) * df["CR_DR"].map({"CR":1.0,"DR":-1.0}).fillna(1.0)

# optional small categorical encodings
label_cols = [c for c in ["DocType","TransactionType","Source","SourceType"] if c in df.columns]
for col in label_cols:
    le = LabelEncoder()
    df[col + "_enc"] = le.fit_transform(df[col].astype(str))

# ----------------------------
# 1) Build character vocabulary from selected fields
# ----------------------------
all_text = ""
for f in FIELDS:
    if f in df.columns:
        all_text += " ".join(df[f].astype(str).tolist()) + " "
chars = sorted(set([c for c in all_text]))
if len(chars) == 0:
    chars = list("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_-/.")
char_to_idx = {c:i+1 for i,c in enumerate(chars)}  # 0 reserved for padding
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
# 2) Build supervised class mapping from MatchGroupId
# We map MatchGroupIds with >=2 members to classes 1..K.
# Label 0 is reserved as IGNORE (singletons / groups of size 1).
# ----------------------------
group_to_idxs = defaultdict(list)
for idx, mg in enumerate(df["MatchGroupId"].astype(str)):
    group_to_idxs[mg].append(idx)

# Identify groups with size >= 2 as trainable classes
train_groups = [mg for mg, idxs in group_to_idxs.items() if len(idxs) >= 2]
train_groups.sort()
group_to_class = {mg: (i+1) for i, mg in enumerate(train_groups)}  # classes start at 1
num_classes = len(train_groups) + 1  # +1 to include class 0 (ignored) in model output dims

# Build labels: class index for rows in train_groups, else 0
labels = np.zeros(len(df), dtype=np.int64)
for i, mg in enumerate(df["MatchGroupId"].astype(str)):
    if mg in group_to_class:
        labels[i] = group_to_class[mg]
    else:
        labels[i] = 0  # ignored by loss

df["train_label"] = labels

# ----------------------------
# 3) Supervised Dataset for classification
# Each sample returns: sequence (SEQ_TOTAL), scaled amount, label (int)
# ----------------------------
class SupervisedSeqDataset(Dataset):
    def __init__(self, df, seq_array, labels, amt_scaler=None):
        self.df = df.reset_index(drop=True)
        self.seq = seq_array
        self.labels = labels
        # amount scaler (fit outside if needed)
        if amt_scaler is None:
            self.amt_scaler = StandardScaler()
            self.amt_scaler.fit(self.df[["SignedAmount"]].values)
        else:
            self.amt_scaler = amt_scaler
        self.amounts = (self.df[["SignedAmount"]].values - self.amt_scaler.mean_) / (self.amt_scaler.scale_ + 1e-9)
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        return self.seq[idx].astype(np.int64), float(self.amounts[idx]), int(self.labels[idx])

# fit amount scaler on full data (or training split)
amt_scaler = StandardScaler()
amt_scaler.fit(df[["SignedAmount"]].values)

dataset = SupervisedSeqDataset(df, sequences, labels, amt_scaler=amt_scaler)
loader = DataLoader(dataset, batch_size=min(BATCH_SIZE, len(dataset)), shuffle=True, drop_last=False)

# ----------------------------
# 4) Model: Field-wise small CNN encoders -> Transformer -> combine with amount -> classifier
# We'll provide a method to extract embeddings (before classifier) for DBSCAN clustering.
# ----------------------------
class SmallFieldCNN(nn.Module):
    def __init__(self, vocab_size, char_emb=CHAR_EMB, conv_out=CONV_OUT, kernel=KERNEL):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, char_emb, padding_idx=IDX_PAD)
        self.conv = nn.Conv1d(char_emb, conv_out, kernel_size=kernel, padding=kernel//2)
        self.pool = nn.AdaptiveMaxPool1d(1)
    def forward(self, seq):
        x = self.embed(seq)           # (batch, seq_len, char_emb)
        x = x.transpose(1,2)          # (batch, char_emb, seq_len)
        x = F.relu(self.conv(x))      # (batch, conv_out, seq_len)
        x = self.pool(x).squeeze(-1)  # (batch, conv_out)
        return x

class FieldWiseTransformerClassifier(nn.Module):
    def __init__(self, vocab_size, seq_per_field=SEQ_LEN, num_fields=len(FIELDS),
                 conv_out=CONV_OUT, trans_dim=128, nhead=4, num_layers=2, proj_dim=PROJ_DIM, num_classes=2):
        super().__init__()
        self.num_fields = num_fields
        self.seq_per_field = seq_per_field
        # field-specific small encoders
        self.field_encoders = nn.ModuleList([SmallFieldCNN(vocab_size, conv_out=conv_out) for _ in range(num_fields)])
        # transformer across field vectors
        encoder_layer = nn.TransformerEncoderLayer(d_model=conv_out, nhead=nhead, dim_feedforward=trans_dim, activation='relu')
        self.field_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # amount projection into same dim
        self.amount_proj = nn.Sequential(nn.Linear(1, conv_out), nn.ReLU())
        # projection to embedding (used for clustering)
        self.proj = nn.Sequential(nn.Linear(conv_out, conv_out), nn.ReLU(), nn.Linear(conv_out, proj_dim))
        # classifier (include class 0 in output dims, we will use ignore_index=0)
        self.classifier = nn.Linear(proj_dim, num_classes)
    def forward(self, seq_batch, amount_batch):
        """
        seq_batch: (batch, SEQ_TOTAL) long
        amount_batch: (batch,) float scaled
        returns:
           logits: (batch, num_classes)
           embedding: (batch, proj_dim)  -- normalized embedding
        """
        b = seq_batch.size(0)
        x = seq_batch.view(b, self.num_fields, self.seq_per_field)
        field_vecs = []
        for i in range(self.num_fields):
            seq_i = x[:, i, :].long()
            v = self.field_encoders[i](seq_i)   # (batch, conv_out)
            field_vecs.append(v.unsqueeze(1))
        field_stack = torch.cat(field_vecs, dim=1)  # (batch, num_fields, conv_out)
        tr_in = field_stack.transpose(0,1)          # (num_fields, batch, conv_out)
        tr_out = self.field_transformer(tr_in)      # (num_fields, batch, conv_out)
        pooled = tr_out.mean(dim=0)                 # (batch, conv_out)
        amt_proj = self.amount_proj(amount_batch.view(-1,1))  # (batch, conv_out)
        combined = pooled + amt_proj
        emb = self.proj(combined)                   # (batch, proj_dim)
        emb = F.normalize(emb, p=2, dim=1)
        logits = self.classifier(emb)               # (batch, num_classes)
        return logits, emb

model = FieldWiseTransformerClassifier(VOCAB_SIZE, seq_per_field=SEQ_LEN, num_fields=len(FIELDS),
                                       conv_out=CONV_OUT, trans_dim=128, nhead=4, num_layers=2,
                                       proj_dim=PROJ_DIM, num_classes=num_classes).to(DEVICE)

# ----------------------------
# 5) Loss and optimizer
# ----------------------------
criterion = nn.CrossEntropyLoss(ignore_index=0)  # 0 is ignored (singletons)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ----------------------------
# 6) Supervised training loop
# ----------------------------
model.train()
for epoch in range(EPOCHS):
    epoch_loss = 0.0
    n = 0
    for seq_np, amt_scalar, lbl in loader:
        seq = torch.tensor(seq_np, dtype=torch.long, device=DEVICE)
        amt = torch.tensor(amt_scalar, dtype=torch.float32, device=DEVICE)
        target = torch.tensor(lbl, dtype=torch.long, device=DEVICE)  # values in {0..K}
        optimizer.zero_grad()
        logits, emb = model(seq, amt)
        # CrossEntropyLoss expects targets in [0..C-1]; we use ignore_index=0 so
        # targets equal 0 are ignored (we still produce logits for class-0)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()
        batch_n = seq.size(0)
        epoch_loss += loss.item() * batch_n
        n += batch_n
    if (epoch+1) % 5 == 0 or epoch == 0:
        print(f"[Supervised] Epoch {epoch+1}/{EPOCHS}, avg_loss={epoch_loss / max(1,n):.6f}")

# ----------------------------
# 7) Extract embeddings for all rows (inference mode)
# ----------------------------
model.eval()
with torch.no_grad():
    seq_tensor = torch.tensor(sequences, dtype=torch.long, device=DEVICE)
    amt_scaled_all = torch.tensor((df[["SignedAmount"]].values - amt_scaler.mean_) / (amt_scaler.scale_ + 1e-9),
                                  dtype=torch.float32, device=DEVICE).squeeze(1)
    _, embeddings = model(seq_tensor, amt_scaled_all)
    embeddings = embeddings.cpu().numpy()  # shape (n_rows, PROJ_DIM)

# ----------------------------
# 8) DBSCAN clustering on embeddings (+ optional categorical)
# ----------------------------
cat_cols_enc = [c + "_enc" for c in label_cols] if len(label_cols) > 0 else []
cat_feats = df[cat_cols_enc].values if len(cat_cols_enc) > 0 else np.zeros((len(df), 0))
X = np.hstack([embeddings, cat_feats])

best_score = -1.0
best_labels = None
best_params = None

if len(X) < 2:
    best_labels = np.array([-1] * len(X))
else:
    for eps in DBSCAN_EPS:
        for ms in DBSCAN_MIN_SAMPLES:
            try:
                db = DBSCAN(eps=eps, min_samples=ms, metric="euclidean")
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
        db = DBSCAN(eps=0.5, min_samples=2, metric="euclidean")
        best_labels = db.fit_predict(X)
        best_params = ("fallback", "fallback")

df["PredCluster"] = best_labels
print("DBSCAN best params:", best_params, "best silhouette:", best_score)

# ----------------------------
# 9) Optional pairwise evaluation using MatchGroupId
# ----------------------------
if "MatchGroupId" in df.columns:
    true_pairs = set()
    pred_pairs = set()
    n = len(df)
    for i, j in combinations(range(n), 2):
        if str(df.loc[i, "MatchGroupId"]) == str(df.loc[j, "MatchGroupId"]):
            true_pairs.add((i, j))
        if df.loc[i, "PredCluster"] != -1 and df.loc[i, "PredCluster"] == df.loc[j, "PredCluster"]:
            pred_pairs.add((i, j))
    tp = len(true_pairs & pred_pairs)
    fp = len(pred_pairs - true_pairs)
    fn = len(true_pairs - pred_pairs)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    print(f"Pairwise evaluation -> precision: {prec:.4f}, recall: {rec:.4f}, f1: {f1:.4f}")

# ----------------------------
# 10) Save / output
# ----------------------------
print("Sample output (first rows):")
show_cols = ["SignedAmount", "PredCluster"] + ([c + "_enc" for c in label_cols] if label_cols else [])
print(df[show_cols].head())
df.to_csv("supervised_transformer_embeddings_dbscan.csv", index=False)
