"""
Full end-to-end pipeline: FIELD-WISE HYBRID TRANSFORMER encoder + Autoencoder + CUSTOM METRIC LOSS
Goal:
- Learn embeddings so that transactions within +/- 3 days AND those that form zero-sum amount pairs/groups
  are close in embedding space (no rule-based postprocessing required).
- Training is unsupervised: MatchGroupId is NOT used for training (only optional evaluation).
- SignedAmount is NOT fed into the encoder; it's used in the loss (pairwise amount-balancing mask)
  and appended to final features for clustering.
- After training, embeddings are combined with scaled SignedAmount and categorical encodings,
  DBSCAN clustering is run, and a small subset-search post-processing step enforces sum(SignedAmount)=0.

Notes:
- Replace the sample `df` with pd.read_csv("your.csv") to run on real data.
- Tune hyperparameters (EPOCHS, BATCH_SIZE, PROJ_DIM, loss weights, DBSCAN grid).
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

# Model/training hyperparams
BATCH_SIZE = 128
EPOCHS = 30
LR = 1e-3
PROJ_DIM = 128        # embedding dim
CHAR_EMB = 32
CONV_OUT = 64
KERNEL = 5
SEQ_LEN = 64

# Fields to encode (confirmed)
FIELDS = [
    "TransactionRefNo",
    "MerchantRefNum",
    "AcquireRefNumber",
    "WebOrderNumber",
    "PONumber",
    "CardNo"
]

# DBSCAN hyperparams grid
DBSCAN_EPS = [0.2, 0.5, 0.8]
DBSCAN_MIN_SAMPLES = [2, 3, 5]

# Custom loss hyperparams
DATE_MAX_DAYS = 3                     # ±3 days considered date-positive
AMOUNT_ABS_TOL = 0.01                 # absolute tolerance for amount zero-sum (currency units)
AMOUNT_REL_TOL = 0.01                 # relative tolerance (fraction of magnitude)
MARGIN = 1.0                          # margin for negative hinge loss
W_DATE = 1.0
W_AMOUNT = 2.0
W_NEG = 1.0
RECON_WEIGHT = 0.1                    # weight for autoencoder reconstruction loss (MSE)
# Subset postprocessing
MAX_SUBSET_SIZE = 4
POST_ABS_TOL = 0.01
POST_REL_TOL = 0.005

# ----------------------------
# 0) Example data (replace with your CSV)
# ----------------------------
data = {
    "DocumentDate": ["02/01/2025","02/01/2025","02/02/2025","02/05/2025"],
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

# Replace with:
# df = pd.read_csv("your_transactions.csv")
# then ensure numeric columns are cast as below

# ----------------------------
# 1) Basic cleaning & feature prep
# ----------------------------
# Drop fully-unknown columns if present (user-specified earlier)
drop_cols = ["ReceiptNumber", "RefDocument", "Assignment", "StoreNumber", "AuthCode"]
for c in drop_cols:
    if c in df.columns:
        df = df.drop(columns=c)

# Date and amount processing
df["DocumentDate"] = pd.to_datetime(df["DocumentDate"], errors="coerce", dayfirst=True)
df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0.0)
df["CR_DR"] = df.get("CR_DR", pd.Series(["CR"] * len(df))).fillna("CR")
df["SignedAmount"] = df["Amount"].astype(float) * df["CR_DR"].map({"CR":1.0,"DR":-1.0}).fillna(1.0)

# Build integer days for dates relative to min date (used in loss)
df["DocDay"] = (df["DocumentDate"] - df["DocumentDate"].min()).dt.days.fillna(0).astype(int)

# small categorical encodings (optional side info)
label_cols = [c for c in ["DocType","TransactionType","Source","SourceType"] if c in df.columns]
for col in label_cols:
    le = LabelEncoder()
    df[col + "_enc"] = le.fit_transform(df[col].astype(str))

# ----------------------------
# 2) Character vocabulary from chosen fields
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
# 3) Dataset for unsupervised training (autoencoder + metric loss)
# Each sample returns: sequence (SEQ_TOTAL), docday (int), signedamount (float)
# ----------------------------
class UnsupervisedDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.seq = np.array([build_row_seq(r) for _, r in self.df.iterrows()], dtype=np.int64)
        self.docday = self.df["DocDay"].values.astype(int)
        self.amounts = self.df["SignedAmount"].values.astype(float)
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        return self.seq[idx].astype(np.int64), int(self.docday[idx]), float(self.amounts[idx])

dataset = UnsupervisedDataset(df)
loader = DataLoader(dataset, batch_size=min(BATCH_SIZE, len(dataset)), shuffle=True, drop_last=False)

# ----------------------------
# 4) Model: Field-wise small CNN encoders -> Transformer -> projection (embedding)
# Decoder reconstructs per-field pooled vectors (lightweight autoencoder)
# ----------------------------
class SmallFieldCNN(nn.Module):
    def __init__(self, vocab_size, char_emb=CHAR_EMB, conv_out=CONV_OUT, kernel=KERNEL):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, char_emb, padding_idx=IDX_PAD)
        self.conv = nn.Conv1d(char_emb, conv_out, kernel_size=kernel, padding=kernel//2)
        self.pool = nn.AdaptiveMaxPool1d(1)
    def forward(self, seq):
        x = self.embed(seq)            # (batch, seq_len, char_emb)
        x = x.transpose(1,2)           # (batch, char_emb, seq_len)
        x = F.relu(self.conv(x))       # (batch, conv_out, seq_len)
        x = self.pool(x).squeeze(-1)   # (batch, conv_out)
        return x

class FieldTransformerAutoencoderMetric(nn.Module):
    def __init__(self, vocab_size, seq_per_field=SEQ_LEN, num_fields=len(FIELDS),
                 char_emb=CHAR_EMB, conv_out=CONV_OUT, trans_dim=128,
                 nhead=4, num_layers=2, proj_dim=PROJ_DIM):
        super().__init__()
        self.num_fields = num_fields
        self.seq_per_field = seq_per_field
        # per-field encoders (used both as target pooler and as encoder inputs)
        self.field_encoders = nn.ModuleList([SmallFieldCNN(vocab_size, char_emb=char_emb, conv_out=conv_out) for _ in range(num_fields)])
        # transformer across field vectors
        encoder_layer = nn.TransformerEncoderLayer(d_model=conv_out, nhead=nhead, dim_feedforward=trans_dim, activation='relu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # projection to embedding
        self.proj = nn.Sequential(nn.Linear(conv_out, conv_out), nn.ReLU(), nn.Linear(conv_out, proj_dim))
        # decoder: map embedding -> conv_out then reconstruct per-field pooled vectors
        self.decoder_fc = nn.Sequential(nn.Linear(proj_dim, conv_out), nn.ReLU())
        self.reconstruct_field = nn.Linear(conv_out, conv_out)
    def encode_pooled_fields(self, seq_batch):
        b = seq_batch.size(0)
        x = seq_batch.view(b, self.num_fields, self.seq_per_field)
        pooled = []
        for i in range(self.num_fields):
            seq_i = x[:, i, :].long()
            v = self.field_encoders[i](seq_i)   # (batch, conv_out)
            pooled.append(v.unsqueeze(1))
        field_stack = torch.cat(pooled, dim=1)  # (batch, num_fields, conv_out)
        return field_stack
    def forward(self, seq_batch):
        b = seq_batch.size(0)
        seq_batch = seq_batch.long()
        targets = self.encode_pooled_fields(seq_batch)   # (b, num_fields, conv_out)
        tr_in = targets.transpose(0,1)   # (num_fields, b, conv_out)
        tr_out = self.transformer(tr_in) # (num_fields, b, conv_out)
        pooled = tr_out.mean(dim=0)      # (b, conv_out)
        emb = self.proj(pooled)          # (b, proj_dim)
        emb = F.normalize(emb, p=2, dim=1)
        dec = self.decoder_fc(emb)       # (b, conv_out)
        field_recon = self.reconstruct_field(dec).unsqueeze(1).repeat(1, self.num_fields, 1)  # (b,num_fields,conv_out)
        return emb, field_recon, targets

# instantiate
model = FieldTransformerAutoencoderMetric(VOCAB_SIZE, seq_per_field=SEQ_LEN, num_fields=len(FIELDS)).to(DEVICE)

# ----------------------------
# 5) Custom loss components (operate on batch embeddings, batch docdays, batch amounts)
# ----------------------------
def date_positive_mask(batch_days, max_days=DATE_MAX_DAYS):
    # batch_days: tensor (B,) ints
    diff = torch.abs(batch_days.unsqueeze(0).float() - batch_days.unsqueeze(1).float())  # (B,B)
    return (diff <= max_days).float()

def amount_positive_mask(batch_amounts, abs_tol=AMOUNT_ABS_TOL, rel_tol=AMOUNT_REL_TOL):
    # batch_amounts: tensor (B,) floats
    sums = batch_amounts.unsqueeze(0) + batch_amounts.unsqueeze(1)  # (B,B)
    abs_mask = torch.abs(sums) <= abs_tol
    # relative tolerance based on magnitudes of pair
    mag = torch.max(torch.abs(batch_amounts.unsqueeze(0)), torch.abs(batch_amounts.unsqueeze(1)))
    rel_mask = torch.abs(sums) <= (rel_tol * (mag + 1e-9))
    return (abs_mask | rel_mask).float()

def pairwise_sq_dist(emb):
    # emb: (B, D)
    diff = emb.unsqueeze(1) - emb.unsqueeze(0)  # (B,B,D)
    return (diff ** 2).sum(-1)                   # (B,B)

def date_loss(emb, day_mask):
    # encourage small squared distance for date-positive pairs
    sqd = pairwise_sq_dist(emb)
    pos = day_mask
    if pos.sum() < 1:
        return torch.tensor(0.0, device=emb.device)
    return (pos * sqd).sum() / (pos.sum() + 1e-9)

def amount_loss(emb, amt_mask):
    sqd = pairwise_sq_dist(emb)
    pos = amt_mask
    if pos.sum() < 1:
        return torch.tensor(0.0, device=emb.device)
    return (pos * sqd).sum() / (pos.sum() + 1e-9)

def negative_loss(emb, pos_union_mask, margin=MARGIN):
    # push negatives away using hinge on margin
    # pos_union_mask: (B,B) binary where positives (date or amount) are 1
    neg_mask = 1.0 - pos_union_mask
    if neg_mask.sum() < 1:
        return torch.tensor(0.0, device=emb.device)
    dist = torch.sqrt(pairwise_sq_dist(emb) + 1e-6)
    hinge = F.relu(margin - dist)   # encourage dist >= margin
    return (neg_mask * hinge).sum() / (neg_mask.sum() + 1e-9)

# Combined metric loss
def metric_loss(emb, batch_days, batch_amounts, w_date=W_DATE, w_amount=W_AMOUNT, w_neg=W_NEG):
    day_mask = date_positive_mask(batch_days)
    amt_mask = amount_positive_mask(batch_amounts)
    pos_union = ((day_mask + amt_mask) > 0).float()
    ld = date_loss(emb, day_mask)
    la = amount_loss(emb, amt_mask)
    ln = negative_loss(emb, pos_union)
    return w_date * ld + w_amount * la + w_neg * ln

# ----------------------------
# 6) Training loop (autoencoder recon loss + metric loss)
# ----------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
mse_loss = nn.MSELoss()

model.train()
for epoch in range(EPOCHS):
    total_loss = 0.0
    total_mse = 0.0
    total_metric = 0.0
    n = 0
    for seq_batch_np, days_batch_np, amt_batch_np in loader:
        seq_batch = torch.tensor(seq_batch_np, dtype=torch.long, device=DEVICE)
        days_batch = torch.tensor(days_batch_np, dtype=torch.long, device=DEVICE)
        amt_batch = torch.tensor(amt_batch_np, dtype=torch.float32, device=DEVICE)
        optimizer.zero_grad()
        emb, recon, targets = model(seq_batch)
        # reconstruction loss (MSE between reconstructed per-field pooled vectors and original)
        loss_recon = mse_loss(recon, targets)
        # metric loss computed on embeddings and batch-level dates/amounts
        loss_metric = metric_loss(emb, days_batch, amt_batch)
        loss = RECON_WEIGHT * loss_recon + loss_metric
        loss.backward()
        optimizer.step()
        batch_n = seq_batch.size(0)
        total_loss += loss.item() * batch_n
        total_mse += loss_recon.item() * batch_n
        total_metric += loss_metric.item() * batch_n
        n += batch_n
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"[Train] Epoch {epoch+1}/{EPOCHS}, total_loss={total_loss/max(1,n):.6f}, mse={total_mse/max(1,n):.6f}, metric={total_metric/max(1,n):.6f}")

# ----------------------------
# 7) Extract embeddings (inference)
# ----------------------------
model.eval()
with torch.no_grad():
    seq_tensor = torch.tensor(sequences, dtype=torch.long, device=DEVICE)
    embeddings = model(seq_tensor)[0].cpu().numpy()   # (n_rows, PROJ_DIM)

# ----------------------------
# 8) Build final feature matrix for clustering
# Combine embeddings with scaled SignedAmount and categorical encodings (optional)
# ----------------------------
# scale amount
amt_scaler = StandardScaler()
amt_scaler.fit(df[["SignedAmount"]].values)
amounts_scaled = ((df[["SignedAmount"]].values - amt_scaler.mean_) / (amt_scaler.scale_ + 1e-9)).reshape(-1,1)

cat_cols_enc = [c + "_enc" for c in label_cols] if len(label_cols)>0 else []
cat_feats = df[cat_cols_enc].values if len(cat_cols_enc)>0 else np.zeros((len(df),0))

X = np.hstack([embeddings, amounts_scaled, cat_feats])  # final vectors for clustering

# ----------------------------
# 9) DBSCAN hyperparameter search (silhouette on non-noise)
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

# ----------------------------
# 10) Post-process clusters to enforce sum(SignedAmount) ~= 0 by searching balanced subsets
# This is limited to small subsets (MAX_SUBSET_SIZE) to keep compute reasonable.
# ----------------------------
n = len(df)
final_cluster_map = {i: -1 for i in range(n)}
next_cid = 0

def subset_balancing_search(member_indices):
    global next_cid
    assigned = set()
    amounts = {i: float(df.loc[i, "SignedAmount"]) for i in member_indices}
    for r in range(min(MAX_SUBSET_SIZE, len(member_indices)), 1, -1):
        for combo in combinations(member_indices, r):
            if any(c in assigned or final_cluster_map[c] != -1 for c in combo):
                continue
            s = sum(amounts[c] for c in combo)
            abs_tol = POST_ABS_TOL
            rel_tol = POST_REL_TOL * max(max(abs(amounts[c]) for c in combo), 1.0)
            if abs(s) <= max(abs_tol, rel_tol):
                for c in combo:
                    final_cluster_map[c] = next_cid
                    assigned.add(c)
                next_cid += 1
    return

unique_labels = sorted(set(best_labels))
raw_clusters = {lbl: [i for i, lab in enumerate(best_labels) if lab == lbl] for lbl in unique_labels}
for lbl, members in raw_clusters.items():
    if lbl == -1:
        continue
    subset_balancing_search(members)

# small global pass for remaining unassigned (careful on size)
unassigned = [i for i in range(n) if final_cluster_map[i] == -1]
if 2 <= len(unassigned) <= 2000:
    amounts_un = {i: float(df.loc[i, "SignedAmount"]) for i in unassigned}
    for r in range(min(MAX_SUBSET_SIZE, len(unassigned)), 1, -1):
        for combo in combinations(unassigned, r):
            if any(final_cluster_map[c] != -1 for c in combo):
                continue
            s = sum(amounts_un[c] for c in combo)
            abs_tol = POST_ABS_TOL
            rel_tol = POST_REL_TOL * max(max(abs(amounts_un[c]) for c in combo), 1.0)
            if abs(s) <= max(abs_tol, rel_tol):
                for c in combo:
                    final_cluster_map[c] = next_cid
                next_cid += 1

df["FinalCluster"] = [final_cluster_map.get(i, -1) for i in range(n)]

# ----------------------------
# 11) Optional evaluation vs MatchGroupId (pairwise precision/recall/F1)
# ----------------------------
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
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    print(f"Evaluation -> precision: {prec:.4f}, recall: {rec:.4f}, f1: {f1:.4f}")

# ----------------------------
# 12) Save / output
# ----------------------------
print("Sample results:")
out_cols = ["DocumentDate", "SignedAmount", "raw_cluster", "FinalCluster"] + ([c + "_enc" for c in label_cols] if label_cols else [])
print(df[out_cols].head(50))
df.to_csv("autoencoder_metric_loss_dbscan.csv", index=False)
print("Done.")
