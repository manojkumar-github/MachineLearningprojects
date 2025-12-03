"""
Full pipeline with HDBSCAN clustering + contrastive char-encoder embeddings
"""

import os
import sys
import random
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
from scipy.optimize import linear_sum_assignment

# Try to import hdbscan and install if missing
try:
    import hdbscan
except Exception:
    print("hdbscan not found, attempting to install...")
    try:
        os.system(f"{sys.executable} -m pip install hdbscan --quiet")
        import hdbscan
    except Exception as e:
        raise RuntimeError("Failed to import or install hdbscan: " + str(e))

# -----------------------
# CONFIG
# -----------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# contrastive training hyperparams
BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-3
TEMPERATURE = 0.5
EMBED_DIM = 128
SEQ_LEN = 64          # per-field character sequence length
FIELDS = [
    "MerchantRefNum",
    "WebOrderNumber",
    "AcquireRefNumber",
    "PONumber",
    "TransactionRefNo",
    "CardNo",
    "AccountingDocNum"
]
MAX_SUBSET_SIZE = 4
AMOUNT_TOL = 1e-6

# HDBSCAN params
HDB_MIN_CLUSTER_SIZE = 5
HDB_MIN_SAMPLES = 3
HDB_METRIC = "euclidean"

# -----------------------
# 0) Example input (replace with pd.read_csv for real use)
# -----------------------
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
# 1) Clean & preparatory steps
# -----------------------
# Drop fully-unknown columns
drop_cols = ["ReceiptNumber", "RefDocument", "Assignment", "StoreNumber", "AuthCode"]
for c in drop_cols:
    if c in df.columns:
        df = df.drop(columns=c)

# parse date (kept for possible later use)
df["DocumentDate"] = pd.to_datetime(df["DocumentDate"], errors="coerce", dayfirst=True)

# Signed amount: CR -> +, DR -> -
df["SignedAmount"] = df["Amount"].astype(float) * df["CR_DR"].map({"CR": 1.0, "DR": -1.0}).fillna(1.0)

# label-encode small categoricals
label_cols = [c for c in ["DocType", "TransactionType", "Source", "SourceType"] if c in df.columns]
for col in label_cols:
    le = LabelEncoder()
    df[col + "_enc"] = le.fit_transform(df[col].astype(str))

# -----------------------
# 2) Character vocabulary & encoding
# -----------------------
# Build character set from data in chosen fields
all_text = ""
for f in FIELDS:
    if f in df.columns:
        all_text += " ".join(df[f].astype(str).tolist()) + " "
chars = sorted(set(all_text))
if len(chars) == 0:
    # fallback to common ASCII set
    chars = list("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_-/.")
char_to_idx = {c: i+1 for i, c in enumerate(chars)}  # 0 reserved for PAD
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
# 3) Augmentation for contrastive learning
# -----------------------
def augment_string(s):
    s = "" if s is None else str(s)
    # random deletion
    if len(s) > 2 and random.random() < 0.2:
        i = random.randrange(len(s))
        s = s[:i] + s[i+1:]
    # adjacent swap
    if len(s) > 2 and random.random() < 0.2:
        i = random.randrange(len(s)-1)
        lst = list(s)
        lst[i], lst[i+1] = lst[i+1], lst[i]
        s = "".join(lst)
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
# 4) Dataset + DataLoader for contrastive training
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
# 5) Encoder model (char-level CNN + projection head)
# -----------------------
class CharEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, conv_channels=128, proj_dim=EMBED_DIM):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=IDX_PAD)
        self.conv1 = nn.Conv1d(embed_dim, conv_channels, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(conv_channels, conv_channels, kernel_size=5, padding=2)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.proj = nn.Sequential(
            nn.Linear(conv_channels, conv_channels),
            nn.ReLU(),
            nn.Linear(conv_channels, proj_dim)
        )
    def forward(self, x):  # x: (batch, seq_total)
        x = self.embed(x)            # (batch, seq, embed_dim)
        x = x.transpose(1,2)         # (batch, embed_dim, seq)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1) # (batch, conv_ch)
        z = self.proj(x)             # (batch, proj_dim)
        z = F.normalize(z, p=2, dim=1)
        return z

model = CharEncoder(VOCAB_SIZE, embed_dim=32, conv_channels=128, proj_dim=EMBED_DIM).to(DEVICE)

# -----------------------
# 6) NT-Xent contrastive loss function
# -----------------------
def nt_xent_loss(z1, z2, temperature=TEMPERATURE):
    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  # 2B x D
    sim = torch.matmul(z, z.T)      # 2B x 2B
    sim = sim / temperature
    # mask self
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
        x1 = x1_np.to(DEVICE)
        x2 = x2_np.to(DEVICE)
        optimizer.zero_grad()
        z1 = model(x1)
        z2 = model(x2)
        loss = nt_xent_loss(z1, z2, TEMPERATURE)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * x1.size(0)
        n += x1.size(0)
    if (epoch+1) % 5 == 0 or epoch == 0:
        print(f"[Contrastive] Epoch {epoch+1}/{EPOCHS}, avg_loss={epoch_loss / max(1,n):.4f}")

# -----------------------
# 8) Produce embeddings for all rows
# -----------------------
model.eval()
with torch.no_grad():
    seq_tensor = torch.tensor(sequences, dtype=torch.long, device=DEVICE)
    embeddings = model(seq_tensor).cpu().numpy()  # shape (n_rows, EMBED_DIM)

# -----------------------
# 9) Build final feature matrix: embeddings + SignedAmount (scaled) + encoded categories
# -----------------------
num_feats = df[["SignedAmount"]].values.astype(float)
scaler = StandardScaler()
num_feats_scaled = scaler.fit_transform(num_feats)

cat_feats = df[[col + "_enc" for col in label_cols]].values if len(label_cols) > 0 else np.zeros((len(df), 0))

X = np.hstack([embeddings, num_feats_scaled, cat_feats])

# -----------------------
# 10) HDBSCAN clustering on X
# -----------------------
clusterer = hdbscan.HDBSCAN(min_cluster_size=HDB_MIN_CLUSTER_SIZE,
                            min_samples=HDB_MIN_SAMPLES,
                            metric=HDB_METRIC,
                            cluster_selection_method="eom")
labels = clusterer.fit_predict(X)
df["raw_cluster"] = labels
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(f"HDBSCAN produced {n_clusters} clusters (noise label = -1)")

# -----------------------
# 11) Graph + Max-Weight Matching per cluster + multi-way balancing
# -----------------------
n = len(df)
final_cluster_map = {i: -1 for i in range(n)}
next_cid = 0

def pairwise_match_in_cluster(member_indices):
    global next_cid
    pos = [i for i in member_indices if df.loc[i, "SignedAmount"] > 0]
    neg = [i for i in member_indices if df.loc[i, "SignedAmount"] < 0]
    if not pos or not neg:
        return member_indices  # nothing matched, return all as leftover
    emb_pos = embeddings[pos]
    emb_neg = embeddings[neg]
    sim = cosine_similarity(emb_pos, emb_neg)  # pos x neg
    n1, n2 = sim.shape
    N = max(n1, n2)
    BIG = 1e6
    cost = np.ones((N, N)) * BIG
    cost[:n1, :n2] = -sim  # maximize similarity
    row_ind, col_ind = linear_sum_assignment(cost)
    matches = []
    for r, c in zip(row_ind, col_ind):
        if r < n1 and c < n2:
            i = pos[r]; j = neg[c]
            # accept match if amounts cancel within tolerance (or relative tolerance)
            tol = max(AMOUNT_TOL, 0.01 * max(abs(df.loc[i,"SignedAmount"]), abs(df.loc[j,"SignedAmount"]), 1.0))
            if abs(df.loc[i,"SignedAmount"] + df.loc[j,"SignedAmount"]) <= tol:
                matches.append((i, j, sim[r, c]))
    # pick non-overlapping matches ordered by similarity
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

# process each non-noise raw cluster
unique_labels = sorted(set(labels))
raw_clusters = {lbl: [i for i, lab in enumerate(labels) if lab == lbl] for lbl in unique_labels}
for lbl, members in raw_clusters.items():
    if lbl == -1:
        continue
    leftovers = pairwise_match_in_cluster(members)
    if len(leftovers) > 0:
        leftovers = find_multiway_balanced(leftovers)

# attempt global matching for unassigned rows (including noise)
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
            tol = max(AMOUNT_TOL, 0.01 * max(abs(df.loc[i,"SignedAmount"]), abs(df.loc[j,"SignedAmount"]), 1.0))
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

# final multi-way pass for any remaining unassigned
remaining = [i for i in range(n) if final_cluster_map[i] == -1]
if remaining:
    _ = find_multiway_balanced(remaining)

# write results to df
df["FinalCluster"] = [final_cluster_map.get(i, -1) for i in range(n)]

# -----------------------
# 12) Optional evaluation (pairwise) if MatchGroupId present
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
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    print(f"Evaluation -> precision: {prec:.4f}, recall: {rec:.4f}, f1: {f1:.4f}")

# -----------------------
# 13) Output / save
# -----------------------
print("Sample output:")
cols_show = ["SignedAmount", "raw_cluster", "FinalCluster"] + [c for c in df.columns if c.endswith("_enc")]
print(df[cols_show].head(50))
df.to_csv("clusters_hdbscan_contrastive.csv", index=False)
