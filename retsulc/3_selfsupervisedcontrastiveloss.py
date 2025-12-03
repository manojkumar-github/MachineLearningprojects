"""
End-to-end pipeline implementing:
1) Self-supervised contrastive pretraining of a character-level encoder for ID-like fields
2) DBSCAN hyperparameter tuning (grid search using silhouette score on embeddings)
3) Graph construction + max-weight bipartite matching between CR and DR to enforce sum-to-zero
4) Small-subset search to handle multi-way (3-4 item) balanced groups left after pairing

Notes:
- This is unsupervised: MatchGroupId is NOT used for training, only optional evaluation if present.
- This code is self-contained, uses PyTorch + sklearn + scipy + networkx.
- Adjust hyperparameters (epochs, batch_size, DBSCAN grid) per your dataset size.
"""

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
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from scipy.optimize import linear_sum_assignment
import networkx as nx

# -----------------------
# CONFIG / HYPERPARAMS
# -----------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# Contrastive training
BATCH_SIZE = 128
EPOCHS = 30
LR = 1e-3
TEMPERATURE = 0.5
EMBED_DIM = 128
SEQ_LEN = 64         # per field
FIELDS = [
    "MerchantRefNum",
    "WebOrderNumber",
    "AcquireRefNumber",
    "PONumber",
    "TransactionRefNo",
    "CardNo",
    "AccountingDocNum"
]
MAX_SUBSET_SIZE = 4  # for multi-way balancing search
AMOUNT_TOL = 1e-6

# DBSCAN grid for tuning
DBSCAN_EPS = [0.3, 0.5, 0.7, 1.0]
DBSCAN_MIN_SAMPLES = [2, 3, 4]

# -----------------------
# 0) Example: load or construct df
# Replace with pd.read_csv("transactions.csv") for real use
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
# 1) Basic cleaning: drop fully-unknown columns as requested
# -----------------------
drop_cols = ["ReceiptNumber", "RefDocument", "Assignment", "StoreNumber", "AuthCode"]
for c in drop_cols:
    if c in df.columns:
        df = df.drop(columns=c)

# parse date
df["DocumentDate"] = pd.to_datetime(df["DocumentDate"], errors="coerce", dayfirst=True)

# signed amount
df["SignedAmount"] = df["Amount"].astype(float) * df["CR_DR"].map({"CR": 1.0, "DR": -1.0}).fillna(1.0)

# label-encode small categoricals (kept as additional numeric features)
label_cols = [c for c in ["DocType", "TransactionType", "Source", "SourceType"] if c in df.columns]
for col in label_cols:
    le = LabelEncoder()
    df[col + "_enc"] = le.fit_transform(df[col].astype(str))

# -----------------------
# 2) Build char vocabulary from chosen fields
# -----------------------
# Collect characters
all_text = ""
for f in FIELDS:
    if f in df.columns:
        all_text += " ".join(df[f].astype(str).tolist()) + " "
chars = sorted(set([c for c in all_text]))
# Keep ASCII-ish fallback: if no chars, fill digits+letters
if len(chars) == 0:
    chars = list("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_-/.")
char_to_idx = {c: i+1 for i, c in enumerate(chars)}  # reserve 0 for PAD
IDX_PAD = 0
VOCAB_SIZE = len(char_to_idx) + 1

def encode_text(s, max_len=SEQ_LEN):
    s = "" if s is None else str(s)
    idxs = [char_to_idx.get(ch, 0) for ch in s[:max_len]]
    if len(idxs) < max_len:
        idxs += [IDX_PAD] * (max_len - len(idxs))
    return idxs

# Build concatenated sequence per row (each field encoded separately and concatenated)
def build_row_seq(row):
    seq = []
    for f in FIELDS:
        if f in df.columns:
            seq += encode_text(row.get(f, ""), SEQ_LEN)
        else:
            seq += [IDX_PAD] * SEQ_LEN
    return seq  # length = SEQ_LEN * len(FIELDS)

SEQ_TOTAL = SEQ_LEN * len(FIELDS)
sequences = np.array([build_row_seq(r) for _, r in df.iterrows()], dtype=np.int64)

# -----------------------
# 3) Data augmentation functions for contrastive positive pairs
# -----------------------
import re
def augment_string(s):
    s = "" if s is None else str(s)
    # augmentation operations: random deletion, swap, mask, substring
    ops = []
    # random drop with prob
    if len(s) > 3 and random.random() < 0.2:
        i = random.randrange(len(s))
        s = s[:i] + s[i+1:]
    # random swap adjacent
    if len(s) > 3 and random.random() < 0.2:
        i = random.randrange(len(s)-1)
        lst = list(s)
        lst[i], lst[i+1] = lst[i+1], lst[i]
        s = "".join(lst)
    # random masking of a char
    if len(s) > 0 and random.random() < 0.2:
        i = random.randrange(len(s))
        s = s[:i] + "#" + s[i+1:]
    # substring crop
    if len(s) > 6 and random.random() < 0.15:
        a = random.randrange(0, len(s)//2)
        b = a + random.randrange(3, min(6, len(s)-a))
        s = s[:a] + s[b:]
    return s

def build_augmented_sequence(row):
    seq = []
    for f in FIELDS:
        if f in df.columns:
            aug = augment_string(row.get(f, ""))
            seq += encode_text(aug, SEQ_LEN)
        else:
            seq += [IDX_PAD] * SEQ_LEN
    return seq

# -----------------------
# 4) PyTorch dataset for contrastive pretraining (SimCLR-like NT-Xent)
# -----------------------
class ContrastiveDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x1 = np.array(build_row_seq(row), dtype=np.int64)
        x2 = np.array(build_augmented_sequence(row), dtype=np.int64)
        return x1, x2

dataset = ContrastiveDataset(df)
loader = DataLoader(dataset, batch_size=min(BATCH_SIZE, len(dataset)), shuffle=True)

# -----------------------
# 5) Encoder model (char-level CNN -> projection head)
# -----------------------
class CharEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, conv_channels=128, proj_dim=EMBED_DIM, seq_len=SEQ_TOTAL):
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

model = CharEncoder(VOCAB_SIZE, embed_dim=32, conv_channels=128, proj_dim=EMBED_DIM, seq_len=SEQ_TOTAL).to(DEVICE)

# -----------------------
# 6) NT-Xent contrastive loss
# -----------------------
def nt_xent_loss(z1, z2, temperature=TEMPERATURE):
    # z1,z2: (batch, dim) normalized
    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  # 2B x D
    sim = torch.matmul(z, z.T)  # 2B x 2B
    # mask out self-similarity
    mask = (~torch.eye(2*batch_size, 2*batch_size, dtype=torch.bool, device=DEVICE)).float()
    sim = sim / temperature
    # positives are (i,i+batch) and (i+batch,i)
    positives = torch.cat([torch.diag(sim, batch_size), torch.diag(sim, -batch_size)], dim=0)
    nom = torch.exp(positives)
    denom = mask * torch.exp(sim)
    denom = denom.sum(dim=1)
    loss = -torch.log(nom / denom)
    return loss.mean()

# -----------------------
# 7) Train contrastive encoder
# -----------------------
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

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
    if (epoch+1) % 5 == 0 or epoch==0:
        print(f"[Contrastive] Epoch {epoch+1}/{EPOCHS}, avg_loss={epoch_loss / max(1,n):.4f}")

# -----------------------
# 8) Produce final embeddings per row (deterministic forward)
# -----------------------
model.eval()
with torch.no_grad():
    seq_tensor = torch.tensor(sequences, dtype=torch.long, device=DEVICE)
    embeddings = model(seq_tensor).cpu().numpy()  # shape (n_rows, EMBED_DIM)

# -----------------------
# 9) Build final feature matrix (embeddings + amount + categorical encodings)
# -----------------------
num_feats = df[["SignedAmount"]].values.astype(float)
scaler = StandardScaler()
num_feats_scaled = scaler.fit_transform(num_feats)

cat_feats = df[[col + "_enc" for col in label_cols]].values if len(label_cols)>0 else np.zeros((len(df),0))

X = np.hstack([embeddings, num_feats_scaled, cat_feats])

# -----------------------
# 10) DBSCAN hyperparameter tuning (silhouette score on embedding features X)
# Keep best params and their labels
# -----------------------
best_score = -1.0
best_labels = None
best_params = None

# If dataset is too small for silhouette, fallback to a default DBSCAN run
if len(X) < 2:
    best_labels = np.array([-1]*len(X))
    best_params = (None, None)
else:
    for eps in DBSCAN_EPS:
        for ms in DBSCAN_MIN_SAMPLES:
            db = DBSCAN(eps=eps, min_samples=ms, metric="euclidean")
            labels = db.fit_predict(X)
            # silhouette requires at least 2 clusters and less than n clusters
            unique_labels = set(labels)
            n_clusters = len([l for l in unique_labels if l != -1])
            if n_clusters < 2:
                score = -1.0
            else:
                try:
                    score = silhouette_score(X, labels)
                except Exception:
                    score = -1.0
            if score > best_score:
                best_score = score
                best_labels = labels
                best_params = (eps, ms)

print("DBSCAN best params:", best_params, "best silhouette:", best_score)
df["raw_cluster"] = best_labels

# -----------------------
# 11) Graph + Max-Weight Bipartite Matching inside each raw cluster
# For each cluster: build bipartite matrix between CR and DR, weights = cosine similarity of embeddings
# Use linear_sum_assignment after padding to square matrix. Accept pairs only if their amounts cancel within tol.
# Then attempt multi-way subsets for leftover nodes up to MAX_SUBSET_SIZE.
# -----------------------
from sklearn.metrics.pairwise import cosine_similarity

n = len(df)
final_cluster_map = {i: -1 for i in range(n)}
next_cid = 0

def pairwise_match_in_cluster(member_indices):
    global next_cid
    # split by sign
    pos = [i for i in member_indices if df.loc[i, "SignedAmount"] > 0]
    neg = [i for i in member_indices if df.loc[i, "SignedAmount"] < 0]
    if (len(pos) == 0) or (len(neg) == 0):
        return []
    emb_pos = embeddings[pos]
    emb_neg = embeddings[neg]
    # similarity matrix (pos x neg)
    sim = cosine_similarity(emb_pos, emb_neg)
    # We will find maximum weight matching via linear_sum_assignment on negative cost (maximize sim)
    # Need square matrix: pad smaller dimension
    n1, n2 = sim.shape
    N = max(n1, n2)
    cost = np.ones((N, N)) * 1e6  # large cost for padding
    # convert to cost where lower is better; use -sim
    cost[:n1, :n2] = -sim
    row_ind, col_ind = linear_sum_assignment(cost)
    matches = []
    for r, c in zip(row_ind, col_ind):
        if r < n1 and c < n2:
            i = pos[r]; j = neg[c]
            # accept match if amounts approximately cancel within a tolerance relative to max(abs(amounts))
            if abs(df.loc[i, "SignedAmount"] + df.loc[j, "SignedAmount"]) <= max(AMOUNT_TOL, 0.01 * max(abs(df.loc[i,"SignedAmount"]), abs(df.loc[j,"SignedAmount"]), 1.0)):
                matches.append((i, j, sim[r, c]))
    # sort by descending sim, pick non-overlapping
    matches = sorted(matches, key=lambda x: -x[2])
    chosen = []
    used = set()
    for i, j, s in matches:
        if i in used or j in used:
            continue
        chosen.append((i,j))
        used.add(i); used.add(j)
    # assign cluster ids to chosen pairs
    for i, j in chosen:
        final_cluster_map[i] = next_cid
        final_cluster_map[j] = next_cid
        next_cid += 1
    # return leftover indices (in members) not yet assigned
    leftovers = [m for m in member_indices if final_cluster_map[m] == -1]
    return leftovers

def find_multiway_balanced(member_indices):
    """
    Brute-force search for subsets up to MAX_SUBSET_SIZE where sum of SignedAmount is ~0.
    Assign cluster ids greedily, preferring larger subsets.
    """
    global next_cid
    amounts = {i: df.loc[i,"SignedAmount"] for i in member_indices}
    assigned = set()
    # try sizes descending
    for size in range(min(MAX_SUBSET_SIZE, len(member_indices)), 2, -1):
        for combo in combinations(member_indices, size):
            if any(c in assigned for c in combo):
                continue
            s = sum(amounts[c] for c in combo)
            if abs(s) <= AMOUNT_TOL:
                # assign
                for c in combo:
                    final_cluster_map[c] = next_cid
                    assigned.add(c)
                next_cid += 1
    leftovers = [m for m in member_indices if final_cluster_map[m] == -1]
    return leftovers

df.reset_index(inplace=True)
# run per raw cluster
raw_clusters = {}
for lbl in sorted(set(best_labels)):
    members = [i for i, lab in enumerate(best_labels) if lab == lbl]
    raw_clusters[lbl] = members

for lbl, members in raw_clusters.items():
    if lbl == -1:
        # treat noise later
        continue
    # 1) pairwise matching using embeddings similarity + amount-cancel tolerance
    leftovers = pairwise_match_in_cluster(members)
    if len(leftovers) > 0:
        # 2) attempt multi-way balanced subsets among leftovers
        leftovers = find_multiway_balanced(leftovers)
    # leftovers remain unassigned

# Attempt to match across noise and unassigned rows: pairwise global matching between unassigned CR and DR
unassigned = [i for i in range(n) if final_cluster_map[i] == -1]
pos_un = [i for i in unassigned if df.loc[i,"SignedAmount"] > 0]
neg_un = [i for i in unassigned if df.loc[i,"SignedAmount"] < 0]
if len(pos_un) > 0 and len(neg_un) > 0:
    emb_pos = embeddings[pos_un]; emb_neg = embeddings[neg_un]
    sim = cosine_similarity(emb_pos, emb_neg)
    n1, n2 = sim.shape
    N = max(n1, n2)
    cost = np.ones((N,N)) * 1e6
    cost[:n1,:n2] = -sim
    row_ind, col_ind = linear_sum_assignment(cost)
    matches = []
    for r, c in zip(row_ind, col_ind):
        if r < n1 and c < n2:
            i = pos_un[r]; j = neg_un[c]
            if abs(df.loc[i,"SignedAmount"] + df.loc[j,"SignedAmount"]) <= max(AMOUNT_TOL, 0.01 * max(abs(df.loc[i,"SignedAmount"]), abs(df.loc[j,"SignedAmount"]), 1.0)):
                matches.append((i,j, sim[r,c]))
    matches = sorted(matches, key=lambda x: -x[2])
    used = set()
    for i,j,s in matches:
        if i in used or j in used:
            continue
        final_cluster_map[i] = next_cid
        final_cluster_map[j] = next_cid
        next_cid += 1
        used.add(i); used.add(j)

# Final residual multi-way pass on any remaining unassigned rows
remaining = [i for i in range(n) if final_cluster_map[i] == -1]
if remaining:
    leftovers = find_multiway_balanced(remaining)

# assign remaining unassigned rows to -1 (unclustered)
for i in range(n):
    if final_cluster_map[i] == -1:
        continue
    # already assigned

# write results to df
df["FinalCluster"] = [final_cluster_map[i] if final_cluster_map[i] is not None else -1 for i in range(n)]

# -----------------------
# 12) Optional evaluation against MatchGroupId (if present)
# -----------------------
if "MatchGroupId" in df.columns:
    # pairwise metrics
    true_pairs = set()
    pred_pairs = set()
    for i,j in combinations(range(n), 2):
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
# 13) Save / show results
# -----------------------
print("Final clusters assigned (sample):")
print(df[["SignedAmount", "FinalCluster"] + [c for c in df.columns if c.endswith("_enc")]])
df.to_csv("clusters_contrastive_dbscan_matching.csv", index=False)
