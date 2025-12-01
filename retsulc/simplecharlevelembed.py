"""
Step 1 — Data preparation
	•	Clean date, numeric fields
	•	Prepare string IDs

Step 2 — Embedding Model

A small character-level CNN encoder + numeric features → 64-dim vector

Step 3 — Contrastive Learning
	•	Positive pairs = same MatchGroupId
	•	Negative pairs = different MatchGroupId
	•	Loss = Contrastive Loss
L = y \cdot D^2 + (1-y) \cdot \max(0, margin - D)^2

Step 4 — Use learned embeddings for clustering
	•	Block by date window
	•	Build graph with:
	•	Edges for CR ↔ DR
	•	Extra edges where embedding similarity > threshold
	•	Keep only clusters where sum(amount)=0

✔ Embeddings now meaningfully influence clustering

Used through cosine similarity edges in graph.

✔ Contrastive loss trains embeddings to reflect real cluster structure

Positive = same MatchGroupId
Negative = different groups

✔ Better matching for ambiguous cases

Transactions with similar:
	•	IDs
	•	Merchant refs
	•	Accounting doc numbers
	•	Reference numbers
	•	Dates
	•	Amount patterns

get closer in embedding space.

✔ Still respects your strict business rules
	•	Date-window block
	•	Amount-sum-to-zero constraint

ML enhances clustering but does not replace rules.

"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import networkx as nx
from datetime import timedelta
from sklearn.model_selection import train_test_split


# ============================================================================
# 1. LOAD & PREP DATA
# ============================================================================

df = pd.read_csv("transactions.csv")
df["DocumentDate"] = pd.to_datetime(df["DocumentDate"], errors="coerce")

# Signed amount CR(+), DR(-)
def convert_amount(row):
    return row["Amount"] if row["CR_DR"] == "CR" else -row["Amount"]

df["SignedAmount"] = df.apply(convert_amount, axis=1)

df = df.sort_values("DocumentDate").reset_index(drop=True)


# ============================================================================
# 2. ENCODER MODEL: Character-level ID embedding + numeric features
# ============================================================================

CHARS = list("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_-/.")
char_to_idx = {c: i + 1 for i, c in enumerate(CHARS)}
vocab_size = len(char_to_idx) + 1
SEQ_LEN = 32

def encode_string(s, max_len=SEQ_LEN):
    s = str(s)
    idxs = [char_to_idx.get(c, 0) for c in s[:max_len]]
    idxs += [0] * (max_len - len(idxs))
    return torch.tensor(idxs, dtype=torch.long)


ID_FIELDS = [
    "TransactionRefNo",
    "MerchantRefNum",
    "PONumber",
    "CardNo",
    "ReceiptNumber",
    "AccountingDocNum",
    "AcquireRefNumber",
    "WebOrderNumber"
]


class IDEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.char_embed = nn.Embedding(vocab_size, 16)

        # CNN encoder
        self.conv = nn.Conv1d(16, 32, kernel_size=3, padding=1)

        # Combine all ID embeddings + numeric
        self.fc = nn.Linear(32 * len(ID_FIELDS) + 2, 64)

    def encode_single_field(self, seq):
        x = self.char_embed(seq.unsqueeze(0))
        x = x.transpose(1, 2)
        x = F.relu(self.conv(x))
        x = x.max(dim=2)[0]
        return x.squeeze(0)

    def forward(self, row):
        id_embs = []

        for f in ID_FIELDS:
            seq = encode_string(row[f])
            id_embs.append(self.encode_single_field(seq))

        amount = torch.tensor([row["SignedAmount"]], dtype=torch.float)
        date_val = torch.tensor([row["DocumentDate"].timestamp() / (3600 * 24)], dtype=torch.float)

        x = torch.cat(id_embs + [amount, date_val], dim=0)
        return self.fc(x)


model = IDEncoder()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


# ============================================================================
# 3. CONTRASTIVE LOSS (Metric Learning)
# ============================================================================

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, emb1, emb2, label):
        dist = F.pairwise_distance(emb1, emb2)
        loss = label * dist.pow(2) + (1 - label) * torch.clamp(self.margin - dist, min=0).pow(2)
        return loss.mean()


criterion = ContrastiveLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# ============================================================================
# 4. TRAINING DATA: Build positive & negative pairs using MatchGroupId
# ============================================================================

pairs = []
labels = []

# Positive pairs
for g in df["MatchGroupId"].unique():
    idxs = df.index[df["MatchGroupId"] == g].tolist()
    if len(idxs) > 1:
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                pairs.append((idxs[i], idxs[j]))
                labels.append(1)

# Negative pairs (random)
all_indices = df.index.tolist()
np.random.shuffle(all_indices)
for i in range(len(pairs)):
    a = np.random.choice(all_indices)
    b = np.random.choice(all_indices)
    if df.loc[a, "MatchGroupId"] != df.loc[b, "MatchGroupId"]:
        pairs.append((a, b))
        labels.append(0)

# Train / Test split
(train_pairs, test_pairs, train_labels, test_labels) = train_test_split(
    pairs, labels, test_size=0.2, random_state=42
)


# ============================================================================
# 5. TRAIN METRIC EMBEDDINGS
# ============================================================================

def get_embedding(idx):
    row = df.loc[idx]
    with torch.no_grad():
        return model(row).to(device)

def train_metric_model(epochs=3):
    model.train()
    for ep in range(epochs):
        total_loss = 0
        for (i, j), lbl in zip(train_pairs, train_labels):
            optimizer.zero_grad()

            emb_i = model(df.loc[i].to_dict())
            emb_j = model(df.loc[j].to_dict())

            lbl_t = torch.tensor([lbl], dtype=torch.float, device=device)

            loss = criterion(emb_i, emb_j, lbl_t)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {ep+1}: Loss={total_loss:.4f}")

train_metric_model()


# ============================================================================
# 6. USE EMBEDDINGS FOR SIMILARITY-BASED GRAPH CLUSTERING
# ============================================================================

def cosine_sim(v1, v2):
    return F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0))[0].item()


def create_date_blocks(df, window_days=3):
    blocks = []
    current = [df.index[0]]

    for i in range(1, len(df)):
        if (df.loc[i, "DocumentDate"] - df.loc[i-1, "DocumentDate"]).days <= window_days:
            current.append(df.index[i])
        else:
            blocks.append(current)
            current = [df.index[i]]
    blocks.append(current)
    return blocks


def cluster_block(block, sim_threshold=0.6):
    G = nx.Graph()
    G.add_nodes_from(block)

    # Precompute embeddings
    emb_map = {idx: get_embedding(idx) for idx in block}

    # Rule 1: CR ↔ DR edges
    pos = [i for i in block if df.loc[i, "SignedAmount"] > 0]
    neg = [i for i in block if df.loc[i, "SignedAmount"] < 0]
    for i in pos:
        for j in neg:
            G.add_edge(i, j)

    # Rule 2: Embedding-based similarity
    for i in block:
        for j in block:
            if i < j:
                sim = cosine_sim(emb_map[i], emb_map[j])
                if sim >= sim_threshold:
                    G.add_edge(i, j)

    # Find connected components
    comps = nx.connected_components(G)

    # Keep only components that sum to zero
    valid = []
    for comp in comps:
        comp = list(comp)
        if abs(df.loc[comp]["SignedAmount"].sum()) < 1e-6:
            valid.append(comp)
    return valid


all_clusters = []
blocks = create_date_blocks(df)

for block in blocks:
    all_clusters.extend(cluster_block(block))


# ============================================================================
# 7. SAVE CLUSTER RESULTS
# ============================================================================

cluster_map = {}
for cid, nodes in enumerate(all_clusters):
    for n in nodes:
        cluster_map[n] = cid

df["PredictedCluster"] = df.index.map(cluster_map).fillna(-1).astype(int)
df.to_csv("final_cluster_output.csv", index=False)

print("Clustering completed with contrastive-learning embeddings.")
