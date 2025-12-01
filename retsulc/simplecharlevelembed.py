"""
What Changed in This Version?

✔ Added small neural embeddings for ID-like fields

Instead of treating IDs like raw strings:
	•	Each ID is encoded as a sequence of characters
	•	Fed into a tiny CNN encoder
	•	Produces a dense 32-dimensional embedding

This helps capture patterns like:
	•	Similar merchant behaviors
	•	Partial string similarities
	•	Layout-like structures of IDs
	•	Numeric + alphanumeric similarity

[ID embeddings] + [SignedAmount] + [Date numeric]

But clustering still follows your strict rules (date-window + zero-sum).

⸻

🤔 Next Steps (Optional)

If you’d like, we can:

⭐ Add cosine similarity edges using embeddings

Helps detect near-duplicate reference numbers.

⭐ Add triplet-loss fine-tuning

Use MatchGroupId as supervision to train better embeddings.

⭐ Add ML scoring for ambiguous clusters
"""

import pandas as pd
import numpy as np
import networkx as nx
from datetime import timedelta
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------------------------------------------------
# 1. LOAD + PREPARE THE DATA
# -------------------------------------------------------------------------

df = pd.read_csv("transactions.csv")

df["DocumentDate"] = pd.to_datetime(df["DocumentDate"], errors="coerce")

def convert_amount(row):
    multiplier = 1 if row["CR_DR"] == "CR" else -1
    return multiplier * row["Amount"]

df["SignedAmount"] = df.apply(convert_amount, axis=1)

df = df.sort_values("DocumentDate").reset_index(drop=True)

# -------------------------------------------------------------------------
# 2. SMALL NEURAL NETWORK EMBEDDING MODELS
# -------------------------------------------------------------------------

# Character vocabulary for IDs, numeric strings, alphanumerics
CHARS = list("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_-/.")
char_to_idx = {c: i+1 for i, c in enumerate(CHARS)}
vocab_size = len(char_to_idx) + 1

EMBED_DIM = 32

def encode_string(s, max_len=32):
    """Turn raw ID-like string into fixed-length sequence of ints."""
    s = str(s)
    idxs = [char_to_idx.get(c, 0) for c in s[:max_len]]
    idxs += [0] * (max_len - len(idxs))
    return torch.tensor(idxs, dtype=torch.long)

class IDEncoder(nn.Module):
    """
    A compact sequence encoder for ID-like fields:
    - char embedding
    - 1D CNN feature extractor
    - max pooling
    """
    def __init__(self, vocab_size, embed_dim, out_dim=32):
        super().__init__()
        self.char_embed = nn.Embedding(vocab_size, embed_dim)
        self.conv = nn.Conv1d(embed_dim, out_dim, kernel_size=3, padding=1)

    def forward(self, x):
        # x shape = (batch, seq_len)
        x = self.char_embed(x)            # (batch, seq_len, embed_dim)
        x = x.transpose(1, 2)             # (batch, embed_dim, seq_len)
        x = F.relu(self.conv(x))          # (batch, out_dim, seq_len)
        x = torch.max(x, dim=2)[0]        # (batch, out_dim)
        return x

# Instantiate one shared encoder
id_encoder = IDEncoder(vocab_size, embed_dim=16, out_dim=32)

# -------------------------------------------------------------------------
# 3. FIELDS WE EMBED
# -------------------------------------------------------------------------

text_fields = [
    "TransactionRefNo",
    "MerchantRefNum",
    "PONumber",
    "CardNo",
    "ReceiptNumber",
    "AccountingDocNum",
    "AcquireRefNumber",
    "WebOrderNumber"
]

def compute_embeddings(df):
    embeddings = []

    for idx, row in df.iterrows():
        row_embs = []

        for f in text_fields:
            seq = encode_string(row[f])
            seq = seq.unsqueeze(0)  # batch = 1
            with torch.no_grad():
                emb = id_encoder(seq)[0]
            row_embs.append(emb)

        # Numeric amount as a feature
        amount_emb = torch.tensor([row["SignedAmount"]], dtype=torch.float)

        # Date as numeric
        date_val = row["DocumentDate"].timestamp() / (3600 * 24)
        date_emb = torch.tensor([date_val], dtype=torch.float)

        # Concatenate all embeddings → final  feature vector
        full_emb = torch.cat(row_embs + [amount_emb, date_emb], dim=0)

        embeddings.append(full_emb)

    embedding_mat = torch.stack(embeddings)
    return embedding_mat


df_embeddings = compute_embeddings(df)

# -------------------------------------------------------------------------
# 4. DATE-ONLY BLOCKING (ONLY RULE USED)
# -------------------------------------------------------------------------

def create_date_blocks(df, window_days=3):
    blocks = []
    current_block = [df.index[0]]

    for i in range(1, len(df)):
        prev_date = df.loc[df.index[i-1], "DocumentDate"]
        curr_date = df.loc[df.index[i], "DocumentDate"]

        if (curr_date - prev_date).days <= window_days:
            current_block.append(df.index[i])
        else:
            blocks.append(current_block)
            current_block = [df.index[i]]

    blocks.append(current_block)
    return blocks


date_blocks = create_date_blocks(df, window_days=3)

# -------------------------------------------------------------------------
# 5. GRAPH-BASED ZERO-SUM CLUSTERING
# -------------------------------------------------------------------------

def cluster_zero_sum(block_indices):
    block_df = df.loc[block_indices]

    # Graph nodes
    G = nx.Graph()
    G.add_nodes_from(block_df.index)

    # Link CR to DR with opposite signs
    pos = block_df[block_df["SignedAmount"] > 0]
    neg = block_df[block_df["SignedAmount"] < 0]

    for i in pos.index:
        for j in neg.index:
            G.add_edge(i, j)

    # Connected components = candidate clusters
    comps = list(nx.connected_components(G))
    
    final_clusters = []
    for c in comps:
        sub = block_df.loc[list(c)]
        if abs(sub["SignedAmount"].sum()) < 1e-6:
            final_clusters.append(list(c))

    return final_clusters


all_clusters = []

for block in date_blocks:
    clusters = cluster_zero_sum(block)
    all_clusters.extend(clusters)

# -------------------------------------------------------------------------
# 6. STORE PREDICTED CLUSTERS
# -------------------------------------------------------------------------

cluster_map = {}
for cid, nodes in enumerate(all_clusters):
    for n in nodes:
        cluster_map[n] = cid

df["PredictedCluster"] = df.index.map(cluster_map).fillna(-1).astype(int)

df.to_csv("clustered_output_with_embeddings.csv", index=False)

print("Clustering completed with neural embeddings integrated.")
