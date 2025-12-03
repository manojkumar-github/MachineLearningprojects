"""
End-to-end pipeline:
- Load data (single-row example or CSV)
- Drop fully-unknown columns
- Prepare character vocabulary
- Build a deep neural char-level autoencoder (encoder produces embeddings)
- Train autoencoder unsupervised on concatenated ID-like fields
- Produce per-row embeddings from encoder
- Combine embeddings with encoded categorical numeric features
- Run DBSCAN on final feature vectors
- Post-process clusters to enforce sum(SignedAmount)=0 by searching balanced subsets
- Output predicted clusters and (if available) evaluation against MatchGroupId

Notes:
- This is unsupervised: MatchGroupId is NOT used for training, only optional evaluation.
- The autoencoder here is simple and intended as a lightweight, practical embedding trainer.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import DBSCAN
from itertools import combinations
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torch.nn.functional as F

# ---------------------------
# 1) Example data / load CSV
# ---------------------------
# Replace this example by reading your CSV if available:
# df = pd.read_csv("transactions.csv")
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

# ---------------------------
# 2) Drop columns that are all 'unknown'
# ---------------------------
drop_cols = ["ReceiptNumber", "RefDocument", "Assignment", "StoreNumber", "AuthCode"]
for c in drop_cols:
    if c in df.columns:
        df = df.drop(columns=c)

# ---------------------------
# 3) Basic preprocessing
# ---------------------------
# DocumentDate parse (kept but not used in blocking per request)
df["DocumentDate"] = pd.to_datetime(df["DocumentDate"], errors="coerce", dayfirst=True)

# Signed amount (CR -> +, DR -> -)
df["CR_DR"] = df["CR_DR"].fillna("CR")
df["SignedAmount"] = df["Amount"].astype(float) * df["CR_DR"].map({"CR": 1.0, "DR": -1.0}).fillna(1.0)

# Label encode small categorical fields (DocType, TransactionType, Source, SourceType)
label_cols = [c for c in ["DocType", "TransactionType", "Source", "SourceType"] if c in df.columns]
label_encoders = {}
for col in label_cols:
    le = LabelEncoder()
    df[col] = df[col].fillna("unknown").astype(str)
    df[col + "_enc"] = le.fit_transform(df[col])
    label_encoders[col] = le

cat_feature_cols = [col + "_enc" for col in label_cols]

# ---------------------------
# 4) Char-level vocabulary & sequence preparation
# ---------------------------

# fields to use in the char-level encoder (ID-like fields)
char_fields = [
    "MerchantRefNum",
    "WebOrderNumber",
    "AcquireRefNumber",
    "PONumber",
    "TransactionRefNo",
    "CardNo",
    "AccountingDocNum"
]
# Keep only fields that exist in df
char_fields = [f for f in char_fields if f in df.columns]

# Build character vocabulary from dataset (limit to ASCII subset)
#all_text = " ".join(df[f].astype(str).tolist() for f in char_fields)
final = []
for f in char_fields:
  df[f].astype(str).tolist()
  final.extend(df[f].astype(str).tolist())
all_text = " ".join(final)

chars = sorted(list(set([c for c in all_text])))
# Ensure deterministic ordering and reserve index 0 for padding
char_to_idx = {c: i+1 for i, c in enumerate(chars)}
idx_to_char = {i: c for c, i in char_to_idx.items()}
vocab_size = len(char_to_idx) + 1

# Sequence length: choose a reasonable length to cover most IDs
SEQ_LEN = 64

def encode_field(text, max_len=SEQ_LEN):
    s = "" if text is None else str(text)
    seq = [char_to_idx.get(c, 0) for c in s[:max_len]]
    if len(seq) < max_len:
        seq += [0] * (max_len - len(seq))
    return seq

# Build concatenated sequence per row: we will join fields with a special separator index (use 0)
def build_row_sequence(row):
    parts = []
    for f in char_fields:
        parts += encode_field(row.get(f, ""), max_len=SEQ_LEN)
    # Final length = len(char_fields) * SEQ_LEN
    return parts

# Create dataset sequences
seq_len_total = len(char_fields) * SEQ_LEN
sequences = np.vstack([build_row_sequence(row) for _, row in df.iterrows()])

# ---------------------------
# 5) PyTorch Dataset & DataLoader
# ---------------------------
class IDDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
    def __len__(self):
        return self.sequences.shape[0]
    def __getitem__(self, idx):
        return self.sequences[idx]

dataset = IDDataset(sequences)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# ---------------------------
# 6) Deep Autoencoder model (encoder -> embedding vector)
#    - char embedding -> 1D convs -> pooling -> linear -> embedding
#    - decoder mirrors encoder to reconstruct sequences (simple)
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CharAutoencoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, conv_channels=128, bottleneck_dim=128, seq_len_total=seq_len_total):
        super().__init__()
        self.char_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # conv encoder
        self.conv1 = nn.Conv1d(embed_dim, conv_channels, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(conv_channels, conv_channels, kernel_size=5, padding=2)
        self.pool = nn.AdaptiveMaxPool1d(1)  # global pooling
        self.bottleneck = nn.Linear(conv_channels, bottleneck_dim)
        # decoder: expand bottleneck and use transposed convs to reconstruct sequence embeddings
        self.decoder_fc = nn.Linear(bottleneck_dim, conv_channels * seq_len_total // 8)
        # We'll map back to char logits per position using a linear layer after reshaping
        self.out_proj = nn.Linear(conv_channels, vocab_size)

        self.seq_len_total = seq_len_total
        self.conv_channels = conv_channels

    def encode(self, x):
        # x: (batch, seq_len_total)
        x = self.char_embed(x)                  # (batch, seq_len, embed_dim)
        x = x.transpose(1, 2)                   # (batch, embed_dim, seq_len)
        x = F.relu(self.conv1(x))               # (batch, conv_ch, seq_len)
        x = F.relu(self.conv2(x))               # (batch, conv_ch, seq_len)
        x = self.pool(x).squeeze(-1)            # (batch, conv_ch)
        z = self.bottleneck(x)                  # (batch, bottleneck_dim)
        return z

    def decode(self, z):
        # z: (batch, bottleneck_dim)
        # produce a "feature map" we can project to per-char logits
        b = z.size(0)
        # expand to size (batch, conv_ch, seq_len) via linear then reshape
        expanded = self.decoder_fc(z)  # (batch, conv_ch * seq_len//8)
        # compute approximate seq_len by repeating; this is simple and works for small datasets
        # reshape
        target_len = self.seq_len_total
        conv_ch = self.conv_channels
        # reshape to (batch, conv_ch, target_len//8)
        reshaped = expanded.view(b, conv_ch, target_len // 8)
        # upsample to target_len via interpolation
        upsampled = F.interpolate(reshaped, size=target_len, mode='linear', align_corners=False)
        # now upsampled: (batch, conv_ch, target_len)
        # project per-position to vocab logits
        logits = self.out_proj(upsampled.transpose(1, 2))  # (batch, target_len, vocab_size)
        return logits

    def forward(self, x):
        z = self.encode(x)
        logits = self.decode(z)
        return logits, z

# instantiate model
model = CharAutoencoder(vocab_size=vocab_size, embed_dim=32, conv_channels=64, bottleneck_dim=128, seq_len_total=seq_len_total)
model = model.to(device)

# ---------------------------
# 7) Training loop for autoencoder
# ---------------------------
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore padding index

EPOCHS = 30  # modest number; increase as needed for larger datasets

model.train()
for epoch in range(EPOCHS):
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)  # (batch, seq_len)
        optimizer.zero_grad()
        logits, _ = model(batch)  # logits: (batch, seq_len, vocab_size)
        # For CrossEntropyLoss, target should be (batch, seq_len) and logits (batch*seq_len, vocab)
        b, L, V = logits.shape
        logits_flat = logits.view(b * L, V)
        target_flat = batch.view(b * L)
        loss = criterion(logits_flat, target_flat)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * b
    avg_loss = total_loss / len(dataset)
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"[Autoencoder] Epoch {epoch+1}/{EPOCHS}  AvgLoss={avg_loss:.4f}")

# ---------------------------
# 8) Produce embeddings for each row (encoder outputs)
# ---------------------------
model.eval()
with torch.no_grad():
    seq_tensor = torch.tensor(sequences, dtype=torch.long, device=device)
    _, embeddings = model(seq_tensor)
    # embeddings: (n_rows, bottleneck_dim)
    embeddings_np = embeddings.cpu().numpy()

# ---------------------------
# 9) Combine embeddings with categorical/numeric features
# ---------------------------
# Standardize signed amount
num_feats = np.vstack([
    df["SignedAmount"].astype(float).values
]).T  # shape (n,1)

scaler = StandardScaler()
num_feats_scaled = scaler.fit_transform(num_feats)

# categorical encoded features
cat_feats = df[cat_feature_cols].values if len(cat_feature_cols) > 0 else np.zeros((len(df), 0))

# final feature matrix: [embedding | numeric | categories]
X = np.hstack([embeddings_np, num_feats_scaled, cat_feats])

# ---------------------------
# 10) DBSCAN clustering on combined features
# ---------------------------
# choose eps based on scale; with embeddings and scaled numeric, a default may work but tuning is recommended
db = DBSCAN(eps=1.0, min_samples=2, metric='euclidean', n_jobs=1)
cluster_labels = db.fit_predict(X)
df["RawCluster"] = cluster_labels

print("Raw DBSCAN label distribution:", np.unique(cluster_labels, return_counts=True))

# ---------------------------
# 11) Post-processing: enforce sum(SignedAmount) = 0 for clusters
# ---------------------------
def find_balanced_subsets(amounts, max_subset_size=6, tol=1e-6):
    """
    Find subsets of indices whose sum is within tolerance of zero.
    Brute force up to subset size to keep compute reasonable.
    Returns list of tuples of indices (relative to amounts array).
    """
    n = len(amounts)
    results = []
    indices = list(range(n))
    # Try sizes from 2 up to max_subset_size or n
    upper = min(max_subset_size, n)
    for r in range(2, upper + 1):
        for combo in combinations(indices, r):
            if abs(sum(amounts[i] for i in combo)) <= tol:
                results.append(combo)
    return results

# Build final clusters by scanning each DBSCAN cluster and extracting balanced subsets.
final_cluster_map = {}  # row_index -> final_cluster_id
next_cid = 0

unique_clusters = sorted(set(cluster_labels))
for cl in unique_clusters:
    if cl == -1:
        # noise; consider attempting to find pairs/triples across noise later
        continue
    members = np.where(cluster_labels == cl)[0].tolist()
    if len(members) == 0:
        continue
    amounts = [df.iloc[i]["SignedAmount"] for i in members]
    balanced = find_balanced_subsets(amounts, max_subset_size=min(6, len(members)))
    assigned = set()
    # Greedy assign non-overlapping found balanced subsets, prefer larger subsets
    balanced = sorted(balanced, key=lambda x: (-len(x), x))
    for subset in balanced:
        # map relative indices to original row indices
        rows = [members[i] for i in subset]
        # skip if any row already assigned
        if any(r in assigned for r in rows):
            continue
        for r in rows:
            final_cluster_map[r] = next_cid
            assigned.add(r)
        next_cid += 1
    # leftover members not assigned: try pairwise matching (best-effort)
    leftover = [m for m in members if m not in assigned]
    # try simple pairwise combos
    for a, b in combinations(leftover, 2):
        if abs(df.iloc[a]["SignedAmount"] + df.iloc[b]["SignedAmount"]) <= 1e-6:
            final_cluster_map[a] = next_cid
            final_cluster_map[b] = next_cid
            next_cid += 1
            assigned.update([a, b])
    # any still leftover -> mark as -1 (unclustered)
    for r in leftover:
        if r not in assigned:
            final_cluster_map[r] = -1

# Optionally handle noise points (cluster -1 from DBSCAN): attempt pairwise matching across all unassigned rows
unassigned = [i for i in range(len(df)) if final_cluster_map.get(i, None) is None or final_cluster_map.get(i) == -1]
# attempt pairwise across unassigned to find zero-sum pairs
for a, b in combinations(unassigned, 2):
    if final_cluster_map.get(a, None) is None or final_cluster_map.get(a) == -1:
        if final_cluster_map.get(b, None) is None or final_cluster_map.get(b) == -1:
            if abs(df.iloc[a]["SignedAmount"] + df.iloc[b]["SignedAmount"]) <= 1e-6:
                final_cluster_map[a] = next_cid
                final_cluster_map[b] = next_cid
                next_cid += 1

# assign remaining unassigned to -1
for i in range(len(df)):
    cid = final_cluster_map.get(i, -1)
    df.at[df.index[i], "FinalCluster"] = cid

# ---------------------------
# 12) Optional evaluation if MatchGroupId exists
# ---------------------------
if "MatchGroupId" in df.columns:
    # compute simple pairwise precision/recall/f1
    true_pairs = set()
    pred_pairs = set()
    for i, j in combinations(range(len(df)), 2):
        same_true = (str(df.iloc[i]["MatchGroupId"]) == str(df.iloc[j]["MatchGroupId"]))
        same_pred = (df.iloc[i]["FinalCluster"] == df.iloc[j]["FinalCluster"]) and (df.iloc[i]["FinalCluster"] != -1)
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

# ---------------------------
# 13) Output results
# ---------------------------
print("\nRows with final clusters and key fields:")
out_cols = ["FinalCluster", "SignedAmount"] + [c for c in df.columns if c not in ["FinalCluster"]]
print(df[["FinalCluster", "SignedAmount"] + cat_feature_cols + char_fields].head(50))

# Save to CSV if desired
df.to_csv("clusters_with_deep_embeddings.csv", index=False)
