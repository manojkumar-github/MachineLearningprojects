import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import DBSCAN
from itertools import combinations

# -------------------------------------------------------------------
# 1. LOAD YOUR DATA (replace with your actual dataframe)
# -------------------------------------------------------------------

df = pd.DataFrame([
    {
        "DocumentDate": "02/01/2025",
        "DocType": "unknown",
        "TransactionType": "SAP",
        "BankTrfRef": "unknown",
        "Amount": 7800,
        "TransactionRefNo": "W1dsfsafjdjfb",
        "MerchantRefNum": "777741344598",
        "CR_DR": "CR",
        "GLRecordID": "unknown",
        "OID": "unknown",
        "PONumber": "13u350u5u05",
        "CardNo": "13415555531535",
        "ReceiptNumber": "unknown",
        "AccountingDocNum": "KD0nfdkk",
        "AuthCode": "unknown",
        "RefDocument": "unknown",
        "Assignment": "unknown",
        "StoreNumber": "unknown",
        "AcquireRefNumber": "unknown",
        "WebOrderNumber": "W1342421414",
        "MatchGroupId": "14443553",
        "Source": "SAP",
        "SourceType": "Internal"
    }
])

# -------------------------------------------------------------------
# 2. DROP FULLY UNKNOWN COLUMNS
# -------------------------------------------------------------------

drop_cols = ["ReceiptNumber", "RefDocument", "Assignment", "StoreNumber", "AuthCode"]
df = df.drop(columns=drop_cols)


# -------------------------------------------------------------------
# 3. FEATURE PREPARATION
# -------------------------------------------------------------------

# Convert CR/DR → +1 / -1
df["SignedAmount"] = df["Amount"] * df["CR_DR"].map({"CR": 1, "DR": -1})

# Label encode categorical columns
label_cols = ["DocType", "TransactionType", "Source", "SourceType"]
for col in label_cols:
    df[col] = LabelEncoder().fit_transform(df[col])


# -------------------------------------------------------------------
# 4. CHARACTER-LEVEL EMBEDDINGS (Untrained, deterministic)
# -------------------------------------------------------------------

def char_level_embed(text, embedding_dim=32):
    """
    Simple char embedding: assign each ASCII char an embedding vector, 
    then average the embeddings for the characters in the string.
    """
    if pd.isna(text) or text is None:
        return np.zeros(embedding_dim)

    # Build a fixed random embedding for each ASCII code
    np.random.seed(42)
    char_table = {chr(i): np.random.randn(embedding_dim) for i in range(128)}

    # Compute mean embedding
    vectors = []
    for ch in str(text):
        if ord(ch) < 128:
            vectors.append(char_table[ch])

    if len(vectors) == 0:
        return np.zeros(embedding_dim)

    return np.mean(vectors, axis=0)

# Columns requiring char embeddings
char_cols = [
    "MerchantRefNum",
    "WebOrderNumber",
    "AcquireRefNumber",
    "PONumber",
    "TransactionRefNo",
    "CardNo"
]

# Apply the character embedding
for col in char_cols:
    emb = np.vstack([char_level_embed(x) for x in df[col]])
    for i in range(emb.shape[1]):
        df[f"{col}_emb_{i}"] = emb[:, i]


# -------------------------------------------------------------------
# 5. BUILD FINAL FEATURE MATRIX
# -------------------------------------------------------------------

feature_cols = []

# numeric columns
feature_cols += ["SignedAmount"]

# encoded categorical columns
feature_cols += label_cols

# embedded character columns
for col in char_cols:
    feature_cols += [f"{col}_emb_{i}" for i in range(32)]

X = df[feature_cols].values


# -------------------------------------------------------------------
# 6. RUN DBSCAN (UNSUPERVISED CLUSTERING)
# -------------------------------------------------------------------

db = DBSCAN(eps=0.5, min_samples=2, metric='euclidean')
df["Cluster"] = db.fit_predict(X)

print("\nRaw DBSCAN Clusters:")
print(df[["Amount", "SignedAmount", "Cluster"]])


# -------------------------------------------------------------------
# 7. POST-PROCESSING TO ENFORCE SUM(Amount)=0 CONSTRAINT
# -------------------------------------------------------------------

def get_balanced_subsets(amounts):
    """
    Search for subsets whose amounts sum to zero.
    Used to salvage valid clusters from DBSCAN clusters.
    """
    indices = list(range(len(amounts)))
    balanced_sets = []

    # Try all pairs, triples, etc.
    for r in range(2, min(6, len(amounts)) + 1):  # limit for compute safety
        for combo in combinations(indices, r):
            if abs(sum(amounts[i] for i in combo)) < 1e-6:
                balanced_sets.append(combo)

    return balanced_sets


# Step 1: Extract cluster groups
final_clusters = {}
cluster_id = 0

for clust in df["Cluster"].unique():
    cluster_members = df[df["Cluster"] == clust]
    amts = cluster_members["SignedAmount"].values

    # Find valid subsets
    subsets = get_balanced_subsets(amts)

    if len(subsets) == 0:
        continue  # skip — no valid balancing combinations

    # Assign new constraint-satisfying clusters
    for subset in subsets:
        row_indices = cluster_members.iloc[list(subset)].index
        final_clusters[cluster_id] = row_indices
        cluster_id += 1

# Assign FinalCluster label
df["FinalCluster"] = -1
for cid, rows in final_clusters.items():
    df.loc[rows, "FinalCluster"] = cid


print("\nFinal Clusters (with Amount sum=0):")
print(df[["Amount", "SignedAmount", "Cluster", "FinalCluster"]])
