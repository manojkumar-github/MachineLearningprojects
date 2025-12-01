import pandas as pd
import numpy as np
import networkx as nx
from datetime import timedelta

# -------------------------------------------------------------------------
# 1. LOAD + CLEAN DATA
# -------------------------------------------------------------------------

df = pd.read_csv("transactions.csv") #

# Clean date field
df["DocumentDate"] = pd.to_datetime(df["DocumentDate"], errors="coerce")

# Convert CR/DR into signed numeric amount
def convert_amount(row):
    multiplier = 1 if row["CR_DR"] == "CR" else -1  # CR = +, DR = -
    return multiplier * row["Amount"]

df["SignedAmount"] = df.apply(convert_amount, axis=1)

# Sort by date to help with efficient blocking
df = df.sort_values("DocumentDate").reset_index(drop=True)

# -------------------------------------------------------------------------
# 2. BLOCKING BY DATE (ONLY RULE USED)
#    Group only rows within a 3-day window together
# -------------------------------------------------------------------------

def create_date_blocks(df, window_days=3):
    blocks = []
    current_block = [df.index[0]]

    for i in range(1, len(df)):
        prev_date = df.loc[df.index[i-1], "DocumentDate"]
        curr_date = df.loc[df.index[i], "DocumentDate"]

        # If difference ≤ window, same block
        if (curr_date - prev_date).days <= window_days:
            current_block.append(df.index[i])
        else:
            blocks.append(current_block)
            current_block = [df.index[i]]

    blocks.append(current_block)
    return blocks


date_blocks = create_date_blocks(df, window_days=3)

print(f"Created {len(date_blocks)} date-based candidate blocks.")

# -------------------------------------------------------------------------
# 3. GRAPH-BASED CLUSTERING INSIDE EACH BLOCK
#    Rule: Clusters must sum to 0 based on SignedAmount
# -------------------------------------------------------------------------

def find_zero_sum_subgraphs(block_indices):
    block_df = df.loc[block_indices]

    # Create graph nodes
    G = nx.Graph()
    G.add_nodes_from(block_df.index)

    # Connect records that could logically be matched:
    # If their amounts have opposite signs, they are potential matches
    pos = block_df[block_df["SignedAmount"] > 0]
    neg = block_df[block_df["SignedAmount"] < 0]

    for i in pos.index:
        for j in neg.index:
            # Add an edge representing potential matching pair
            G.add_edge(i, j)

    # Now find connected components to explore combinations
    connected_components = list(nx.connected_components(G))

    clusters = []

    for comp in connected_components:
        comp_df = block_df.loc[list(comp)]

        # The sum over the component must be zero
        if abs(comp_df["SignedAmount"].sum()) < 1e-6:
            clusters.append(list(comp))

    return clusters


# -------------------------------------------------------------------------
# 4. RUN GROUPING ACROSS ALL BLOCKS
# -------------------------------------------------------------------------

all_clusters = []

for block in date_blocks:
    clusters = find_zero_sum_subgraphs(block)
    all_clusters.extend(clusters)

# Assign cluster IDs
cluster_map = {}
for cluster_id, indices in enumerate(all_clusters):
    for idx in indices:
        cluster_map[idx] = cluster_id

df["PredictedCluster"] = df.index.map(cluster_map).fillna(-1).astype(int)

# -------------------------------------------------------------------------
# 5. SAVE RESULTS
# -------------------------------------------------------------------------
df.to_csv("clustered_output.csv", index=False)

print("Clustering complete.")
print(df[["DocumentDate", "SignedAmount", "PredictedCluster"]].head())
