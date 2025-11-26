# End-to-end data preparation, feature engineering, blocking,
# constraint-aware grouping (graph + search), optional ML refinement,
# and evaluation for transaction reconciliation clustering.
# (No MatchGroupId used in features; it is used only for training/eval labels.)

import pandas as pd
import numpy as np
import itertools
import networkx as nx
from datetime import timedelta
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import adjusted_rand_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

# -------------------------
# CONFIG
# -------------------------
AMOUNT_TOLERANCE = 0.01       # tolerance for zero-sum checks (currency units)
MAX_CLUSTER_SIZE = 5          # per your dataset stats
TIME_WINDOW_DAYS = 3          # business rule
PAIRWISE_NEG_SAMPLE_RATIO = 3 # for training negative pairs per positive pair

# -------------------------
# INPUT: load your DataFrame
# -------------------------
# Replace this with pd.read_csv("transactions.csv") as needed.

df = df_filtered.copy(deep=True)

# Ensure correct dtypes
df = df.replace({"": None})
# Parse date (try multiple formats)
df["DocumentDate"] = pd.to_datetime(df["DocumentDate"], errors="coerce", dayfirst=True)

# Numeric Amount
df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0.0)

# Standardize text fields
for col in ["DocType", "TransactionType", "BankTrfRef", "TransactionRefNo", "MerchantRefNum",
            "CR_DR", "GLRecordID", "OID", "PONumber", "CardNo", "ReceiptNumber",
            "AccountingDocNum", "AuthCode", "RefDocument", "Assignment", "StoreNumber",
            "AcquireRefNumber", "WebOrderNumber", "Source", "SourceType"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().replace({"nan": None, "None": None}).fillna("unknown")

# Signed amount: treat CR as +, DR as - (adjust if your business uses opposite sign)
df["CR_DR"] = df["CR_DR"].str.upper().fillna("unknown")
df["signed_amount"] = df["Amount"].astype(float) * df["CR_DR"].map({"CR": 1.0, "DR": -1.0}).fillna(1.0)

# Prefix blocking keys per business rules
df["bank_prefix_6"] = df["BankTrfRef"].apply(lambda x: x[:6] if isinstance(x, str) and len(x) >= 6 and x.lower() != "unknown" else None)
df["acq_prefix_9"] = df["AcquireRefNumber"].apply(lambda x: x[:9] if isinstance(x, str) and len(x) >= 9 and x.lower() != "unknown" else None)

# Combined block key (require both prefixes per business rule)
df["block_key"] = df.apply(lambda r: (r["bank_prefix_6"] + "|" + r["acq_prefix_9"]) if (r["bank_prefix_6"] and r["acq_prefix_9"]) else None, axis=1)

# Assign a unique integer ID for each row
df = df.reset_index(drop=True)
df["_idx"] = df.index

# -------------------------
# Feature engineering helpers
# -------------------------
def last_n(s, n=4):
    return s[-n:] if isinstance(s, str) and len(s) >= n else None

df["card_last4"] = df["CardNo"].apply(lambda x: last_n(x, 4))
df["trxnref_last6"] = df["TransactionRefNo"].apply(lambda x: last_n(x, 6))
df["merchant_last6"] = df["MerchantRefNum"].apply(lambda x: last_n(x, 6))
df["amount_rounded"] = df["Amount"].round(2)

# -------------------------
# Pairwise feature function
# -------------------------
def pairwise_features(a: pd.Series, b: pd.Series):
    feats = {}
    feats["idx_a"] = int(a["_idx"])
    feats["idx_b"] = int(b["_idx"])
    feats["abs_amount_diff"] = abs(a["Amount"] - b["Amount"])
    feats["signed_sum"] = a["signed_amount"] + b["signed_amount"]
    feats["amount_ratio"] = (a["Amount"] / b["Amount"]) if b["Amount"] != 0 else 0.0
    feats["days_diff"] = abs((a["DocumentDate"] - b["DocumentDate"]).days) if pd.notnull(a["DocumentDate"]) and pd.notnull(b["DocumentDate"]) else 9999
    feats["same_bank_prefix"] = int(bool(a["bank_prefix_6"] and b["bank_prefix_6"] and a["bank_prefix_6"] == b["bank_prefix_6"]))
    feats["same_acq_prefix"] = int(bool(a["acq_prefix_9"] and b["acq_prefix_9"] and a["acq_prefix_9"] == b["acq_prefix_9"]))
    feats["same_card_last4"] = int(bool(a["card_last4"] and b["card_last4"] and a["card_last4"] == b["card_last4"]))
    feats["exact_transref"] = int(bool(a["TransactionRefNo"] and b["TransactionRefNo"] and a["TransactionRefNo"] == b["TransactionRefNo"]))
    feats["exact_po"] = int(bool(a["PONumber"] and b["PONumber"] and a["PONumber"] == b["PONumber"]))
    feats["exact_weborder"] = int(bool(a["WebOrderNumber"] and b["WebOrderNumber"] and a["WebOrderNumber"] == b["WebOrderNumber"]))
    feats["same_store"] = int(bool(a["StoreNumber"] and b["StoreNumber"] and a["StoreNumber"] == b["StoreNumber"]))
    feats["same_doctype"] = int(a["DocType"] == b["DocType"])
    feats["same_trxtype"] = int(a["TransactionType"] == b["TransactionType"])
    return feats

# -------------------------
# Candidate graph building per block
# -------------------------
def build_candidate_graph(block_df):
    G = nx.Graph()
    for idx in block_df["_idx"]:
        G.add_node(int(idx))
    records = block_df.set_index("_idx").to_dict("index")
    idx_list = list(records.keys())
    for i, j in itertools.combinations(idx_list, 2):
        a = records[i]; b = records[j]
        # time constraint
        if pd.isnull(a["DocumentDate"]) or pd.isnull(b["DocumentDate"]):
            days_ok = True
        else:
            days_ok = abs((a["DocumentDate"] - b["DocumentDate"]).days) <= TIME_WINDOW_DAYS
        if not days_ok:
            continue

        # quick candidate criteria: complementary signs and amount nearly cancels
        signed_sum = a["signed_amount"] + b["signed_amount"]
        if abs(signed_sum) <= AMOUNT_TOLERANCE:
            G.add_edge(int(i), int(j), reason="amount_zero")
            continue

        # or exact identifier matches (strong)
        strong_id_match = False
        for fld in ["TransactionRefNo", "PONumber", "WebOrderNumber", "MerchantRefNum", "AccountingDocNum", "AuthCode"]:
            if a.get(fld) and b.get(fld) and a[fld] == b[fld] and a[fld] != "unknown":
                strong_id_match = True
                break
        if strong_id_match:
            G.add_edge(int(i), int(j), reason="id_match")
            continue

        # or last-n card match + amount complement within relative tolerance
        if a.get("card_last4") and a.get("card_last4") == b.get("card_last4"):
            if abs(a["signed_amount"] + b["signed_amount"]) <= (max(1.0, abs(a["Amount"])) * 0.05):  # 5% tolerance heuristic
                G.add_edge(int(i), int(j), reason="card_and_amount")
                continue

        # else no edge
    return G

# -------------------------
# Partition search: find clusters in a component such that each cluster sums to zero (within tol)
# -------------------------
def find_zero_sum_partition(node_list, node_signed_amount_map, node_date_map, tol=AMOUNT_TOLERANCE, max_size=MAX_CLUSTER_SIZE):
    nodes = list(node_list)
    nodes = sorted(nodes)
    memo = {}

    def helper(remaining_tuple):
        if not remaining_tuple:
            return []
        if remaining_tuple in memo:
            return memo[remaining_tuple]
        remaining = list(remaining_tuple)
        # try larger subsets first (prefer larger clusters)
        for size in range(min(max_size, len(remaining)), 1, -1):
            # iterate combinations
            for comb in itertools.combinations(remaining, size):
                s = sum(node_signed_amount_map[n] for n in comb)
                if abs(s) <= tol:
                    # check time span within cluster
                    dates = [node_date_map[n] for n in comb if node_date_map[n] is not None]
                    if dates:
                        span = (max(dates) - min(dates)).days
                        if span > TIME_WINDOW_DAYS:
                            continue
                    # accept this comb and recurse
                    remaining_after = tuple(x for x in remaining if x not in comb)
                    rest = helper(tuple(sorted(remaining_after)))
                    if rest is not None:
                        res = [list(comb)] + rest
                        memo[remaining_tuple] = res
                        return res
        # no valid subset partition found; return None
        memo[remaining_tuple] = None
        return None

    return helper(tuple(nodes))

# -------------------------
# Main clustering pipeline
# -------------------------
predicted_clusters = {}  # idx -> cluster_id
cluster_id_counter = 0
unresolved_components = []

# iterate blocks
blocks = df.groupby("block_key")
for block_key, block in blocks:
    if block_key is None:
        # records without required prefixes: optionally handle separately (skip or attempt looser matching)
        # For simplicity, we skip heavy grouping here and mark them as singletons initially.
        for idx in block["_idx"]:
            predicted_clusters[int(idx)] = cluster_id_counter
            cluster_id_counter += 1
        continue

    # further split block into overlapping windows by date to enforce time rule and keep components small
    block = block.sort_values("DocumentDate").reset_index(drop=True)
    # build candidate graph for the block
    G = build_candidate_graph(block)
    for comp in nx.connected_components(G):
        comp_nodes = sorted(list(comp))
        node_signed_amount_map = {n: float(df.loc[df["_idx"]==n, "signed_amount"].iloc[0]) for n in comp_nodes}
        node_date_map = {n: df.loc[df["_idx"]==n, "DocumentDate"].iloc[0] for n in comp_nodes}
        # attempt exact zero-sum partition search
        partition = find_zero_sum_partition(comp_nodes, node_signed_amount_map, node_date_map)
        if partition is not None:
            # assign cluster ids
            for group in partition:
                cid = cluster_id_counter
                cluster_id_counter += 1
                for idx in group:
                    predicted_clusters[int(idx)] = cid
        else:
            # fallback: try pairwise greedy matching using strong edges or leave as singletons
            # Attempt greedy pairing by absolute signed sum minimal pairs
            nodes_left = set(comp_nodes)
            groups = []
            while nodes_left:
                if len(nodes_left) == 1:
                    n = nodes_left.pop()
                    groups.append([n])
                    break
                # find pair with minimal abs(signed_sum)
                best_pair = None
                best_val = float("inf")
                for a, b in itertools.combinations(nodes_left, 2):
                    val = abs(node_signed_amount_map[a] + node_signed_amount_map[b])
                    days_ok = True
                    if node_date_map[a] is not None and node_date_map[b] is not None:
                        days_ok = abs((node_date_map[a] - node_date_map[b]).days) <= TIME_WINDOW_DAYS
                    if days_ok and val < best_val:
                        best_val = val
                        best_pair = (a, b)
                if best_pair and best_val <= (max(abs(node_signed_amount_map[best_pair[0]]), abs(node_signed_amount_map[best_pair[1]])) * 0.2):
                    # accept pair as group (heuristic tolerance 20%)
                    groups.append([best_pair[0], best_pair[1]])
                    nodes_left.remove(best_pair[0])
                    nodes_left.remove(best_pair[1])
                else:
                    # no good pair found: make singletons out of remaining nodes
                    for n in list(nodes_left):
                        groups.append([n])
                        nodes_left.remove(n)
            # assign cluster ids for fallback groups
            for group in groups:
                cid = cluster_id_counter
                cluster_id_counter += 1
                for idx in group:
                    predicted_clusters[int(idx)] = cid
            # mark for potential ML refinement
            unresolved_components.append(list(comp_nodes))

# assign predicted cluster id to df
df["pred_cluster"] = df["_idx"].apply(lambda x: predicted_clusters.get(int(x), -1))

# -------------------------
# Optional ML refinement: train a pairwise classifier and use it to refine ambiguous components
# -------------------------
# Build pairwise training data from blocks where MatchGroupId is available (labels only for training)
if "MatchGroupId" in df.columns:
    # build candidate pairs across entire dataset for training (within same block_key to avoid unrealistic negatives)
    pairs = []
    for block_key, block in df.groupby("block_key"):
        block_idxs = list(block["_idx"])
        for i, j in itertools.combinations(block_idxs, 2):
            a = df.loc[df["_idx"]==i].iloc[0]
            b = df.loc[df["_idx"]==j].iloc[0]
            feats = pairwise_features(a, b)
            # label: same MatchGroupId (use for training only)
            same = int(a["MatchGroupId"] == b["MatchGroupId"])
            feats["label"] = same
            pairs.append(feats)
    pair_df = pd.DataFrame(pairs)
    # balance: sample negatives
    pos = pair_df[pair_df["label"]==1]
    neg = pair_df[pair_df["label"]==0].sample(frac=1.0, random_state=42)
    if len(pos) * PAIRWISE_NEG_SAMPLE_RATIO < len(neg):
        neg = neg.sample(n=min(len(neg), len(pos) * PAIRWISE_NEG_SAMPLE_RATIO), random_state=42)
    train_df = pd.concat([pos, neg]).sample(frac=1.0, random_state=42)

    if len(train_df) >= 10 and train_df["label"].nunique() > 1:
        X = train_df.drop(columns=["idx_a", "idx_b", "label"])
        y = train_df["label"].astype(int)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        # optional validation metrics
        y_pred = clf.predict(X_val)
        prec, rec, f1, _ = precision_recall_fscore_support(y_val, y_pred, average="binary", zero_division=0)
        print("Pairwise classifier val precision, recall, f1:", prec, rec, f1)

        # Use classifier to refine unresolved components: for each unresolved comp, add edges for high-prob pairs and re-run partitioning
        for comp_nodes in unresolved_components:
            # only attempt if component size reasonable
            if len(comp_nodes) > 1 and len(comp_nodes) <= 20:
                # compute pairwise probabilities
                prob_edges = []
                for a, b in itertools.combinations(comp_nodes, 2):
                    ra = df.loc[df["_idx"]==a].iloc[0]; rb = df.loc[df["_idx"]==b].iloc[0]
                    feats = pairwise_features(ra, rb)
                    X_row = pd.DataFrame([feats]).drop(columns=["idx_a", "idx_b"])
                    prob = clf.predict_proba(X_row)[0,1]
                    prob_edges.append((a, b, prob, feats))
                # build new graph with edges above threshold
                H = nx.Graph()
                for n in comp_nodes:
                    H.add_node(n)
                # threshold can be tuned; use 0.7
                for a, b, prob, feats in prob_edges:
                    if prob >= 0.7:
                        H.add_edge(a, b, prob=prob)
                # connected components in H
                for sub in nx.connected_components(H):
                    sub = list(sub)
                    # attempt zero-sum partition on this subset
                    node_signed_amount_map = {n: float(df.loc[df["_idx"]==n, "signed_amount"].iloc[0]) for n in sub}
                    node_date_map = {n: df.loc[df["_idx"]==n, "DocumentDate"].iloc[0] for n in sub}
                    partition = find_zero_sum_partition(sub, node_signed_amount_map, node_date_map)
                    if partition is not None:
                        for group in partition:
                            cid = cluster_id_counter
                            cluster_id_counter += 1
                            for idx in group:
                                predicted_clusters[int(idx)] = cid
                    else:
                        # fallback: greedy pairs by classifier probability
                        used = set()
                        sorted_pairs = sorted([e for e in prob_edges if e[0] in sub and e[1] in sub], key=lambda x: -x[2])
                        for a, b, prob, _ in sorted_pairs:
                            if a in used or b in used:
                                continue
                            if prob >= 0.75:
                                cid = cluster_id_counter
                                cluster_id_counter += 1
                                predicted_clusters[int(a)] = cid
                                predicted_clusters[int(b)] = cid
                                used.update([a,b])
                        for n in sub:
                            if n not in used:
                                cid = cluster_id_counter
                                cluster_id_counter += 1
                                predicted_clusters[int(n)] = cid

# update df predictions after ML refinement
df["pred_cluster"] = df["_idx"].apply(lambda x: predicted_clusters.get(int(x), -1))

# -------------------------
# Evaluation against MatchGroupId (if available)
# -------------------------
if "MatchGroupId" in df.columns:
    # Compute pairwise-level metrics
    idx_to_pred = dict(zip(df["_idx"], df["pred_cluster"]))
    true_pairs = set()
    pred_pairs = set()
    for a, b in itertools.combinations(df["_idx"], 2):
        a = int(a); b = int(b)
        same_true = int(df.loc[df["_idx"]==a, "MatchGroupId"].iloc[0] == df.loc[df["_idx"]==b, "MatchGroupId"].iloc[0])
        same_pred = int(idx_to_pred[a] == idx_to_pred[b])
        if same_true:
            true_pairs.add((a,b))
        if same_pred:
            pred_pairs.add((a,b))
    tp = len(true_pairs & pred_pairs)
    fp = len(pred_pairs - true_pairs)
    fn = len(true_pairs - pred_pairs)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    print(f"Pairwise precision={prec:.4f}, recall={rec:.4f}, f1={f1:.4f}")

    # Cluster-level metrics: ARI
    true_labels = df["MatchGroupId"].astype(str).tolist()
    pred_labels = df["pred_cluster"].astype(str).tolist()
    ari = adjusted_rand_score(true_labels, pred_labels)
    print("Adjusted Rand Index (ARI):", ari)

# -------------------------
# Output: final df with predicted clusters
# -------------------------
# reorder columns for readability
out_cols = ["_idx", "pred_cluster", "MatchGroupId"] + [c for c in df.columns if c not in ["_idx", "pred_cluster", "MatchGroupId"]]
out_cols = [c for c in out_cols if c in df.columns]
final_df = df[out_cols].copy()
print(final_df.head(50))
