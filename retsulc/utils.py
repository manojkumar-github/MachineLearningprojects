stats = {}

for idx, row in pred_cluster_counts_df.iterrows():

  current_category = f"pred_cluster_size_{row['count']}"
  current_cluster_id = row["pred_cluster"]
  current_df = df[df["pred_cluster"]==current_cluster_id]
  assert current_df.shape[0] == row["count"], "current df should match row count"
  n_uniq_match_group_ids = current_df["MatchGroupId1"].nunique()
  if current_category not in stats:
    # handle initliazation
    stats[current_category] = {"Clusters with SameMatchGroupIds (TP)": 0, "Clusters with DiffMatchGroupIds (FN)": 0}
  # first and next time
  if n_uniq_match_group_ids == 1:
    stats[current_category]["Clusters with SameMatchGroupIds (TP)"] += 1
  else:
    stats[current_category]["Clusters with DiffMatchGroupIds (FN)"] += 1


### LCS 
def longest_common_substring(a, b):
    if a == "unknown" or b == "unknown":
        return 0
    m, n = len(a), len(b)
    dp = [[0]*(n+1) for _ in range(m+1)]
    longest = 0
    end_pos = 0  # end position in string a

    for i in range(1, m+1):
        for j in range(1, n+1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
                if dp[i][j] > longest:
                    longest = dp[i][j]
                    end_pos = i
            # no else needed — dp[i][j] stays 0

    return len(a[end_pos - longest : end_pos])


# Character fields to compare
char_fields = [
    "MerchantRefNum",
    "WebOrderNumber",
    "AcquireRefNumber",
    "TransactionRefNo",
    "AccountingDocNum"
]

for col1, col2 in combinations(char_fields, 2):
    df[f"{col1}_{col2}"] = df.apply(
        lambda r: longest_common_substring(str(r[col1]), str(r[col2])),
        axis=1
    )


### Date Stats
# New cell

import pandas as pd

df["DocumentDate"] = pd.to_datetime(df["DocumentDate"], format="%Y/%m/%d")

# Size of each MatchGroupId
df["matchgroup_size"] = df.groupby("MatchGroupId")["MatchGroupId"].transform("size")

date_range_df = (
    df.groupby("MatchGroupId")["DocumentDate"]
      .agg(lambda x: (x.max() - x.min()).days)
      .rename("date_range_days")
      .reset_index()
)

df = df.merge(date_range_df, on="MatchGroupId", how="left")

df.columns

## Next cell
date_range_days = max(DocumentDate) - min(DocumentDate) within MatchGroupId

date_range_stats = (
    df[["MatchGroupId", "matchgroup_size", "date_range_days"]]
      .drop_duplicates("MatchGroupId")
      .groupby("matchgroup_size")["date_range_days"]
      .agg(
          max_date_range="max",
          min_date_range="min",
          mean_date_range="mean"
      )
      .reset_index()
)

date_range_stats


# LCS stats

import pandas as pd
import numpy as np
from itertools import combinations

# Treat NaN / null / 'unknown' as empty string
df[char_fields] = (
    df[char_fields]
    .fillna("")
    .replace("unknown", "", regex=False)
)

# MatchGroup size
df["matchgroup_size"] = df.groupby("MatchGroupId")["MatchGroupId"].transform("size")

def lcs_continuous_len(a, b):
    m, n = len(a), len(b)
    dp = [[0]*(n+1) for _ in range(m+1)]
    longest = 0

    for i in range(1, m+1):
        for j in range(1, n+1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
                longest = max(longest, dp[i][j])
    return longest

lcs_cols = []

for c1, c2 in combinations(char_fields, 2):
    col_name = f"{c1}_{c2}_LCS_len"
    lcs_cols.append(col_name)
    df[col_name] = df.apply(
        lambda r: lcs_continuous_len(r[c1], r[c2]),
        axis=1
    )

group_level = (
    df[["MatchGroupId", "matchgroup_size"] + lcs_cols]
    .drop_duplicates("MatchGroupId")
)

stats_df = (
    group_level
    .groupby("matchgroup_size")[lcs_cols]
    .agg(["mean", "max", "min", "std"])
)

# Flatten column names
stats_df.columns = [
    f"{col}_{stat}"
    for col, stat in stats_df.columns
]

stats_df = stats_df.reset_index()


