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

def lcs_len(a, b):
    m, n = len(a), len(b)
    dp = [[0]*(n+1) for _ in range(m+1)]

    for i in range(m):
        for j in range(n):
            if a[i] == b[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
    return dp[m][n]

for col1, col2 in combinations(char_fields, 2):
    df[f"{col1}_{col2}"] = df.apply(
        lambda r: lcs_len(str(r[col1]), str(r[col2])),
        axis=1
    )

