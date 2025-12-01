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
