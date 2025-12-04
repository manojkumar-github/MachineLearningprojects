4th:

Below is the correct way to design a custom loss function so that the model learns embeddings where:
	1.	Transactions within ±3 days should be close in embedding space,
	2.	Transactions whose SignedAmount values sum to 0 (in pairs or groups) should be close,
	3.	Everything else should be pushed far apart,
	4.	Fully unsupervised OR weakly supervised,
	5.	No rule-based processing outside the loss — the network learns the rules directly.

This is the same idea used by:
	•	Deep metric learning,
	•	Self-supervised contrastive learning,
	•	Temporal sensitivity embedding,
	•	Value-balancing embedding (your case).

⸻

✅ Define the Problem as a Metric Learning Task

We want to learn an embedding function:

f(x) → ℝ^D

such that:

✔ If two items A and B meet these conditions:
	•	|DocumentDate(A) – DocumentDate(B)| ≤ 3 days
	•	They participate in a zero-sum amount balancing group
(e.g., A.Amount + B.Amount = 0 OR A+B+C = 0)

→ Their embeddings should be close
 ‖f(A) – f(B)‖ small

✔ Otherwise embeddings should be far

 ‖f(A) – f(B)‖ large

This naturally enables DBSCAN/HDBSCAN to cluster correctly.

⸻

🌟 THE BEST WAY: Create a Compound Contrastive Loss Function

We build a multi-term contrastive loss:

L = w1 * L_date + w2 * L_amount + w3 * L_negative

Where:
	•	L_date enforces closeness for date-proximal pairs
	•	L_amount enforces closeness for zero-sum pairs
	•	L_negative pushes far apart everything else

This is the cleanest and most powerful design.

⸻

🧠 1. Date-based Positive Pair Loss

Two transactions with Date difference ≤ 3 days are positive pairs.

def date_positive_mask(dates, max_days=3):
    diff = torch.abs(dates.unsqueeze(0) - dates.unsqueeze(1))
    return (diff <= max_days).float()

Loss (contrastive):

def date_loss(emb, date_mask, margin=1.0):
    # Positive similarity
    pos_pairs = date_mask > 0
    pos_dist = (emb.unsqueeze(1) - emb.unsqueeze(0)).pow(2).sum(-1)
    pos_loss = (pos_pairs * pos_dist).sum() / (pos_pairs.sum() + 1e-6)
    return pos_loss


⸻

💰 2. Amount-balancing Positive Pair Loss

Two samples are positive if they are part of any zero-sum group.

Define:

balance_mask[i][j] = 1 if Amount_i + Amount_j ≈ 0

For multi-item clusters, use residue matching:

abs(sum(group_amounts)) small → group positive

Simplest pairwise version:

def amount_positive_mask(amounts, tol=1e-4):
    sums = amounts.unsqueeze(0) + amounts.unsqueeze(1)
    return (torch.abs(sums) < tol).float()

Loss:

def amount_loss(emb, amount_mask):
    pos_pairs = amount_mask > 0
    pos_dist = (emb.unsqueeze(1) - emb.unsqueeze(0)).pow(2).sum(-1)
    return (pos_pairs * pos_dist).sum() / (pos_pairs.sum() + 1e-6)

This alone will significantly improve clustering.

⸻

🧨 3. Negative Contrastive Loss

Ensure embeddings are far apart for negative pairs:

negative_mask = 1 - (date_mask OR amount_mask)

Contrastive “push-away” loss:

def negative_loss(emb, pos_mask, margin=1.0):
    neg_mask = 1 - pos_mask
    dist = (emb.unsqueeze(1) - emb.unsqueeze(0)).pow(2).sum(-1)
    neg_dist = torch.clamp(margin - torch.sqrt(dist + 1e-6), min=0)
    return (neg_mask * neg_dist).sum() / (neg_mask.sum() + 1e-6)


⸻

🧮 Final Loss Function

Putting it all together:

def custom_loss(emb, dates, amounts,
                w_date=1.0, w_amount=2.0, w_neg=1.0, margin=1.0):

    date_mask = date_positive_mask(dates)
    amt_mask = amount_positive_mask(amounts)
    pos_mask = torch.clamp(date_mask + amt_mask, 0, 1)

    Ld = date_loss(emb, date_mask)
    La = amount_loss(emb, amt_mask)
    Ln = negative_loss(emb, pos_mask, margin)

    return w_date * Ld + w_amount * La + w_neg * Ln


⸻

🎁 Why This Works Perfectly for Your Business Case

✔ Learns “±3-day proximity” naturally

 No hand-coded windowing needed in the clustering stage.

✔ Learns amount balancing implicitly

 The embeddings will encode “this belongs to group that sums to 0”.

✔ No need for graph matching

 Embeddings already encode balancing constraints.

✔ Unsupervised

 Does not use MatchGroupId — works for production.

✔ Fast at inference

 After training, clustering is just DBSCAN.

⸻

🚀 Want full code integrated into your hybrid Transformer encoder?

I can regenerate the full pipeline:
	•	Field-wise CNN encoders
	•	Transformer encoder
	•	Autoencoder training
	•	This new custom loss function
	•	DBSCAN
	•	Evaluation

Just say:

“Generate full code with this loss function integrated.”



Planning:

4 places

Data:

Increase datasetsize - 2<cluster_size<5 instead of 5 vs 10?
Dataset Cleaning - Are there existing columns that are causing noise?

Feature Engineering:
DocumentDate not being used any current features
categorical
Char embedding?
Some idea

Model training:
Epochs
Customloss


DBScan settings hyperparamter grid search

Some quick stats:

- 2_ : 157 clusters (stats look better)
- 3_ : 22 clusters
- 4_: 51 clusters
- 5_: 
- 6_: 13 clusters
- 7_: 44 clusters
- 8_: 24 clusters
- 9_: 14 clusters
- 10_: 105 clusters






### Inital Plan:

Plan for Unsupervised Clustering (Not using MatchGroupIds information through any means)

- Data cleaning:  Remove the specified columns (‘ReceiptNumber’, ‘RefDocument’, ‘Assignment’, ‘StoreNumber’, ‘AuthCode’) since they contain no useful information (all “unknown” values).  Also remove any “date-window” grouping logic so that transactions are not constrained by time windows.  Keep the remaining fields (including text fields like IDs and reference numbers) for feature generation.
  
- Feature encoding:  Convert the remaining transaction fields into a numeric feature space.  For example, we can use character-level embeddings encodings for categorical/text fields and use the numeric Amount.  We should include the credit/debit indicator (CR_DR) as a feature (e.g. map “CR”→+1, “DR”→–1).  This yields an (n×d) feature matrix for n transactions.
  
- Clustering method:  Apply an unsupervised clustering algorithm to group similar transactions like DBScan
  
- Sum-to-zero constraint (Key step):  Standard clustering does not enforce that the sum of Amount in each cluster is zero.  To satisfy this constraint, we can treat it as a `constraint optimization problem` ￼.  In practice one approach is: first do the unsupervised clustering normally, then post-process the clusters to enforce balance.  For example, after forming clusters, compute each cluster’s total Amount.  If a cluster’s sum isn’t zero, we can adjust it by moving transactions between clusters or splitting/merging clusters until each cluster’s sum is 0.  This is akin to solving a knapsack/partition problem (as suggested on StackOverflow ￼).  In other words, we might reassign or pair transactions so that each cluster contains offsetting credits and debits (so that their net sum is zero).
- Iteration and validation:  Iterate the above steps (possibly adjusting features or cluster count) until clusters meet the sum=0 requirement.
References: Unsupervised clustering (e.g. DBScan) groups data without labels ￼ ￼.  Enforcing a sum-of-amounts constraint requires formulating a constrained optimization or adjusting clusters after clustering


### Experiments:

#### Experiment: 1

Code Link: https://github.com/manojkumar-github/MachineLearningprojects/blob/master/retsulc/2_dnn_charembedding.py

Summary:

"""
End-to-end pipeline:
- Load data (single-row example or CSV)
- Drop fully-unknown columns drop_cols = ["ReceiptNumber", "RefDocument", "Assignment", "StoreNumber", "AuthCode"]
- Prepare character vocabulary
- Build a deep neural char-level autoencoder (encoder produces embeddings)
```
fields to use in the char-level encoder (ID-like fields)
char_fields = [
    "MerchantRefNum",
    "WebOrderNumber",
    "AcquireRefNumber",
    "PONumber",
    "TransactionRefNo",
    "CardNo",
    "AccountingDocNum"
]
```

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

Results:
precision: 0.4787, recall 0.1931 and f1: 0.2752 (TODO: Attach stats{} dataframe screenshot)


#### Experiment-2:

Code link : https://github.com/manojkumar-github/MachineLearningprojects/blob/master/retsulc/3_selfsupervisedcontrastiveloss.py

Previous experiment: simple CNN → avg pool → projection → no pretraining → no semantics learned

Improvement-1:

Added contrastive pretraining (self-supervised)

Even without MatchGroupId, we can create positive pairs using augmentation:

For each ID-like field (TransactionRefNo, MerchantRefNum, etc.):
	•	Positive pair = (raw string, augmented string)
	•	Augmentations:
	•	random character dropout
	•	random transposition
	•	lowercase/uppercase
	•	special symbol masking
	•	prefix cropping

This forces embeddings to learn invariances in payment IDs.

Used NT-Xent or Contrastive Loss.

Improvement-2:

DBSCAN is extremely sensitive to:
	•	eps
	•	min_samples
```
eps ∈ [0.1, 0.3, 0.5, 0.7, 1.0]
min_samples ∈ [2, 3, 4, 5]
```

So tried  grid search and used precision@cluster as tuning metric.

Improvment-3:

Then solve:
	•	maximum weight perfect matching (Hungarian algorithm)
	•	OR minimum-cost flow with balance constraint

This ensures sum=0 and uses similarity more intelligently.

```
node = transaction
edge weight = similarity score (embedding distance inverse)
constraint: sum(amounts)=0
```

Summary of experiment-2:

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
