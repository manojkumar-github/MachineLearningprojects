
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

node = transaction
edge weight = similarity score (embedding distance inverse)
constraint: sum(amounts)=0
