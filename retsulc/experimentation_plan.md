
### Inital Plan:

Plan for Unsupervised Clustering (Not using MatchGroupIds information through any means)

- Data cleaning:  Remove the specified columns (‘ReceiptNumber’, ‘RefDocument’, ‘Assignment’, ‘StoreNumber’, ‘AuthCode’) since they contain no useful information (all “unknown” values).  Also remove any “date-window” grouping logic so that transactions are not constrained by time windows.  Keep the remaining fields (including text fields like IDs and reference numbers) for feature generation.
  
- Feature encoding:  Convert the remaining transaction fields into a numeric feature space.  For example, we can use character-level embeddings encodings for categorical/text fields and use the numeric Amount.  We should include the credit/debit indicator (CR_DR) as a feature (e.g. map “CR”→+1, “DR”→–1).  This yields an (n×d) feature matrix for n transactions.
  
- Clustering method:  Apply an unsupervised clustering algorithm to group similar transactions like DBScan
  
- Sum-to-zero constraint (Key step):  Standard clustering does not enforce that the sum of Amount in each cluster is zero.  To satisfy this constraint, we can treat it as a `constraint optimization problem` ￼.  In practice one approach is: first do the unsupervised clustering normally, then post-process the clusters to enforce balance.  For example, after forming clusters, compute each cluster’s total Amount.  If a cluster’s sum isn’t zero, we can adjust it by moving transactions between clusters or splitting/merging clusters until each cluster’s sum is 0.  This is akin to solving a knapsack/partition problem (as suggested on StackOverflow ￼).  In other words, we might reassign or pair transactions so that each cluster contains offsetting credits and debits (so that their net sum is zero).
- Iteration and validation:  Iterate the above steps (possibly adjusting features or cluster count) until clusters meet the sum=0 requirement.
References: Unsupervised clustering (e.g. DBScan) groups data without labels ￼ ￼.  Enforcing a sum-of-amounts constraint requires formulating a constrained optimization or adjusting clusters after clustering


### Experiments:

1. Experiment: 1 - https://github.com/manojkumar-github/MachineLearningprojects/blob/master/retsulc/2_dnn_charembedding.py
