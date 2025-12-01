High-level pipeline:

1.	Blocking/Dataset Partition:
	
	- Partition dataset by BankTrfRef[:6] AND AcquireRefNumber[:9]
	(Assumption: Most of the records within MatchGroupIds share similar/same prefixes)

2.	Time window split:
	- Within each block/partition, split into overlapping windows by DocumentDate so that every window contains only records within any rolling 3-day span. This reduces component size.

3.	Create candidate graph:
	- Add edges where transactions are within 3 days AND have complementary sign (CR vs DR) and optionally have amount closeness (|a+b| < tolerance) or matching other identifiers. This yields connected components of likely matches.

4.	Cluster formation per component:
	- For each component, perform a constrained grouping algorithm: because clusters are small (2–5 per your earlier stats), a backtracking/subset enumeration that tries all subsets to find partitions whose signed sums ≈ 0 is feasible.
	- If multiple valid partitions exist, pick the one that maximizes a secondary score (e.g., minimal intra-cluster time span, maximal count of exact identifier matches, or a learned pairwise confidence).

5.	Fallback for ambiguous cases
	- Flag components where no zero-sum partition is found or there are many candidate partitions
