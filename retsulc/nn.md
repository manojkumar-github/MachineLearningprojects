Business Problem:
- Objective: Missing files 


External System:
	Files -> Format -> JSON format -> dateTime_Naming_Pattern_X3566_InterfaceName_Status


Internal System:
	- Mongo DB
	- S3?

File Import Rules:
	-> Where: Mongo DB?
	-> What: 
	-> CSV files
	-> What type of Rule
		-> What time what file should come?
		-> How many occurences in a week? (File pattern should come one week or twice in a week)
	-> Curent System:
		-> How?
			-> While Saving using a fileName pattern -> 

		-> What logic is currently being used?
			-> Every 10 mins?
	-> Edge Cases:
		-> Missing files

Problem Statement:
	-> Trying to come up with a logic (Model)
			-> Objective:
				-> Missing do not happen
				-> False alerts do not happen
			-> Model:
				-> Backwards:
					-> Occurrence Suggest -> Rule is not good -> File twice per week as per rule -> Historical behavior showing once
					-> Time Suggest -> 9:00 AM a file should come -> file miss alert happens -> File arrives near time -> According to historical behavior ->
				-> learning technique:

				-> Features:
					-> Historic Aggregations (Can you confirm whether rule schedule is unique per rule)
						-> Rule1, Rule Monday Current Schedule, N_Historic_Occurences, N_Passes, N_Fails, N_Fails_In_10_min_interval, N_Fails_In_20_min_interval, N_Fails_In_30_min_interval, N_Fails_In_40_min_interval
							Rule_Type, 123, da, 2, 1, 1 , 20min, 25min, 10min  <Time Suggest>
					-> Rule_Type, 123, da, Scheduled-Rule-file-1, CameInOneWeek-2-freq, CameInOneWeek-3-freq , 20min, 25min, 10min  <Occurrence Suggest>
							CameInOneWeek-2-freq-per-week -> 20 weeks
							CameInOneWeek-3-freq-per-week -> 30 weeks
							CameInOneWeek-15-freq-per-week -> 200 weeks

							max(20,30,200) -> freq -> 15

							Is this the scope of behavior -> Just basis on frequency should we go?

					-> What if there is completely new rule with schedule in the import file that is coming into system?


				-> DataSet Preparation:
					-> Rule1, Rule Monday Current Schedule, PASS
					-> Rule2, Rule Monday Current Schedule, FAIL


				-> Data:
					-> Rule -> File Import Rule -> 09:00 AM before all files should come -> Access?
					->  1 file -> transaction match -> 
					-> 09:00 AM -> Scheduler Schedule ->
					-> How many file import rules?
						-> Categorize
					-> iRecon MetaData
						- Files metadata?
					-> Job Alert Information (Clarify):
						-> As of today 10,000 files 
						-> File Import DB -> 20,000 files
						-> Future File import DB-> 5000 files


				-> Data Analysis?
					-> what is the nearest time>
						->

	-> Missing files
	-> False alert happening

Dashboard:
	 -> Interface Dashboard shows the file import rules -> 
	 -> Input Files -> Clustering Service -> AI -> Match logic -> match validate -> Output matching






