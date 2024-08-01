"""
Function to rank frequency of 'yes' and 'no' answers of each user's Spain Domestic Violence Form Submission. 
The more yes answers the higher the risk:
        If the user has 2 'yes' in the risk assessment (section 5) define them as high risk, more than 2 should be as extreme risk. 
        If the user includes more than 10 words in the 'other' sections (which must be labeled and grouped up to check this), define them as medium risk and go up from there if needed.
        If the user does not meet the circumstances listed above, define them as low risk:
            Define as medium risk if perpetrator has been violent in the past (section 4)
            Define as medium risk if user answers 'yes' at least once in section 6.

Form uses the same risk classifications as the Police of Spain are used to with VioGen:

    - unappreciated
	- low
	- medium
	- high
	- extreme
"""