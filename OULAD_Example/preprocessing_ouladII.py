import pandas as pd

df = pd.read_csv("oulad_first_stage_preprocessed.csv")

df = df[["gender","region","highest_education", "imd_band", "age_band", "disability", "num_of_prev_attempts", "studied_credits", "avg_tma", "avg_cma", "final_result"]]




df["region"] = df['region'].replace({'East Anglian Region': 1, 'Scotland': 2, 'South East Region': 3,
                            'West Midlands Region': 4, 'Wales': 5, 'North Western Region': 6, 'North Region': 7,
                            'South Region': 8, 'Ireland': 9, 'South West Region': 10, 'East Midlands Region': 11,
                            'Yorkshire Region': 12, 'London Region': 13})

df["highest_education"] = df['highest_education'].replace({'No Formal quals': 0, 'Lower Than A Level': 1, 'A Level or Equivalent': 2,
                            'HE Qualification': 3, 'Post Graduate Qualification': 4})

df["imd_band"] = df['imd_band'].replace({'0-10%': 0, '10-20': 1, '20-30%': 2,
                            '30-40%': 3, '40-50%': 4, "50-60%": 5, "60-70%": 6, "70-80%": 7, "80-90%": 8, "90-100%": 9})

df["age_band"] = df['age_band'].replace({'0-35': 0, '35-55': 1, '55<=': 2})

df.to_csv("oulad_preprocessed.csv")