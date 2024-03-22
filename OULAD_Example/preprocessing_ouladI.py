import pandas as pd
import numpy as np

dataset_name = "OULAD"

assessments_df = pd.read_csv(f'{dataset_name}/assessments.csv')
courses_df = pd.read_csv(f'{dataset_name}/courses.csv')
studentAssessment_df = pd.read_csv(f'{dataset_name}/studentAssessment.csv')
studentInfo_df = pd.read_csv(f'{dataset_name}/studentInfo.csv')
studentRegistration_df = pd.read_csv(f'{dataset_name}/studentRegistration.csv')
studentVle_df = pd.read_csv(f'{dataset_name}/studentVle.csv')
vle_df = pd.read_csv(f'{dataset_name}/vle.csv')

studentInfo_df = studentInfo_df.loc[studentInfo_df.final_result != "Withdrawn"]

merged_assessments_df = pd.merge(studentAssessment_df, assessments_df, on="id_assessment")

avg_tma = [merged_assessments_df.loc[np.logical_and(merged_assessments_df.id_student == i,
                                                    merged_assessments_df.assessment_type == "TMA"), "score"].mean()
           for i in studentInfo_df.id_student.values]
avg_cma = [merged_assessments_df.loc[np.logical_and(merged_assessments_df.id_student == i,
                                                    merged_assessments_df.assessment_type == "CMA"), "score"].mean()
           for i in studentInfo_df.id_student.values]
avg_exam = [merged_assessments_df.loc[np.logical_and(merged_assessments_df.id_student == i,
                                                     merged_assessments_df.assessment_type == "Exam"), "score"].mean()
            for i in studentInfo_df.id_student.values]

studentInfo_df["avg_tma"] = avg_tma
studentInfo_df["avg_cma"] = avg_cma
studentInfo_df["avg_exam"] = avg_exam

vle_merged = pd.merge(studentVle_df, vle_df, on=["code_module", "code_presentation", "id_site"])
for activity_type in vle_df.activity_type.unique():
    agg = vle_merged.loc[vle_merged.activity_type == activity_type].groupby("id_student")
    count_click_dict = dict(agg.count()["sum_click"])
    sum_click_dict = dict(agg.sum()["sum_click"])
    studentInfo_df[f"Count_Visits_{activity_type}"] = studentInfo_df["id_student"].apply(
        lambda x: sum_click_dict[x] if x in count_click_dict.keys() else 0)
    studentInfo_df[f"Sum_Clicks_{activity_type}"] = studentInfo_df["id_student"].apply(
        lambda x: sum_click_dict[x] if x in sum_click_dict.keys() else 0)

df = studentInfo_df.copy()
df = df.drop(["id_student", "code_module", "code_presentation", "avg_exam"], axis=1)

df["final_result"] = df["final_result"].apply(lambda x: 0 if x == "Fail" else 1)

df.index = list(range(df.shape[0]))

y_col = "final_result"
bin_cols = list(set(df.nunique()[df.nunique() == 2].index) - set([y_col]))

mode_value = df["imd_band"].mode()[0]
df["imd_band"] = df["imd_band"].fillna(mode_value)

mean_value = df["avg_tma"].mean()
df["avg_tma"] = df["avg_tma"].fillna(mean_value)

mean_value = df["avg_cma"].mean()
df["avg_cma"] = df["avg_cma"].fillna(mean_value)

df[bin_cols] = df[bin_cols].apply(lambda x: pd.factorize(x)[0])

print(df.head())

df.to_csv("oulad_first_stage_preprocessed.csv")
