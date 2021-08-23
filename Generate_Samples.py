from __future__ import print_function, division
import pdb
import unittest
import random
from collections import Counter
import pandas as pd
import numpy as np
from scipy.spatial import distance as dist
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors as NN


def get_ngbr(df, knn):
    rand_sample_idx = random.randint(0, df.shape[0] - 1)
    parent_candidate = df.iloc[rand_sample_idx]
    ngbr = knn.kneighbors(parent_candidate.values.reshape(1, -1), 3, return_distance=False)
    candidate_1 = df.iloc[ngbr[0][0]]
    candidate_2 = df.iloc[ngbr[0][1]]
    candidate_3 = df.iloc[ngbr[0][2]]
    return parent_candidate, candidate_2, candidate_3


def generate_samples(no_of_samples_zeros, no_of_samples_ones, df, df_name):
    total_data_zero = []
    total_data_one = []
    knn = NN(n_neighbors=5, algorithm='auto').fit(df)
    count_zero = 0
    count_one = 0
    while (count_one < no_of_samples_ones or count_zero < no_of_samples_zeros):
        cr = 0.8
        f = 0.8
        parent_candidate, child_candidate_1, child_candidate_2 = get_ngbr(df, knn)
        new_candidate = []
        for key, value in parent_candidate.items():
            if isinstance(parent_candidate[key], bool):
                new_candidate.append(parent_candidate[key] if cr < random.random() else not parent_candidate[key])
            elif isinstance(parent_candidate[key], str):
                new_candidate.append(
                    random.choice([parent_candidate[key], child_candidate_1[key], child_candidate_2[key]]))
            elif isinstance(parent_candidate[key], list):
                temp_lst = []
                for i, each in enumerate(parent_candidate[key]):
                    temp_lst.append(parent_candidate[key][i] if cr < random.random() else
                                    int(parent_candidate[key][i] +
                                        f * (child_candidate_1[key][i] - child_candidate_2[key][i])))
                new_candidate.append(temp_lst)
            else:
                new_candidate.append(abs(parent_candidate[key] + f * (child_candidate_1[key] - child_candidate_2[key])))
        if(new_candidate[-1] == 0 and count_zero < no_of_samples_zeros):
            total_data_zero.append(new_candidate)
            count_zero += 1
        elif(new_candidate[-1] == 1 and count_one < no_of_samples_ones):
            total_data_one.append(new_candidate)
            count_one += 1

    final_df_zero = pd.DataFrame(total_data_zero)
    final_df_one = pd.DataFrame(total_data_one)
    if df_name == 'HMDA':
        column_array = ['derived_msa-md', 'derived_loan_product_type', 'derived_ethnicity', 'derived_race',
                        'derived_sex', 'purchaser_type', 'preapproval', 'loan_type', 'loan_purpose', 'lien_status',
                        'reverse_mortgage', 'open-end_line_of_credit', 'business_or_commercial_purpose', 'loan_amount',
                        'hoepa_status', 'negative_amortization', 'interest_only_payment', 'balloon_payment',
                        'other_nonamortizing_features', 'construction_method', 'occupancy_type',
                        'manufactured_home_secured_property_type', 'manufactured_home_land_property_interest',
                        'applicant_credit_score_type', 'co-applicant_credit_score_type', 'applicant_ethnicity-1',
                        'co-applicant_ethnicity-1', 'applicant_ethnicity_observed', 'co-applicant_ethnicity_observed',
                        'applicant_race-1', 'co-applicant_race-1', 'applicant_race_observed',
                        'co-applicant_race_observed', 'applicant_sex', 'co-applicant_sex', 'applicant_sex_observed',
                        'co-applicant_sex_observed', 'submission_of_application', 'initially_payable_to_institution',
                        'aus-1', 'denial_reason-1', 'tract_population', 'tract_minority_population_percent',
                        'ffiec_msa_md_median_family_income', 'tract_to_msa_income_percentage',
                        'tract_owner_occupied_units', 'tract_one_to_four_family_homes',
                        'tract_median_age_of_housing_units', 'action_taken']
        final_df_zero = final_df_zero.rename(columns={c: column_array[c] for c in range(len(column_array))}, errors="raise")
        final_df_one = final_df_one.rename(columns={c: column_array[c] for c in range(len(column_array))},errors="raise")
    # if df_name == 'Adult':
    #     final_df = final_df.rename(
    #         columns={0: "age", 1: "education-num", 2: "race", 3: "sex", 4: "capital-gain", 5: "capital-loss",
    #                  6: "hours-per-week", 7: "Probability"}, errors="raise")
    # print(final_df_one)
    # print(final_df_zero)
    return final_df_zero, final_df_one
