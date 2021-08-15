#-------------------Imports---------------------------
import os
import sys
from itertools import product

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
from aribabalancing import balance
from SMOTE import smote

sys.path.append(os.path.abspath('..'))

###======================Part 1: Code and Preprocessing Begins======================
base_path = str(sys.path[0])

input_file = base_path + '\\Data\\raw_state_CT.csv'
interm_file = base_path + '\\Data\\processed_scaled_state_CT.csv'
output_file = base_path + '\\Data\\debiased_state_CT.csv'


dataset_orig = pd.read_csv(input_file, dtype=object)
print('Data', dataset_orig.shape())
print(dataset_orig[['derived_msa-md', 'derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']])

###------------Scaling Function to be used later----------------------
def scale_dataset(processed_df):
    #####------------------Scaling------------------------------------
    scaler = MinMaxScaler()
    scaled_df = pd.DataFrame(scaler.fit_transform(processed_df), columns=processed_df.columns)

    return scaled_df



###------------------Preprocessing Function (includes Scaling)------------------------
def preprocessing(dataset_orig):
    # if you want 'derived_loan_product_type' column add here
    dataset_orig = dataset_orig[['derived_msa-md',  'derived_loan_product_type', 'derived_ethnicity', 'derived_race', 'derived_sex', 'purchaser_type', 'preapproval', 'loan_type', 'loan_purpose', 'lien_status', 'reverse_mortgage', 'open-end_line_of_credit', 'business_or_commercial_purpose', 'loan_amount', 'hoepa_status', 'negative_amortization', 'interest_only_payment', 'balloon_payment', 'other_nonamortizing_features', 'construction_method', 'occupancy_type', 'manufactured_home_secured_property_type', 'manufactured_home_land_property_interest', 'applicant_credit_score_type', 'co-applicant_credit_score_type', 'applicant_ethnicity-1', 'co-applicant_ethnicity-1', 'applicant_ethnicity_observed', 'co-applicant_ethnicity_observed', 'applicant_race-1', 'co-applicant_race-1', 'applicant_race_observed', 'co-applicant_race_observed', 'applicant_sex', 'co-applicant_sex', 'applicant_sex_observed', 'co-applicant_sex_observed', 'submission_of_application', 'initially_payable_to_institution', 'aus-1', 'denial_reason-1', 'tract_population', 'tract_minority_population_percent', 'ffiec_msa_md_median_family_income', 'tract_to_msa_income_percentage', 'tract_owner_occupied_units', 'tract_one_to_four_family_homes', 'tract_median_age_of_housing_units', 'action_taken']]


    # Below we are taking out rows in the dataset with values we do not care for. This is from lines 23 - 99.
    ###--------------------Sex------------------------
    dataset_orig = dataset_orig[(dataset_orig['derived_sex'] == 'Male') |
                                (dataset_orig['derived_sex'] == 'Female') |
                                (dataset_orig['derived_sex'] == 'Joint')]
    dataset_orig['derived_sex'] = dataset_orig['derived_sex'].replace(['Female', 'Male', 'Joint'],
                                                                      [0, 1, 2])
    print('sex: ' + str(dataset_orig.shape))
    ###-------------------Races-----------------------
    dataset_orig = dataset_orig[(dataset_orig['derived_race'] == 'White') |
                                (dataset_orig['derived_race'] == 'Black or African American') |
                                (dataset_orig['derived_race'] == 'Joint')]
    dataset_orig['derived_race'] = dataset_orig['derived_race'].replace(['Black or African American', 'White', 'Joint'],
                                                                        [0, 1, 2])
    print('race: ' + str(dataset_orig.shape))
    ####----------------Ethnicity-------------------
    dataset_orig = dataset_orig[(dataset_orig['derived_ethnicity'] == 'Hispanic or Latino') |
                                (dataset_orig['derived_ethnicity'] == 'Not Hispanic or Latino') |
                                (dataset_orig['derived_ethnicity'] == 'Joint')]
    dataset_orig['derived_ethnicity'] = dataset_orig['derived_ethnicity'].replace(['Hispanic or Latino', 'Not Hispanic or Latino', 'Joint'],
                                                                                  [0, 1, 2])
    print('ethnicity: ' + str(dataset_orig.shape))
    ###----------------Action_Taken-----------------
    dataset_orig = dataset_orig[(dataset_orig['action_taken'] == '1') |
                                (dataset_orig['action_taken'] == '2') |
                                (dataset_orig['action_taken'] == '3')]

    dataset_orig['action_taken'] = dataset_orig['action_taken'].replace(['1', '2', '3'],
                                                                        [1, 0, 0])

    ######----------------Loan Product-------------------
    # assigns each unique categorical value a unique integer id
    dataset_orig['derived_loan_product_type'] = dataset_orig['derived_loan_product_type'].astype('category').cat.codes

    print('loan product: ' + str(dataset_orig.shape))

    ####---------------Scale Dataset---------------
    dataset_orig = scale_dataset(dataset_orig)

    ##----------------Solve NAN problem-----------
    dataset_orig = dataset_orig.apply(pd.to_numeric)
    dataset_orig = dataset_orig.dropna()
    float_col = dataset_orig.select_dtypes(include=['float64'])

    for col in float_col.columns.values:
        dataset_orig[col] = dataset_orig[col].astype('int64')
    ####---------------Reset Indexes----------------
    dataset_orig.reset_index(drop=True, inplace=True)

    return dataset_orig


###---------Call the Function to create processed_scaled_df---------------
processed_scaled_df = preprocessing(dataset_orig)
print(processed_scaled_df[['derived_msa-md', 'derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']])
processed_scaled_shape = processed_scaled_df.shape

processed_scaled_df.to_csv(interm_file, index=True)

###===============Part 2: Working with the processed_scaled_df=================
class EmptyList(Exception):
    pass


# nhol --> not hispanic or latino
# hol --> hispanic or latino
# f --> female
# m --> male
# b --> black
# w --> white
# j --> joint

# # print(list(product(scaled_df['derived_ethnicity'].unique(), scaled_df['derived_race'].unique(), scaled_df['derived_sex'].unique())))
ind_cols = ['derived_ethnicity', 'derived_race', 'derived_sex']
dep_col = 'action_taken'


def split_dataset(processed_scaled_df, ind_cols):
    uniques = [processed_scaled_df[i].unique().tolist() for i in ind_cols]
    unique_df = pd.DataFrame(product(*uniques), columns=ind_cols)
    print(unique_df)
    combination_df = [pd.merge(processed_scaled_df, unique_df.iloc[[i]], on=ind_cols, how='inner') for i in
                      range(unique_df.shape[0])]

    return combination_df


combination_df = split_dataset(processed_scaled_df, ind_cols)

combination_names = ['nhol_w_m', 'nhol_w_f', 'nhol_w_j',
                     'nhol_j_m', 'nhol_j_f', 'nhol_j_j',
                     'nhol_b_m', 'nhol_b_f', 'nhol_b_j',
                     'hol_w_m', 'hol_w_f', 'hol_w_j',
                     'hol_j_m', 'hol_j_f', 'hol_j_j',
                     'hol_b_m', 'hol_b_f', 'hol_b_j',
                     'j_w_m', 'j_w_f', 'j_w_j',
                     'j_j_m', 'j_j_f', 'j_j_j',
                     'j_b_m', 'j_b_f','j_b_j']

# # uncomment to see which name is associated with which combination
for n, c in zip(combination_names, combination_df):
    print(n, c[['derived_msa-md', 'derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']])

# nhol_w_m = combination_df[0]
# nhol_w_f = combination_df[1]
# nhol_w_j = combination_df[2]
# nhol_j_m = combination_df[3]
# nhol_j_f = combination_df[4]
# nhol_j_j = combination_df[5]
# nhol_b_m = combination_df[6]
# nhol_b_f = combination_df[7]
# nhol_b_j = combination_df[8]
# hol_w_m = combination_df[9]
# hol_w_f = combination_df[10]
# hol_w_j = combination_df[11]
# hol_j_m = combination_df[12]
# hol_j_f = combination_df[13]
# hol_j_j = combination_df[14]
# hol_b_m = combination_df[15]
# hol_b_f = combination_df[16]
# hol_b_j = combination_df[17]
# j_w_m = combination_df[18]
# j_w_f = combination_df[19]
# j_w_j = combination_df[20]
# j_j_m = combination_df[21]
# j_j_f = combination_df[22]
# j_j_j = combination_df[23]
# j_b_m = combination_df[24]
# j_b_f = combination_df[25]
# j_b_j = combination_df[26]

max_val = 0
def get_max_val(max_val):
    for df in combination_df:
        temp1 = 0
        temp0 = 0
        print('The Distribution\n', df['action_taken'].value_counts())
        try:
            temp1 = len(df[(df['action_taken'] == 1)])
            temp0 = len(df[(df['action_taken'] == 0)])
        except:
            pass

        current_max = max(temp1, temp0)

        if(current_max > max_val):
            max_val = current_max

    return max_val

print("Here:\n", get_max_val(max_val))


####
def smote_balance(c):
    def apply_smote(df):
        df.reset_index(drop=True,inplace=True)
        cols = df.columns
        smt = smote(df)
        df = smt.run()
        df.columns = cols
        return df


    X_train, y_train = c.loc[:, c.columns != 'action_taken'], c['action_taken']

    train_df = X_train
    train_df['action_taken'] = y_train

    train_df = apply_smote(train_df)

    print('Next Distribution\n', train_df['action_taken'].value_counts())


classifiers = [smote_balance(c) for c in combination_df]
























