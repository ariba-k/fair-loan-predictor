#-------------------Imports---------------------------
import os
import sys
from itertools import product

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
from aribabalancing import balance

sys.path.append(os.path.abspath('..'))

###======================Part 1: Code and Preprocessing Begins======================
base_path = str(sys.path[0])

input_file = base_path + '\\Data\\raw_state_WY.csv'
interm_file = base_path + '\\Data\\processed_scaled_state_WY.csv'
# debug_file = base_path + '\\Data\\DebuggingDatasetArash.csv'
output_file = base_path + '\\Data\\debiased_state_WY.csv'


dataset_orig = pd.read_csv(input_file, dtype=object)


###------------Scaling Function to be used later----------------------
def scale_dataset(processed_df):
    #####------------------Scaling------------------------------------
    scaler = MinMaxScaler()
    scaled_df = pd.DataFrame(scaler.fit_transform(processed_df), columns=processed_df.columns)

    return scaled_df



###------------------Preprocessing Function (includes Scaling)------------------------
def preprocessing(dataset_orig):
    print('original data: ' + str(dataset_orig.shape))
    # if you want 'derived_loan_product_type' column add here
    dataset_orig = dataset_orig[['derived_msa-md',  'derived_loan_product_type', 'derived_ethnicity', 'derived_race', 'derived_sex', 'purchaser_type', 'preapproval', 'loan_type', 'loan_purpose', 'lien_status', 'reverse_mortgage', 'open-end_line_of_credit', 'business_or_commercial_purpose', 'loan_amount', 'hoepa_status', 'negative_amortization', 'interest_only_payment', 'balloon_payment', 'other_nonamortizing_features', 'construction_method', 'occupancy_type', 'manufactured_home_secured_property_type', 'manufactured_home_land_property_interest', 'applicant_credit_score_type', 'co-applicant_credit_score_type', 'applicant_ethnicity-1', 'co-applicant_ethnicity-1', 'applicant_ethnicity_observed', 'co-applicant_ethnicity_observed', 'applicant_race-1', 'co-applicant_race-1', 'applicant_race_observed', 'co-applicant_race_observed', 'applicant_sex', 'co-applicant_sex', 'applicant_sex_observed', 'co-applicant_sex_observed', 'submission_of_application', 'initially_payable_to_institution', 'aus-1', 'denial_reason-1', 'tract_population', 'tract_minority_population_percent', 'ffiec_msa_md_median_family_income', 'tract_to_msa_income_percentage', 'tract_owner_occupied_units', 'tract_one_to_four_family_homes', 'tract_median_age_of_housing_units', 'action_taken']]

    print('column selection: ' + str(dataset_orig.shape))

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
    print('action taken: ' + str(dataset_orig.shape))

    ######----------------Loan Product-------------------
    # assigns each unique categorical value a unique integer id
    dataset_orig['derived_loan_product_type'] = dataset_orig['derived_loan_product_type'].astype('category').cat.codes

    print('loan product: ' + str(dataset_orig.shape))

    ####---------------Scale Dataset---------------
    dataset_orig = scale_dataset(dataset_orig)
    ####---------------Reset Indexes----------------
    dataset_orig.reset_index(drop=True, inplace=True)

    return dataset_orig


###----------------Balancing Function-----------------

def balance(dataset_orig):
    if dataset_orig.empty:
        return dataset_orig
    # print('imbalanced data:\n', dataset_orig['action_taken'].value_counts())
    action_df = dataset_orig['action_taken'].value_counts()
    maj_label = action_df.index[0]
    min_label = action_df.index[-1]
    if(maj_label == min_label):
            return dataset_orig
    df_majority = dataset_orig[dataset_orig.action_taken == maj_label]
    df_minority = dataset_orig[dataset_orig.action_taken == min_label]

    df_majority_downsampled = resample(df_majority,
                                       replace=False,  # sample without replacement
                                       n_samples=len(df_minority.index),  # to match minority class
                                       random_state=123)
    # Combine minority class with downsampled majority class
    df_downsampled = pd.concat([df_majority_downsampled, df_minority])

    df_downsampled.reset_index(drop=True, inplace=True)

    # print('balanced data:\n', df_downsampled['action_taken'].value_counts())
    #
    # print('processed data: ' + str(df_downsampled.shape))

    return df_downsampled


###---------Call the Function to create processed_scaled_df---------------
processed_scaled_df = preprocessing(dataset_orig)
processed_scaled_shape = processed_scaled_df.shape
# processed_scaled_df = balance(processed_scaled_df)
# processed_scaled_balanced_shape = processed_scaled_df.shape
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
    combination_df = [balance(pd.merge(processed_scaled_df, unique_df.iloc[[i]], on=ind_cols, how='inner')) for i in
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
# for n, c in zip(combination_names, combination_df):
#     print(n, c[['derived_msa-md', 'derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']])

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


# Note: to get the string name of a variable --> f'{foo=}'.split('=')[0]

# check to make sure a column only contains allowed values
def check_values(df, allowed_values, column):
    if set(df[column].unique()) == set(allowed_values):
        return True


# check_values(nhol_w_m, [0.0, 1.0], 'action_taken')

##################################
# Classifier Function (helps to build classifiers):

# add this as parameter
def create_classifier(comb_df):
    min_rows = 15
    comb_df.reset_index(drop=True, inplace=True)
    # print(comb_df[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']].head(50))

    num_rows = len(comb_df)

    # comb_df.to_csv(debug_file)

    X_train, y_train = comb_df.loc[:, comb_df.columns != 'action_taken'], comb_df['action_taken'] #Might be a problem with how the dataset is unordered due to how we do balancing BTW
    # --- LSR
    clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=200)
    if num_rows >= min_rows and check_values(comb_df, [0.0, 1.0], 'action_taken'):
        return clf.fit(X_train, y_train)
    else:
        return None


##################################
# --------------------Classifiers ---------------------
# clf0 = createClassifier(allFemaleDataset)
# clf1 = create_classifier(nhol_b_f)
# clf2 = create_classifier(hol_b_f)
# clf3 = create_classifier(j_b_f)
# clf4 = create_classifier(nhol_w_f)
# clf5 = create_classifier(hol_w_f)
# clf6 = create_classifier(j_w_f)
# clf7 = create_classifier(nhol_j_f)
# clf8 = create_classifier(hol_j_f)
# clf9 = create_classifier(j_j_f)
# clf10 = create_classifier(nhol_b_m)
# clf11 = create_classifier(hol_b_m)
# clf12 = create_classifier(j_b_m)
# clf13 = create_classifier(nhol_w_m)
# clf14 = create_classifier(hol_w_m)
# clf15 = create_classifier(j_w_m)
# clf16 = create_classifier(nhol_j_m)
# clf17 = create_classifier(hol_j_m)
# clf18 = create_classifier(j_j_m)
# clf19 = create_classifier(nhol_b_j)
# clf20 = create_classifier(hol_b_j)
# clf21 = create_classifier(j_b_j)
# clf22 = create_classifier(nhol_w_j)
# clf23 = create_classifier(hol_w_j)
# clf24 = create_classifier(j_w_j)
# clf25 = create_classifier(nhol_j_j)
# clf26 = create_classifier(hol_j_j)
# clf27 = create_classifier(j_j_j)

classifiers = [create_classifier(c) for c in combination_df]
print('Classifiers\n:', classifiers)

# -----------Make Debiased Dataset (Remove Biased Points)------------


def debias_dataset(dataset_orig, classifiers):
    for index, row in dataset_orig.iterrows():
        true_y = row[-1]
        row = [row.values[0:-1]]
        pred_y = [None if c is None else c.predict(row)[0] for c in classifiers]
        filter_pred_y = list(filter(lambda x: x is not None, pred_y))

        print('True Y Label:', true_y)
        print('Predicted Y Labels:', pred_y)
        print('Filtered Predicted Y Labels', filter_pred_y)

        num_unique_vals = len(set(filter_pred_y))

        print(num_unique_vals)

        if num_unique_vals > 1:
            dataset_orig = dataset_orig.drop(index)
        elif num_unique_vals == 0:
            raise EmptyList

    dataset_orig.reset_index(drop=True, inplace=True)
    return dataset_orig


debias_df = debias_dataset(processed_scaled_df, classifiers)
debias_df_shape = debias_df.shape

print('processed_scaled_shape\n', processed_scaled_shape)
# print('processed_scaled_balanced_shape\n', processed_scaled_balanced_shape)
print('debias_shape\n', debias_df_shape)

# debias_balance_df = balance(debias_df)
# debias_balance_shape = debias_balance_df.shape
# print('Debias Balance Shape:\n', debias_balance_shape)

debias_df.to_csv(output_file, index=True)


























