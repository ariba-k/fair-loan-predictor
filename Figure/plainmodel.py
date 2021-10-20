#-------------------Imports---------------------------

import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

input_file = r'C:\Users\jasha\Documents\GitHub\fair-loan-predictor\Data\raw_state_CT.csv'
dataset_orig = pd.read_csv(input_file, dtype=object)


def scale_dataset(processed_df):
    #####------------------Scaling------------------------------------
    scaler = MinMaxScaler()
    scaled_df = pd.DataFrame(scaler.fit_transform(processed_df), columns=processed_df.columns)

    return scaled_df


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
    print(dataset_orig[['derived_msa-md', 'derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']])

    ###-------------------Races-----------------------
    dataset_orig = dataset_orig[(dataset_orig['derived_race'] == 'White') |
                                (dataset_orig['derived_race'] == 'Black or African American') |
                                (dataset_orig['derived_race'] == 'Joint')]
    dataset_orig['derived_race'] = dataset_orig['derived_race'].replace(['Black or African American', 'White', 'Joint'],
                                                                        [0, 1, 2])
    print('race: ' + str(dataset_orig.shape))
    print(dataset_orig[['derived_msa-md', 'derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']])

    ####----------------Ethnicity-------------------
    dataset_orig = dataset_orig[(dataset_orig['derived_ethnicity'] == 'Hispanic or Latino') |
                                (dataset_orig['derived_ethnicity'] == 'Not Hispanic or Latino') |
                                (dataset_orig['derived_ethnicity'] == 'Joint')]
    dataset_orig['derived_ethnicity'] = dataset_orig['derived_ethnicity'].replace(['Hispanic or Latino', 'Not Hispanic or Latino', 'Joint'],
                                                                                  [0, 1, 2])
    print('ethnicity: ' + str(dataset_orig.shape))
    print(dataset_orig[['derived_msa-md', 'derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']])

    ###----------------Action_Taken-----------------
    dataset_orig = dataset_orig[(dataset_orig['action_taken'] == '1') |
                                (dataset_orig['action_taken'] == '2') |
                                (dataset_orig['action_taken'] == '3')]

    dataset_orig['action_taken'] = dataset_orig['action_taken'].replace(['1', '2', '3'],
                                                                        [1, 0, 0])

    print(dataset_orig[['derived_msa-md', 'derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']])

    ######----------------Loan Product-------------------
    # assigns each unique categorical value a unique integer id
    dataset_orig['derived_loan_product_type'] = dataset_orig['derived_loan_product_type'].astype('category').cat.codes

    print('loan product: ' + str(dataset_orig.shape))
    print(dataset_orig[['derived_msa-md', 'derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']])

    ####---------------Scale Dataset---------------
    dataset_orig = scale_dataset(dataset_orig)

    print(dataset_orig[['derived_msa-md', 'derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']])

    ##----------------Solve NAN problem-----------

    dataset_orig = dataset_orig.apply(pd.to_numeric)
    dataset_orig = dataset_orig.dropna()
    float_col = dataset_orig.select_dtypes(include=['float64'])

    # for col in float_col.columns.values:
    #     dataset_orig[col] = dataset_orig[col].astype('int64')
    ####---------------Reset Indexes----------------
    dataset_orig.reset_index(drop=True, inplace=True)

    return dataset_orig


processed_scaled_df = preprocessing(dataset_orig)
ds_train, ds_test = train_test_split(processed_scaled_df, test_size=0.3, random_state=0,shuffle =True)

X_train, y_train = ds_train.loc[:, ds_train.columns != 'action_taken'], ds_train['action_taken']
X_test , y_test = ds_test.loc[:, ds_test.columns != 'action_taken'], ds_test['action_taken']

clf = .

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
cnf_matrix = confusion_matrix(y_test, y_pred)
from sklearn.metrics import f1_score
f1 = f1_score(y_test, y_pred)
print('HEY, this is f1_score:', f1)
TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()

def calculate_accuracy(TP,FP,FN,TN):
    return (TP + TN)/(TP + TN + FP + FN)

print("This is second Accuracy", calculate_accuracy(TP,FP,FN,TN))
print('THIS IS ACCURARCY:', clf.score(X_test, y_test))
