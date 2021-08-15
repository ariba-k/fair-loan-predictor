import os
import sys

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

sys.path.append(os.path.abspath('../..'))

from otherSMOTE import smote
from Measure import measure_final_score
from Generate_Samples import generate_samples
###======================Part 1: Code and Preprocessing Begins======================

base_path = str(sys.path[0])

input_file = base_path + '\\Data\\raw_state_WY.csv'
interm_file = base_path + '\\Data\\processed_scaled_state_WY.csv'
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

processed_scaled_df = preprocessing(dataset_orig)
processed_scaled_shape = processed_scaled_df.shape

processed_scaled_df.to_csv(interm_file, index=True)

dataset_orig_train, dataset_orig_test = train_test_split(processed_scaled_df, test_size=0.2,shuffle = True)

# dataset_orig
#------------- Check original scores--------------------------

X_train, y_train = dataset_orig_train.loc[:, dataset_orig_train.columns != 'action_taken'], dataset_orig_train['action_taken']
X_test , y_test = dataset_orig_test.loc[:, dataset_orig_test.columns != 'action_taken'], dataset_orig_test['action_taken']

clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100) # LSR


arrayDatasets = [
                'allBFNHOLdataset',
                'allBFHOLdataset',
                'allBFJointEthnicitydataset',
                'allWFNHOLdataset',
                'allWFHOLdataset',
                'allWFJointEthnicitydataset',
                'allJointRaceFemaleNHOLdataset',
                'allJointRaceFemaleHOLdataset',
                'allJointRaceFemaleJointEthnicitydataset',
                'allBMNHOLdataset',
                'allBMHOLdataset',
                'allBMJointEthnicitydataset',
                'allWMNHOLdataset',
                'allWMHOLdataset',
                'allWMJointEthnicitydataset',
                'allJointRaceMaleNHOLdataset',
                'allJointRaceMaleHOLdataset',
                'allJointRaceMaleJointEthnicitydataset',
                'allJointSexBlacksNHOLdataset',
                'allJointSexBlacksHOLdataset',
                'allJointSexBlacksJointEthnicitydataset',
                'allJointSexWhitesNHOLdataset',
                'allJointSexWhitesHOLdataset',
                'allJointSexWhitesJointEthnicitydataset',
                'allJointSexJointRaceNHOLdataset',
                'allJointSexJointRaceHOLdataset',
                'allJointSexJointRaceJointEthnicitydataset'
                 ]

sexCArray = [0,0,0,0,0,0,0,0,0,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,1,1,1,1,1,1,1,1,1]
raceCArray = [0,0,0,0.5,0.5,0.5,1,1,1,0,0,0,0.5,0.5,0.5,1,1,1,0,0,0,0.5,0.5,0.5,1,1,1]
ethnicityCArray = [0.5, 0, 1, 0.5, 0, 1, 0.5, 0, 1, 0.5, 0, 1, 0.5, 0, 1, 0.5, 0, 1, 0.5, 0, 1, 0.5, 0, 1, 0.5, 0, 1]

print("mAOD :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, arrayDatasets, sexCArray, raceCArray, ethnicityCArray, 'mAOD'))
print("mEOD :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, arrayDatasets, sexCArray, raceCArray, ethnicityCArray, 'mEOD'))

## Might be a problem with mEOD and mAOD in general instead of using EOD sex -- look into


# Check SMOTE Scores

def apply_smote(df):
    df.reset_index(drop=True,inplace=True)
    cols = df.columns
    smt = smote(df)
    df = smt.run()
    df.columns = cols
    return df

# dataset_orig_train, dataset_orig_test = train_test_split(dataset_orig, test_size=0.2, random_state=0,shuffle = True)

X_train, y_train = dataset_orig_train.loc[:, dataset_orig_train.columns != 'action_taken'], dataset_orig_train['action_taken']
X_test , y_test = dataset_orig_test.loc[:, dataset_orig_test.columns != 'action_taken'], dataset_orig_test['action_taken']

train_df = X_train
train_df['action_taken'] = y_train

train_df = apply_smote(train_df)

y_train = train_df.action_taken
X_train = train_df.drop('action_taken', axis = 1)

clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100) # LSR

print("mAOD :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, arrayDatasets, sexCArray, raceCArray, ethnicityCArray, 'mAOD'))
print("mEOD :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, arrayDatasets, sexCArray, raceCArray, ethnicityCArray, 'mEOD'))




# first one is class value and second one is protected attribute value
zero_zero = len(dataset_orig_train[(dataset_orig_train['action_taken'] == 0) & (dataset_orig_train[protected_attribute] == 0)])
zero_one = len(dataset_orig_train[(dataset_orig_train['action_taken'] == 0) & (dataset_orig_train[protected_attribute] == 1)])
one_zero = len(dataset_orig_train[(dataset_orig_train['action_taken'] == 1) & (dataset_orig_train[protected_attribute] == 0)])
one_one = len(dataset_orig_train[(dataset_orig_train['action_taken'] == 1) & (dataset_orig_train[protected_attribute] == 1)])

print(zero_zero,zero_one,one_zero,one_one)




maximum = max(zero_zero,zero_one,one_zero,one_one)
if maximum == zero_zero:
    print("zero_zero is maximum")
if maximum == zero_one:
    print("zero_one is maximum")
if maximum == one_zero:
    print("one_zero is maximum")
if maximum == one_one:
    print("one_one is maximum")

zero_zero_to_be_incresed = maximum - zero_zero ## where both are 0
one_zero_to_be_incresed = maximum - one_zero ## where class is 1 attribute is 0
one_one_to_be_incresed = maximum - one_one ## where class is 1 attribute is 1

print(zero_zero_to_be_incresed,one_zero_to_be_incresed,one_one_to_be_incresed)

df_zero_zero = dataset_orig_train[(dataset_orig_train['action_taken'] == 0) & (dataset_orig_train[protected_attribute] == 0)]
df_one_zero = dataset_orig_train[(dataset_orig_train['action_taken'] == 1) & (dataset_orig_train[protected_attribute] == 0)]
df_one_one = dataset_orig_train[(dataset_orig_train['action_taken'] == 1) & (dataset_orig_train[protected_attribute] == 1)]

df_zero_zero['race'] = df_zero_zero['race'].astype(str)
df_zero_zero['sex'] = df_zero_zero['sex'].astype(str)


df_one_zero['race'] = df_one_zero['race'].astype(str)
df_one_zero['sex'] = df_one_zero['sex'].astype(str)

df_one_one['race'] = df_one_one['race'].astype(str)
df_one_one['sex'] = df_one_one['sex'].astype(str)


df_zero_zero = generate_samples(zero_zero_to_be_incresed,df_zero_zero,'Adult')
df_one_zero = generate_samples(one_zero_to_be_incresed,df_one_zero,'Adult')
df_one_one = generate_samples(one_one_to_be_incresed,df_one_one,'Adult')


df = df_zero_zero.append(df_one_zero)
df = df.append(df_one_one)

df['race'] = df['race'].astype(float)
df['sex'] = df['sex'].astype(float)

df_zero_one = dataset_orig_train[(dataset_orig_train['action_taken'] == 0) & (dataset_orig_train[protected_attribute] == 1)]
df = df.append(df_zero_one)


X_train, y_train = df.loc[:, df.columns != 'action_taken'], df['action_taken']
X_test , y_test = dataset_orig_test.loc[:, dataset_orig_test.columns != 'action_taken'], dataset_orig_test['action_taken']

clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100) # LSR


print("mAOD :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, arrayDatasets, sexCArray, raceCArray, ethnicityCArray, 'mAOD'))
print("mEOD :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, arrayDatasets, sexCArray, raceCArray, ethnicityCArray, 'mEOD'))



# first one is class value and second one is protected attribute value
zero_zero = len(df[(df['action_taken'] == 0) & (df[protected_attribute] == 0)])
zero_one = len(df[(df['action_taken'] == 0) & (df[protected_attribute] == 1)])
one_zero = len(df[(df['action_taken'] == 1) & (df[protected_attribute] == 0)])
one_one = len(df[(df['action_taken'] == 1) & (df[protected_attribute] == 1)])

print(zero_zero,zero_one,one_zero,one_one)

X_train, y_train = df.loc[:, df.columns != 'action_taken'], df['action_taken']

clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100)
clf.fit(X_train,y_train)
removal_list = []

for index,row in df.iterrows():
    row_ = [row.values[0:len(row.values)-1]]
    y_normal = clf.predict(row_)
    # Here protected attribute value gets switched
    if row_[0][3] == 0: ## index of Sex is 3, Race is 2
        row_[0][3] = 1
    else:
        row_[0][3] = 0
    y_reverse = clf.predict(row_)
    if y_normal[0] != y_reverse[0]:
        removal_list.append(index)

removal_list = set(removal_list)
print(len(removal_list))


print(df.shape)
df_removed = pd.DataFrame(columns=df.columns)

for index,row in df.iterrows():
    if index in removal_list:
        df_removed = df_removed.append(row, ignore_index=True)
        df = df.drop(index)
print(df.shape)

# first one is class value and second one is protected attribute value
zero_zero = len(df[(df['action_taken'] == 0) & (df[protected_attribute] == 0)])
zero_one = len(df[(df['action_taken'] == 0) & (df[protected_attribute] == 1)])
one_zero = len(df[(df['action_taken'] == 1) & (df[protected_attribute] == 0)])
one_one = len(df[(df['action_taken'] == 1) & (df[protected_attribute] == 1)])

print(zero_zero,zero_one,one_zero,one_one)

# Check Score after Removal

X_train, y_train = df.loc[:, df.columns != 'action_taken'], df['action_taken']
X_test , y_test = dataset_orig_test.loc[:, dataset_orig_test.columns != 'action_taken'], dataset_orig_test['action_taken']

clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100)




print("mAOD :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, arrayDatasets, sexCArray, raceCArray, ethnicityCArray, 'mAOD'))
print("mEOD :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, arrayDatasets, sexCArray, raceCArray, ethnicityCArray, 'mEOD'))
