#-------------------Imports---------------------------
import copy
import os
import sys

import numpy as np
from itertools import product

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample


from SMOTE import smote
from Generate_Samples import generate_samples
from Delete_Samples import delete_samples
from Measure import measure_final_score
from Usage import arrayDatasets, sexCArray, raceCArray, ethnicityCArray

sys.path.append(os.path.abspath('..'))

#Before we start, I want the person using thie code to understand that this code will not work if you do not have a large dataset enough to have values for both 0 and 1 for all 27 DATAFRAMES

###======================Part 1: Code and Preprocessing Begins======================
base_path = str(sys.path[0])

input_file = base_path + '\\Data\\raw_state_CA.csv'
interm_file = base_path + '\\Data\\FirstBalancedCA.csv'
process_scale_file = base_path + '\\Data\\processedscaledCANOW.csv'
other_file = base_path + '\\Data\\newDatasetOrig.csv'
output_file = base_path + '\\Data\\DoubleBalancedCA.csv'


dataset_orig = pd.read_csv(input_file, dtype=object)
print('Data', dataset_orig.shape)
print(dataset_orig[['derived_msa-md', 'derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']])

#switch action_taken to last column 
action_taken_col = dataset_orig.pop('action_taken')
dataset_orig.insert(len(dataset_orig.columns), 'action_taken', action_taken_col)

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


###---------Call the Function to create processed_scaled_df---------------
processed_scaled_df = preprocessing(dataset_orig)
print(processed_scaled_df[['derived_msa-md', 'derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']])
processed_scaled_shape = processed_scaled_df.shape

processed_scaled_df.to_csv(process_scale_file, index=False)
print('NOWWWWWWWWWWWWWWW P')

##------------------Check beginning Measures----------------------

processed_scaled_df["derived_sex"] = pd.to_numeric(processed_scaled_df.derived_sex, errors='coerce')
processed_scaled_df["derived_race"] = pd.to_numeric(processed_scaled_df.derived_race, errors='coerce')
processed_scaled_df["derived_ethnicity"] = pd.to_numeric(processed_scaled_df.derived_ethnicity, errors='coerce')
processed_scaled_df["action_taken"] = pd.to_numeric(processed_scaled_df.action_taken, errors='coerce')


print(processed_scaled_df.shape)
np.random.seed(0)
# Divide into train,validation,test

processed_scaled_train, processed_and_scaled_test = train_test_split(processed_scaled_df, test_size=0.2, random_state=0,shuffle = True)
print(processed_scaled_train)
print(processed_and_scaled_test)
X_train, y_train = processed_scaled_train.loc[:, processed_scaled_train.columns != 'action_taken'], processed_scaled_train['action_taken']
X_test , y_test = processed_and_scaled_test.loc[:, processed_and_scaled_test.columns != 'action_taken'], processed_and_scaled_test['action_taken']

# --- LSR
clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100)

print("mAOD:",measure_final_score(processed_and_scaled_test, clf, X_train, y_train, X_test, y_test, arrayDatasets, sexCArray, raceCArray, ethnicityCArray, 'mAOD'))
print("mEOD:",measure_final_score(processed_and_scaled_test, clf, X_train, y_train, X_test, y_test, arrayDatasets, sexCArray, raceCArray, ethnicityCArray,'mEOD'))
##-----------------------------------------------------------------------------------------------------------------------------------------------------





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

def get_unique_df(processed_scaled_df,ind_cols):
    uniques = [processed_scaled_df[i].unique().tolist() for i in ind_cols]
    unique_df = pd.DataFrame(product(*uniques), columns=ind_cols)
    return unique_df


def split_dataset(processed_scaled_df, ind_cols):
    unique_df = get_unique_df(processed_scaled_df, ind_cols)
    combination_df = [pd.merge(processed_scaled_df, unique_df.iloc[[i]], on=ind_cols, how='inner') for i in
                      range(unique_df.shape[0])]

    return combination_df

global_unique_df = get_unique_df(processed_scaled_df, ind_cols)
print(global_unique_df)
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

max_val = 0
num_of_combs = 54

def get_mean_val():
    current_total = 0
    for df in combination_df:
        temp1 = 0
        temp0 = 0
        try:
            temp1 = len(df[(df['action_taken'] == 1)])
            temp0 = len(df[(df['action_taken'] == 0)])
        except:
            pass

        current_total = current_total + temp1 + temp0

    mean_val = (current_total/num_of_combs)

    return round(mean_val)

mean_val = 500 #THIS IS HYPER
print("Here is the aim:", mean_val)
####
def RUS_balance(dataset_orig):
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

    train_df["action_taken"] = y_train.astype("category")

    train_df = apply_smote(train_df)


    return train_df

def do_balancing_procedure(dataframe):
    combination_df = split_dataset(dataframe, ind_cols)
    smoted_list =[]
    RUS_list = []
    for c in combination_df:
        num1 = 0
        num0 = 0
        try:
            num1 = len(c[(c['action_taken'] == 1)])
            num0 = len(c[(c['action_taken'] == 0)])
        except:
            pass

        # print("NOW HERE aslkd:", num1, num0, c[['derived_race', 'derived_sex', 'derived_ethnicity','action_taken']].head(10))
        current_max = max(num1, num0)
        current_min = min(num1, num0)

        if(current_max <  mean_val):
            print('current_max', current_max)
            smoted_df = smote_balance(c)
            smoted_list.append(smoted_df)
        elif((current_max > mean_val) and (current_min < mean_val)):
            print('current_max2', current_max, current_min)
            diff_to_max = current_max - mean_val
            diff_to_min = mean_val - current_min
            if(diff_to_max < diff_to_min):
                smoted_df = smote_balance(c)
                RUS_list.append(smoted_df)
            else:
                RUS_df = RUS_balance(c)
                smoted_list.append(RUS_df) ##don't assign to correct list?
        elif((current_max > mean_val) and (current_min > mean_val)):
            print('current_max3', current_max, current_min)
            RUS_df = RUS_balance(c)
            RUS_list.append(RUS_df)

    # print('smoted', smoted_list)
    # print('RUS list', RUS_list)
    super_balanced_RUS = []
    for df in RUS_list:
        temp_val_0 = len(df[(df['action_taken'] == 0)])
        temp_val_1 = len(df[(df['action_taken'] == 1)])

        num_decrease_of_0 = temp_val_0 - mean_val
        num_decrease_of_1 = temp_val_1 - mean_val

        print('Before Distribution\n', df['action_taken'].value_counts())

        df = delete_samples(num_decrease_of_0, df, 0) ##@params = {value_of_increase, dataset_name, only_this_action_taken_value}
        df = delete_samples(num_decrease_of_1, df, 1)

        print('After Distribution\n', df['action_taken'].value_counts())
        super_balanced_RUS.append(df)


    super_balanced_smote = []
    # count = 0
    for df in smoted_list:
        # count = count + 1
        # print('WE ARE ON THE', count, 'DATASET')
        temp_val_0 = len(df[(df['action_taken'] == 0)])
        temp_val_1 = len(df[(df['action_taken'] == 1)])

        num_increase_of_0 = mean_val - temp_val_0
        num_increase_of_1 = mean_val - temp_val_1

        print("This is the num of increase:", num_increase_of_0, num_increase_of_1)
        print('Before Distribution\n', df['action_taken'].value_counts())

        df_zeros, df_ones = generate_samples(num_increase_of_0, num_increase_of_1, df, 'HMDA') ##@params = {value_of_increase, dataframe, dataset_name, only_this_action_taken_value}
        df_added = pd.concat([df_zeros, df_ones])
        concat_df = pd.concat([df, df_added])
        concat_df = concat_df.sample(frac=1).reset_index(drop=True)
        print('After Distribution\n', concat_df['action_taken'].value_counts())
        super_balanced_smote.append(concat_df)

    def concat_and_shuffle(smote_version, RUS_version):
        concat_smote_df = pd.concat(smote_version)
        concat_RUS_df = pd.concat(RUS_version)
        tempArray = [concat_RUS_df, concat_smote_df]
        total_concat_df = pd.concat(tempArray)
        total_concat_df = total_concat_df.sample(frac=1).reset_index(drop=True)

        print('shuffle:', total_concat_df.head(50))
        return total_concat_df

    new_dataset_orig = concat_and_shuffle(super_balanced_smote, super_balanced_RUS)
    return new_dataset_orig

new_dataset_orig = do_balancing_procedure(processed_scaled_df)
new_dataset_orig.to_csv(other_file, index=False)


#-----------------Situation Testing-------------------
X_train, y_train = new_dataset_orig.loc[:, new_dataset_orig.columns != 'action_taken'], new_dataset_orig['action_taken']

clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100)
clf.fit(X_train,y_train)

removal_list = []

##Gets you bad rows that need to be deleted
for index,row in new_dataset_orig.iterrows():
    pred_list = []
    row_ = [row.values[0:len(row.values) - 1]]
    for other_index, other_row in global_unique_df.iterrows():
        current_comb = other_row.values[0:len(other_row.values)]  ## indexes are 2, 3, 4 for ethnicity, race, sex respectively
        print('current_com', current_comb)
        original_ethnic, original_race, original_sex = row_[0][2], row_[0][3], row_[0][4]
        row_[0][2] = current_comb[0] ##don't know confirm what this is
        row_[0][3] = current_comb[1]
        row_[0][4] = current_comb[2]
        y_current_pred = clf.predict(row_)[0]
        pred_list.append(y_current_pred)
        print('pred_list', pred_list)
        row_[0][2], row_[0][3], row_[0][4] = original_ethnic, original_race, original_sex

    print('DONE HERE 1')
    num_unique_vals = len(set(pred_list))

    print('Num Unique Values:', num_unique_vals)

    if num_unique_vals > 1:
        removal_list.append(index)
    elif num_unique_vals == 0:
        raise EmptyList

removal_list = set(removal_list)
print('length of removal:', len(removal_list))

print('YEAH IM REMOVAL', removal_list)
print(new_dataset_orig.shape)
df_removed = pd.DataFrame(columns=new_dataset_orig.columns)


for index,row in new_dataset_orig.iterrows():
    if index in removal_list:
        df_removed = df_removed.append(row, ignore_index=True)
        balanced_and_situation_df = new_dataset_orig.drop(index)
        print(balanced_and_situation_df.shape)

df_removed.to_csv(r'C:\Users\jasha\Documents\GitHub\fair-loan-predictor\Data\hijashan.csv')
print(df_removed)



# print('Final Distribution1\n', balanced_and_situation_df['action_taken'].value_counts())
print('XXXXXXXXTRAINNNNNNNNNNNN', X_train, y_train)


##--------------------------Get Final Measures----------------------------
def get_metrics(df):
    df["derived_sex"] = pd.to_numeric(df.derived_sex, errors='coerce')
    df["derived_race"] = pd.to_numeric(df.derived_race, errors='coerce')
    df["derived_ethnicity"] = pd.to_numeric(df.derived_ethnicity, errors='coerce')
    df["action_taken"] = pd.to_numeric(df.action_taken, errors='coerce')


    print(df.shape)
    np.random.seed(0)
    # Divide into train,validation,test

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=0,shuffle = True)
    print(df_train)
    print(df_test)
    X_train, y_train = df_train.loc[:, df_train.columns != 'action_taken'], df_train['action_taken']
    X_test , y_test = df_test.loc[:, df_test.columns != 'action_taken'], df_test['action_taken']


    clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100)

    print("mAOD:",measure_final_score(df_test, clf, X_train, y_train, X_test, y_test, arrayDatasets, sexCArray, raceCArray, ethnicityCArray, 'mAOD'))
    print("mEOD:",measure_final_score(df_test, clf, X_train, y_train, X_test, y_test, arrayDatasets, sexCArray, raceCArray, ethnicityCArray,'mEOD'))

balanced_and_situation_df.to_csv(interm_file)
get_metrics(balanced_and_situation_df)
#-----------------------------------Second Balancing--------------------------------
# othernew_dataset_orig = do_balancing_procedure(balanced_and_situation_df)
#
# othernew_dataset_orig.to_csv(output_file, index=False)
# get_metrics(othernew_dataset_orig)

