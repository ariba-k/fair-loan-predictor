# -------------------Imports---------------------------
import os
import sys
import numpy as np
import pandas as pd
import statistics
import copy,math

from statistics import median
from itertools import product

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix

from SMOTE import smote
from Generate_Samples import generate_samples
from Delete_Samples import delete_samples
from Measure import measure_new_eod, measure_new_aod

sys.path.append(os.path.abspath('../..'))


# Custom Exceptions
class EmptyList(Exception):
    pass


# Dataset Used Needs to be Large Enough to Have Data for all 27 Subsets

###======================Part 1: Code and Preprocessing Begins======================
base_path = str(sys.path[0])

input_file = base_path + '/Data/raw_state_NV.csv'
input_file_1 = base_path + '/Data/HMDA_2020_Data.csv'
input_file_2 = base_path + '/Data/HMDA_2019_Data.csv'
input_file_3 = base_path + '/Data/HMDA_2018_Data.csv'
# interm_file = base_path + '\\Data\\FirstBalancedCA.csv'
add_file = base_path + '/Data/addition_file.csv'
add_file_2 = base_path + '/Data/new_addition_file.csv'
final_file = base_path + '/Data/All_HMDA_Debiased.csv'
# process_scale_file = base_path + '\\Data\\processedscaledCANOW.csv'
process_scale_file = base_path + '/Data/processedscaledCANOW.csv'
# other_file = base_path + '\\Data\\newDatasetOrig.csv'
other_file = base_path + '/Data/newDatasetOrig.csv'
# output_file = base_path + '\\Data\\DoubleBalancedCA.csv'
output_file = base_path + '/Data/DoubleBalancedCA.csv'
result_file = base_path + '/Results/CT_results.csv'


# print("I'M INPUT:", input_file)
print("I'M INPUT:", input_file_1, input_file_2, input_file_3)
# print("Yeah, the add file is here", add_file)
# print("Yello:", add_file_2)
# dataset_orig = pd.read_csv(input_file, dtype=object)
df_2020 = pd.read_csv(input_file_1, dtype=object).sample(n=755000)
df_2019 = pd.read_csv(input_file_2, dtype=object).sample(n=755000)
df_2018 = pd.read_csv(input_file_3, dtype=object).sample(n=755000)
dataset_orig = pd.concat([df_2020, df_2019, df_2018])
dataset_orig = dataset_orig.sample(frac=1)
dataset_orig.reset_index(drop=True, inplace=True)
print('This is the rows', dataset_orig.shape[0])

print('Data', dataset_orig.shape)
print(dataset_orig[['derived_msa-md', 'derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']])

# switch action_taken to last column
action_taken_col = dataset_orig.pop('action_taken')
dataset_orig.insert(len(dataset_orig.columns), 'action_taken', action_taken_col)


#####-----------------------Declaring Cutting Functions-----------------
def removeExempt(array_columns, df):
    for startIndex in range(len(array_columns)):
        currentIndexName = df[df[array_columns[startIndex]] == "Exempt"].index
        df.drop(currentIndexName, inplace=True)


def removeBlank(array_columns, df):
    for startIndex in range(len(array_columns)):
        currentIndexName = df[df[array_columns[startIndex]] == ""].index
        df.drop(currentIndexName, inplace=True)


def bucketingColumns(column, arrayOfUniqueVals, nicheVar):
    currentCol = column
    for firstIndex in range(len(arrayOfUniqueVals)):
        try:
            dataset_orig.loc[(nicheVar == arrayOfUniqueVals[firstIndex]), currentCol] = firstIndex
        except:
            print("This number didn't work:\n", firstIndex)


#####------------------Scaling------------------------------------
def scale_dataset(processed_df):
    scaler = MinMaxScaler()
    scaled_df = pd.DataFrame(scaler.fit_transform(processed_df), columns=processed_df.columns)
    return scaled_df


###------------------Preprocessing Function (includes Scaling)------------------------
def preprocessing(dataset_orig):
    # if you want 'derived_loan_product_type' column add here
    dataset_orig = dataset_orig[
        ['derived_msa-md', 'derived_loan_product_type', 'derived_ethnicity', 'derived_race', 'derived_sex',
         'purchaser_type', 'preapproval', 'loan_type', 'loan_purpose', 'lien_status', 'reverse_mortgage',
         'open-end_line_of_credit', 'business_or_commercial_purpose', 'loan_amount', 'hoepa_status',
         'negative_amortization', 'interest_only_payment', 'balloon_payment', 'other_nonamortizing_features',
         'construction_method',
         'occupancy_type', 'manufactured_home_secured_property_type', 'manufactured_home_land_property_interest',
         'applicant_credit_score_type',
         'co-applicant_credit_score_type', 'applicant_ethnicity-1', 'co-applicant_ethnicity-1',
         'applicant_ethnicity_observed',
         'co-applicant_ethnicity_observed', 'applicant_race-1', 'co-applicant_race-1', 'applicant_race_observed',
         'co-applicant_race_observed',
         'applicant_sex', 'co-applicant_sex', 'applicant_sex_observed', 'co-applicant_sex_observed',
         'submission_of_application',
         'initially_payable_to_institution', 'aus-1', 'denial_reason-1', 'tract_population',
         'tract_minority_population_percent',
         'ffiec_msa_md_median_family_income', 'tract_to_msa_income_percentage', 'tract_owner_occupied_units',
         'tract_one_to_four_family_homes',
         'tract_median_age_of_housing_units', 'action_taken']]

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
    dataset_orig['derived_ethnicity'] = dataset_orig['derived_ethnicity'].replace(
        ['Hispanic or Latino', 'Not Hispanic or Latino', 'Joint'],
        [0, 1, 2])
    print('ethnicity: ' + str(dataset_orig.shape))
    print(dataset_orig[['derived_msa-md', 'derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']])

    # ----------------Action_Taken-----------------
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

    # #######----------Remove Exempts and Blanks-------------
    # array_columns_to_remove = ["interest_rate", 'loan_to_value_ratio']
    # removeExempt(array_columns_to_remove, dataset_orig)
    # removeBlank(array_columns_to_remove, dataset_orig)

    ####---------------Scale Dataset---------------

    # array_columns_to_remove = ['interest_rate', 'loan_to_value_ratio']
    #
    # removeExempt(array_columns_to_remove, dataset_orig)
    # removeBlank(array_columns_to_remove, dataset_orig)
    # print(dataset_orig[['interest_rate', 'loan_to_value_ratio']])
    dataset_orig = dataset_orig.apply(pd.to_numeric)
    dataset_orig = dataset_orig.dropna()
    dataset_orig = scale_dataset(dataset_orig)
    # print(dataset_orig[['interest_rate', 'loan_to_value_ratio']])

    ####---------------Reset Indexes----------------
    dataset_orig.reset_index(drop=True, inplace=True)

    return dataset_orig


###---------Call Preprocessing to Create Processed_Scaled_Df and Added in Extra Datapoints---------------
processed_scaled_df = preprocessing(dataset_orig)
processed_scaled_shape = processed_scaled_df.shape
# #ADDITION DATASET ADDED HERE
# added_df = pd.read_csv(add_file, dtype=object)
# added_df_2 = pd.read_csv(add_file_2, dtype=object)
# processed_scaled_df = pd.concat([processed_scaled_df, added_df, added_df_2])
# processed_scaled_df = processed_scaled_df.apply(pd.to_numeric)
# processed_scaled_df = processed_scaled_df.sample(frac=1)
# processed_scaled_df.reset_index(drop=True, inplace=True)
# processed_scaled_df.to_csv(process_scale_file, index=False)
# filterinfDataframeMale = processed_scaled_df[(processed_scaled_df['derived_ethnicity'] == 0) & (processed_scaled_df['derived_race'] == 1) & (processed_scaled_df['derived_sex'] == 0.5) & (processed_scaled_df['action_taken'] == 0)]
# filterinfDataframeFemale = processed_scaled_df[(processed_scaled_df['derived_ethnicity'] == 0) & (processed_scaled_df['derived_race'] == 1) & (processed_scaled_df['derived_sex'] == 0) & (processed_scaled_df['action_taken'] == 0)]
#
# print(filterinfDataframeMale)
# print(filterinfDataframeFemale)
##------------------Check beginning Measures----------------------

processed_scaled_df["derived_sex"] = pd.to_numeric(processed_scaled_df.derived_sex, errors='coerce')
processed_scaled_df["derived_race"] = pd.to_numeric(processed_scaled_df.derived_race, errors='coerce')
processed_scaled_df["derived_ethnicity"] = pd.to_numeric(processed_scaled_df.derived_ethnicity, errors='coerce')
# processed_scaled_df["interest_rate"] = pd.to_numeric(processed_scaled_df.interest_rate, errors='coerce')
# processed_scaled_df["loan_to_value_ratio"] = pd.to_numeric(processed_scaled_df.loan_to_value_ratio, errors='coerce')
processed_scaled_df["action_taken"] = pd.to_numeric(processed_scaled_df.action_taken, errors='coerce')

np.random.seed(0)

# Divide into Train Set, Validation Set, Test Set
processed_scaled_train, processed_and_scaled_test = train_test_split(processed_scaled_df, test_size=0.2, random_state=0,
                                                                     shuffle=True)
print(processed_scaled_train)
print(processed_and_scaled_test)
X_train, y_train = processed_scaled_train.loc[:, processed_scaled_train.columns != 'action_taken'], \
                   processed_scaled_train['action_taken']
X_test, y_test = processed_and_scaled_test.loc[:, processed_and_scaled_test.columns != 'action_taken'], \
                 processed_and_scaled_test['action_taken']

clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100)
clf.fit(X_train, y_train)
print("Accuracy:", clf.score(X_test, y_test))

##-----------------------------------------------------------------------------------------------------------------------------------------------------


###===============Part 2: Working w/ Processed_Scaled_Df=================
ind_cols = ['derived_ethnicity', 'derived_race', 'derived_sex']
dep_col = 'action_taken'


def get_unique_df(processed_scaled_df, ind_cols):
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

#-------------------------------Peformance Metrics--------------


#Calculates Recall Metric
def calculate_recall(TP,FP,FN,TN):
    if (TP + FN) != 0:
        recall = TP / (TP + FN)
    else:
        recall = 0
    return recall

#Calculates Far Metric
def calculate_far(TP,FP,FN,TN):
    if (FP + TN) != 0:
        far = FP / (FP + TN)
    else:
        far = 0
    return far

#Calculates Precision Metric
def calculate_precision(TP,FP,FN,TN):
    if (TP + FP) != 0:
        prec = TP / (TP + FP)
    else:
        prec = 0
    return prec

#Calculates Accuracy Metric
def calculate_accuracy(TP,FP,FN,TN):
    return (TP + TN)/(TP + TN + FP + FN)

#Calculates F1 Score
def calculate_F1(TP,FP,FN,TN):
    precision = calculate_precision(TP,FP,FN,TN)
    recall = calculate_recall(TP,FP,FN,TN)
    F1 = (2 * precision * recall)/(precision + recall)
    return round(F1,2)

def evaluate_eod(df, biased_col):
    df_train, df_test = train_test_split(df, test_size=0.3, random_state=0, shuffle=True)
    X_train, y_train = df_train.loc[:, df_train.columns != 'action_taken'], \
                       df_train['action_taken']
    X_test, y_test = df_test.loc[:, df_test.columns != 'action_taken'], \
                     df_test['action_taken']

    clf_1 = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100)
    clf_1.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    return measure_new_eod(df_test, biased_col, y_pred)

def evaluate_aod(df, biased_col):
    df_train, df_test = train_test_split(df, test_size=0.3, random_state=0, shuffle=True)
    X_train, y_train = df_train.loc[:, df_train.columns != 'action_taken'], \
                       df_train['action_taken']
    X_test, y_test = df_test.loc[:, df_test.columns != 'action_taken'], \
                     df_test['action_taken']

    clf_1 = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100)
    clf_1.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    return measure_new_aod(df_test, biased_col, y_pred)


# ---------------------------------------- NOVEL FAIRNESS METRIC ----------------------------------
def evaluate_awi(df):
    df_train, df_test = train_test_split(df, test_size=0.3, random_state=0, shuffle=True)
    X_train, y_train = df_train.loc[:, df_train.columns != 'action_taken'], \
                       df_train['action_taken']
    X_test, y_test = df_test.loc[:, df_test.columns != 'action_taken'], \
                     df_test['action_taken']

    clf_1 = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100)
    clf_1.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    cnf_matrix = confusion_matrix(y_test, y_pred)
    TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()

    acc = calculate_accuracy(TP, FP, FN, TN)
    recall = calculate_recall(TP, FP, FN, TN)
    precision = calculate_precision(TP, FP, FN, TN)
    far = calculate_far(TP, FP, FN, TN)
    F1 = calculate_F1(TP, FP, FN, TN)


    clf_2 = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100)
    clf_2.fit(X_train, y_train)
    removal_list = []

    count = 0
    for index, row in df_test.iterrows():

        pred_list = []
        row_ = [row.values[:len(row.values) - 1]]
        print("This is row", count, "of a total dataset of", df_test.shape[0])
        print('IM Global', global_unique_df)

        for _, other_row in global_unique_df.iterrows():
            current_comb = other_row.values[
                           :len(other_row.values)]  ## indexes are 2, 3, 4 for ethnicity, race, sex respectively
            original_ethnic, original_race, original_sex = row_[0][2], row_[0][3], row_[0][4]
            row_[0][2] = current_comb[0]
            row_[0][3] = current_comb[1]
            row_[0][4] = current_comb[2]
            y_current_pred = clf_2.predict(row_)[0]
            pred_list.append(y_current_pred)
            row_[0][2], row_[0][3], row_[0][4] = original_ethnic, original_race, original_sex

        print('Pred_list:', pred_list)
        num_unique_vals = len(set(pred_list))

        if num_unique_vals > 1:
            removal_list.append(index)
        elif num_unique_vals == 0:
            raise EmptyList
        count = count + 1

    removal_list = set(removal_list)
    total_biased_points = len(removal_list)
    print('total_biased_points:', total_biased_points)
    total_dataset_points = df_test.shape[0]

    # percentage of points unfairly predicted by the model
    AWI = total_biased_points / total_dataset_points

    return AWI, acc, precision, recall, far, F1


# AWI_initial, acc_intial, precision_intial, recall_intial, far_intial, F1_intial = evaluate_awi(processed_scaled_df)
EOD_sex_intial = evaluate_eod(processed_scaled_df, "derived_sex")
EOD_race_intial= evaluate_eod(processed_scaled_df, "derived_race")
EOD_ethnicity_intial = evaluate_eod(processed_scaled_df, "derived_ethnicity")
AOD_sex_intial = evaluate_aod(processed_scaled_df, "derived_sex")
AOD_race_intial= evaluate_aod(processed_scaled_df, "derived_race")
AOD_ethnicity_intial = evaluate_aod(processed_scaled_df, "derived_ethnicity")

# print("||||||||||||||| Initial AWI:", AWI_initial, "||||||||||||||||||||||||||")
# print("||||||||||||||| Initial Acc:", acc_intial, "||||||||||||||||||||||||||")
# print("||||||||||||||| Initial Precision:", precision_intial, "||||||||||||||||||||||||||")
# print("||||||||||||||| Initial Recall:", recall_intial, "||||||||||||||||||||||||||")
# print("||||||||||||||| Initial False Alarm Rate:", far_intial, "||||||||||||||||||||||||||")
# print("||||||||||||||| Initial F1:", F1_intial, "||||||||||||||||||||||||||")
print("||||||||||||||| Initial EOD_sex:", EOD_sex_intial, "||||||||||||||||||||||||||")
print("||||||||||||||| Initial EOD_race:", EOD_race_intial, "||||||||||||||||||||||||||")
print("||||||||||||||| Initial EOD_ethnicity:", EOD_ethnicity_intial, "||||||||||||||||||||||||||")
print("||||||||||||||| Initial AOD_sex:", AOD_sex_intial, "||||||||||||||||||||||||||")
print("||||||||||||||| Initial AOD_race:", AOD_race_intial, "||||||||||||||||||||||||||")
print("||||||||||||||| Initial AOD_ethnicity:", AOD_ethnicity_intial, "||||||||||||||||||||||||||")


def get_median_val(combination_df):
    current_total = 0
    array_of_bars = []
    for df in combination_df:
        pos_count = len(df[(df['action_taken'] == 1)])
        neg_count = len(df[(df['action_taken'] == 0)])
        current_total += (pos_count + neg_count)
        array_of_bars.append(pos_count)
        array_of_bars.append(neg_count)

    print(array_of_bars)
    median_val = median(array_of_bars)
    print(median_val)

    return round(median_val)


mean_val = get_median_val(combination_df)  # HYPER
print("Target MEDIAN Value:", mean_val)


def RUS_balance(dataset_orig):
    if dataset_orig.empty:
        return dataset_orig
    # print('imbalanced data:\n', dataset_orig['action_taken'].value_counts())
    action_df = dataset_orig['action_taken'].value_counts()
    maj_label = action_df.index[0]
    min_label = action_df.index[-1]
    if maj_label == min_label:
        return dataset_orig
    df_majority = dataset_orig[dataset_orig.action_taken == maj_label]
    df_minority = dataset_orig[dataset_orig.action_taken == min_label]

    df_majority_downsampled = resample(df_majority,
                                       replace=False,  # sample without replacement
                                       n_samples=len(df_minority.index),  # to match minority class
                                       random_state=123)
    # Combine minority class with down sampled majority class
    df_downsampled = pd.concat([df_majority_downsampled, df_minority])

    df_downsampled.reset_index(drop=True, inplace=True)

    # print('balanced data:\n', df_downsampled['action_taken'].value_counts())
    # print('processed data: ' + str(df_downsampled.shape))

    return df_downsampled


# TODO: simplify function
def smote_balance(comb_df):
    def apply_smote(df):
        df.reset_index(drop=True, inplace=True)
        cols = df.columns
        smt = smote(df)
        df = smt.run()
        df.columns = cols
        return df

    X_train, y_train = comb_df.loc[:, comb_df.columns != 'action_taken'], comb_df['action_taken']

    train_df = X_train
    train_df['action_taken'] = y_train
    train_df["action_taken"] = y_train.astype("category")

    train_df = apply_smote(train_df)

    return train_df


def apply_balancing(combination_df, mean_val):
    smoted_list = []
    RUS_list = []
    for c in combination_df:
        pos_count = len(c[(c['action_taken'] == 1)])
        neg_count = len(c[(c['action_taken'] == 0)])

        current_max, current_min = max(pos_count, neg_count), min(pos_count, neg_count)

        if current_max < mean_val:
            print('current_max', current_max)
            print(c[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']])
            smoted_df = smote_balance(c)
            smoted_list.append(smoted_df)
        elif (current_max > mean_val) and (current_min < mean_val):
            print('current_max2', current_max, current_min)
            print(c[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']])
            diff_to_max = current_max - mean_val
            diff_to_min = mean_val - current_min
            if diff_to_max < diff_to_min:
                smoted_df = smote_balance(c)
                RUS_list.append(smoted_df)
            else:
                RUS_df = RUS_balance(c)
                smoted_list.append(RUS_df)
        elif (current_max > mean_val) and (current_min > mean_val):
            print('current_max3', current_max, current_min)
            print(c[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']])
            RUS_df = RUS_balance(c)
            RUS_list.append(RUS_df)

    super_balanced_RUS = []
    for df in RUS_list:
        num_decrease_of_0 = len(df[(df['action_taken'] == 0)]) - mean_val
        num_decrease_of_1 = len(df[(df['action_taken'] == 1)]) - mean_val

        print('Before Distribution\n', df['action_taken'].value_counts())

        df = delete_samples(num_decrease_of_0, df, 0)
        df = delete_samples(num_decrease_of_1, df, 1)

        print('After Distribution\n', df['action_taken'].value_counts())
        super_balanced_RUS.append(df)

    super_balanced_smote = []

    for df in smoted_list:
        num_increase_of_0 = mean_val - len(df[(df['action_taken'] == 0)])
        num_increase_of_1 = mean_val - len(df[(df['action_taken'] == 1)])

        print("Num of Increase:", num_increase_of_0, num_increase_of_1)
        print('Before Distribution\n', df['action_taken'].value_counts())

        df_zeros, df_ones = generate_samples(num_increase_of_0, num_increase_of_1, df, 'HMDA')
        df_added = pd.concat([df_zeros, df_ones])
        concat_df = pd.concat([df, df_added])
        concat_df = concat_df.sample(frac=1).reset_index(drop=True)
        print('After Distribution\n', concat_df['action_taken'].value_counts())
        super_balanced_smote.append(concat_df)

    def concat_and_shuffle(smote_version, RUS_version):
        concat_smote_df = pd.concat(smote_version)
        concat_RUS_df = pd.concat(RUS_version)
        total_concat_df = pd.concat([concat_RUS_df, concat_smote_df])
        total_concat_df = total_concat_df.sample(frac=1).reset_index(drop=True)

        print('Shuffle:', total_concat_df.head(50))
        return total_concat_df

    return concat_and_shuffle(super_balanced_smote, super_balanced_RUS)


new_dataset_orig = apply_balancing(combination_df, mean_val)
new_dataset_orig.to_csv(other_file, index=False)

# -----------------Situation Testing-------------------
X_train, y_train = new_dataset_orig.loc[:, new_dataset_orig.columns != 'action_taken'], new_dataset_orig['action_taken']

clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100)
clf.fit(X_train, y_train)

removal_list = []
# Generates list of rows to be removed
for index, row in new_dataset_orig.iterrows():
    pred_list = []
    row_ = [row.values[:len(row.values) - 1]]
    for other_index, other_row in global_unique_df.iterrows():
        current_comb = other_row.values[
                       :len(other_row.values)]  # indexes are 2, 3, 4 for ethnicity, race, sex respectively
        print('current_comb', current_comb)
        original_ethnic, original_race, original_sex = row_[0][2], row_[0][3], row_[0][4]
        row_[0][2] = current_comb[0]
        row_[0][3] = current_comb[1]
        row_[0][4] = current_comb[2]
        y_current_pred = clf.predict(row_)[0]
        pred_list.append(y_current_pred)
        print('pred_list', pred_list)
        row_[0][2], row_[0][3], row_[0][4] = original_ethnic, original_race, original_sex

    num_unique_vals = len(set(pred_list))

    print('Num Unique Values:', num_unique_vals)

    if num_unique_vals > 1:
        removal_list.append(index)
    elif num_unique_vals == 0:
        raise EmptyList

removal_list = set(removal_list)
print('Length of Removal:', len(removal_list))
print('Removal List', removal_list)
print(new_dataset_orig.shape)

df_removed = pd.DataFrame(columns=new_dataset_orig.columns)

for index, row in new_dataset_orig.iterrows():
    if index in removal_list:
        df_removed = df_removed.append(row, ignore_index=True)
        balanced_and_situation_df = new_dataset_orig.drop(index)
        print(balanced_and_situation_df.shape)

print(df_removed)

# print('Final Distribution1\n', balanced_and_situation_df['action_taken'].value_counts())

##--------------------------Get Final Measures----------------------------

balanced_and_situation_df.to_csv(final_file)

# AWI_final, acc_final, precision_final, recall_final, far_final, F1_final = evaluate_awi(balanced_and_situation_df)
EOD_sex_final = evaluate_eod(balanced_and_situation_df, "derived_sex")
EOD_race_final = evaluate_eod(balanced_and_situation_df, "derived_race")
EOD_ethnicity_final = evaluate_eod(balanced_and_situation_df, "derived_ethnicity")
AOD_sex_final = evaluate_aod(processed_scaled_df, "derived_sex")
AOD_race_final= evaluate_aod(processed_scaled_df, "derived_race")
AOD_ethnicity_final = evaluate_aod(processed_scaled_df, "derived_ethnicity")

print("I WAS median", mean_val)

# print("||||||||||||||| Initial AWI:", AWI_initial, "||||||||||||||||||||||||||")
# print("||||||||||||||| Initial Acc:", acc_intial, "||||||||||||||||||||||||||")
# print("||||||||||||||| Initial Precision:", precision_intial, "||||||||||||||||||||||||||")
# print("||||||||||||||| Initial Recall:", recall_intial, "||||||||||||||||||||||||||")
# print("||||||||||||||| Initial False Alarm Rate:", far_intial, "||||||||||||||||||||||||||")
# print("||||||||||||||| Initial F1:", F1_intial, "||||||||||||||||||||||||||")
print("||||||||||||||| Initial EOD_sex:", EOD_sex_intial, "||||||||||||||||||||||||||")
print("||||||||||||||| Initial EOD_race:", EOD_race_intial, "||||||||||||||||||||||||||")
print("||||||||||||||| Initial EOD_ethnicity:", EOD_ethnicity_intial, "||||||||||||||||||||||||||")
print("||||||||||||||| Initial AOD_sex:", AOD_sex_intial, "||||||||||||||||||||||||||")
print("||||||||||||||| Initial AOD_race:", AOD_race_intial, "||||||||||||||||||||||||||")
print("||||||||||||||| Initial AOD_ethnicity:", AOD_ethnicity_intial, "||||||||||||||||||||||||||")

print("////////////////////////////////////////////////////////////////////////////////////////////////////////")

# print("|||||||||||||||Final AWI:", AWI_final, "||||||||||||||||||||||||||")
# print("||||||||||||||| Final Acc:", acc_final, "||||||||||||||||||||||||||")
# print("||||||||||||||| Final Precision:", precision_final, "||||||||||||||||||||||||||")
# print("||||||||||||||| Final Recall:", recall_final, "||||||||||||||||||||||||||")
# print("||||||||||||||| Final False Alarm Rate:", far_final, "||||||||||||||||||||||||||")
# print("||||||||||||||| Final F1:", F1_final, "||||||||||||||||||||||||||")
print("||||||||||||||| Final EOD_sex:", EOD_sex_final, "||||||||||||||||||||||||||")
print("||||||||||||||| Final EOD_race:", EOD_race_final, "||||||||||||||||||||||||||")
print("||||||||||||||| Final EOD_ethnicity:", EOD_ethnicity_final, "||||||||||||||||||||||||||")
print("||||||||||||||| Final AOD_sex:", AOD_sex_final, "||||||||||||||||||||||||||")
print("||||||||||||||| Final AOD_race:", AOD_race_final, "||||||||||||||||||||||||||")
print("||||||||||||||| Final AOD_ethnicity:", AOD_ethnicity_final, "||||||||||||||||||||||||||")
print("---------------------------------------------------------------------------------------")
print("I'M INPUT:", input_file_1, input_file_2, input_file_3)
# d = {'AWI': [AWI_initial, AWI_final], 'Acc': [acc_intial, acc_final], 'Precision': [precision_intial, precision_final] , 'Recall': [recall_intial, recall_final], 'False Alarm Rate': [far_intial, far_final], "F1": [F1_intial, F1_final]}
# results_df = pd.DataFrame(data=d)
# results_df.to_csv(result_file, index=False)



# ======================================================START OF PSCF=========================================
# nonbiased_features = ['derived_msa-md', 'derived_loan_product_type', 'derived_ethnicity', 'derived_race', 'derived_sex',
#                       'purchaser_type', 'preapproval', 'loan_type', 'loan_purpose', 'lien_status', 'reverse_mortgage',
#                       'open-end_line_of_credit', 'business_or_commercial_purpose', 'loan_amount', 'hoepa_status',
#                       'negative_amortization', 'interest_only_payment', 'balloon_payment',
#                       'other_nonamortizing_features',
#                       'construction_method',
#                       'occupancy_type', 'manufactured_home_secured_property_type',
#                       'manufactured_home_land_property_interest',
#                       'applicant_credit_score_type',
#                       'co-applicant_credit_score_type', 'applicant_ethnicity-1', 'co-applicant_ethnicity-1',
#                       'applicant_ethnicity_observed',
#                       'co-applicant_ethnicity_observed', 'applicant_race-1', 'co-applicant_race-1',
#                       'applicant_race_observed',
#                       'co-applicant_race_observed',
#                       'applicant_sex', 'co-applicant_sex', 'applicant_sex_observed', 'co-applicant_sex_observed',
#                       'submission_of_application',
#                       'initially_payable_to_institution', 'aus-1', 'denial_reason-1', 'tract_population',
#                       'tract_minority_population_percent',
#                       'ffiec_msa_md_median_family_income', 'tract_to_msa_income_percentage',
#                       'tract_owner_occupied_units',
#                       'tract_one_to_four_family_homes',
#                       'tract_median_age_of_housing_units', 'action_taken']

#
# ###======================Part 2: Using PSCF Begins================================
# # ensure the list of nonbiased featueres also contains the label
# def PSCF(list_of_nonbiased_features, dataset):
#     dataset_orig_nonbiased_features = dataset[list_of_nonbiased_features]
#     return dataset_orig_nonbiased_features
#
#
# PSCF_df = PSCF(nonbiased_features, balanced_and_situation_df)
# PSCF_AWI = evaluate_awi(PSCF_df)

# print("||||||||||||||| Initial AWI:", AWI_initial, "||||||||||||||||||||||||||")
# print("|||||||||||||||Final AWI:", AWI_final, "||||||||||||||||||||||||||")
# print("Final Result - PSCF_AWI:", PSCF_AWI)