# -------------------Imports---------------------------
import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath('..'))


# Custom Exceptions
class EmptyList(Exception):
    pass


# Dataset Used Needs to be Large Enough to Have Data for all 27 Subsets

###======================Part 1: Code and Preprocessing Begins======================
base_path = str(sys.path[0])

input_file = r'C:\Users\jasha\Documents\GitHub\fair-loan-predictor\Data\raw_state_CT.csv'
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


print("I'M INPUT:", input_file)
# print("Yeah, the add file is here", add_file)
# print("Yello:", add_file_2)
dataset_orig = pd.read_csv(input_file, dtype=object)
# df_2020 = pd.read_csv(input_file_1, dtype=object).sample(n=4000000)
# df_2019 = pd.read_csv(input_file_2, dtype=object).sample(n=4000000)
# df_2018 = pd.read_csv(input_file_3, dtype=object).sample(n=4000000)
# dataset_orig = pd.concat([df_2020, df_2019, df_2018])
# dataset_orig = dataset_orig.sample(frac=1)
# dataset_orig.reset_index(drop=True, inplace=True)


print('Data', dataset_orig.shape)
print(dataset_orig[['derived_msa-md', 'derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']])

# switch action_taken to last column
action_taken_col = dataset_orig.pop('action_taken')
dataset_orig.insert(len(dataset_orig.columns), 'action_taken', action_taken_col)

#
# #####-----------------------Declaring Cutting Functions-----------------
# def removeExempt(array_columns, df):
#     for startIndex in range(len(array_columns)):
#         currentIndexName = df[df[array_columns[startIndex]] == "Exempt"].index
#         df.drop(currentIndexName, inplace=True)
#
#
# def removeBlank(array_columns, df):
#     for startIndex in range(len(array_columns)):
#         currentIndexName = df[df[array_columns[startIndex]] == ""].index
#         df.drop(currentIndexName, inplace=True)
#
#
# def bucketingColumns(column, arrayOfUniqueVals, nicheVar):
#     currentCol = column
#     for firstIndex in range(len(arrayOfUniqueVals)):
#         try:
#             dataset_orig.loc[(nicheVar == arrayOfUniqueVals[firstIndex]), currentCol] = firstIndex
#         except:
#             print("This number didn't work:\n", firstIndex)
#
#
# #####------------------Scaling------------------------------------
# def scale_dataset(processed_df):
#     scaler = MinMaxScaler()
#     scaled_df = pd.DataFrame(scaler.fit_transform(processed_df), columns=processed_df.columns)
#     return scaled_df
#
#
# ###------------------Preprocessing Function (includes Scaling)------------------------
# def preprocessing(dataset_orig):
#     # if you want 'derived_loan_product_type' column add here
#     dataset_orig = dataset_orig[
#         ['derived_msa-md', 'derived_loan_product_type', 'derived_ethnicity', 'derived_race', 'derived_sex',
#          'purchaser_type', 'preapproval', 'loan_type', 'loan_purpose', 'lien_status', 'reverse_mortgage',
#          'open-end_line_of_credit', 'business_or_commercial_purpose', 'loan_amount', 'hoepa_status',
#          'negative_amortization', 'interest_only_payment', 'balloon_payment', 'other_nonamortizing_features',
#          'construction_method',
#          'occupancy_type', 'manufactured_home_secured_property_type', 'manufactured_home_land_property_interest',
#          'applicant_credit_score_type',
#          'co-applicant_credit_score_type', 'applicant_ethnicity-1', 'co-applicant_ethnicity-1',
#          'applicant_ethnicity_observed',
#          'co-applicant_ethnicity_observed', 'applicant_race-1', 'co-applicant_race-1', 'applicant_race_observed',
#          'co-applicant_race_observed',
#          'applicant_sex', 'co-applicant_sex', 'applicant_sex_observed', 'co-applicant_sex_observed',
#          'submission_of_application',
#          'initially_payable_to_institution', 'aus-1', 'denial_reason-1', 'tract_population',
#          'tract_minority_population_percent',
#          'ffiec_msa_md_median_family_income', 'tract_to_msa_income_percentage', 'tract_owner_occupied_units',
#          'tract_one_to_four_family_homes',
#          'tract_median_age_of_housing_units', 'action_taken']]
#
#     # Below we are taking out rows in the dataset with values we do not care for. This is from lines 23 - 99.
#     ###--------------------Sex------------------------
#     dataset_orig = dataset_orig[(dataset_orig['derived_sex'] == 'Male') |
#                                 (dataset_orig['derived_sex'] == 'Female') |
#                                 (dataset_orig['derived_sex'] == 'Joint')]
#     dataset_orig['derived_sex'] = dataset_orig['derived_sex'].replace(['Female', 'Male', 'Joint'],
#                                                                       [0, 1, 2])
#     print('sex: ' + str(dataset_orig.shape))
#     print(dataset_orig[['derived_msa-md', 'derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']])
#
#     ###-------------------Races-----------------------
#     dataset_orig = dataset_orig[(dataset_orig['derived_race'] == 'White') |
#                                 (dataset_orig['derived_race'] == 'Black or African American') |
#                                 (dataset_orig['derived_race'] == 'Joint')]
#     dataset_orig['derived_race'] = dataset_orig['derived_race'].replace(['Black or African American', 'White', 'Joint'],
#                                                                         [0, 1, 2])
#     print('race: ' + str(dataset_orig.shape))
#     print(dataset_orig[['derived_msa-md', 'derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']])
#
#     ####----------------Ethnicity-------------------
#     dataset_orig = dataset_orig[(dataset_orig['derived_ethnicity'] == 'Hispanic or Latino') |
#                                 (dataset_orig['derived_ethnicity'] == 'Not Hispanic or Latino') |
#                                 (dataset_orig['derived_ethnicity'] == 'Joint')]
#     dataset_orig['derived_ethnicity'] = dataset_orig['derived_ethnicity'].replace(
#         ['Hispanic or Latino', 'Not Hispanic or Latino', 'Joint'],
#         [0, 1, 2])
#     print('ethnicity: ' + str(dataset_orig.shape))
#     print(dataset_orig[['derived_msa-md', 'derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']])
#
#     # ----------------Action_Taken-----------------
#     dataset_orig = dataset_orig[(dataset_orig['action_taken'] == '1') |
#                                 (dataset_orig['action_taken'] == '2') |
#                                 (dataset_orig['action_taken'] == '3')]
#
#     dataset_orig['action_taken'] = dataset_orig['action_taken'].replace(['1', '2', '3'],
#                                                                         [1, 0, 0])
#
#     print(dataset_orig[['derived_msa-md', 'derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']])
#
#     ######----------------Loan Product-------------------
#     # assigns each unique categorical value a unique integer id
#     dataset_orig['derived_loan_product_type'] = dataset_orig['derived_loan_product_type'].astype('category').cat.codes
#
#     print('loan product: ' + str(dataset_orig.shape))
#     print(dataset_orig[['derived_msa-md', 'derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']])
#
#     # #######----------Remove Exempts and Blanks-------------
#     # array_columns_to_remove = ["interest_rate", 'loan_to_value_ratio']
#     # removeExempt(array_columns_to_remove, dataset_orig)
#     # removeBlank(array_columns_to_remove, dataset_orig)
#
#     ####---------------Scale Dataset---------------
#
#     # array_columns_to_remove = ['interest_rate', 'loan_to_value_ratio']
#     #
#     # removeExempt(array_columns_to_remove, dataset_orig)
#     # removeBlank(array_columns_to_remove, dataset_orig)
#     # print(dataset_orig[['interest_rate', 'loan_to_value_ratio']])
#     dataset_orig = dataset_orig.apply(pd.to_numeric)
#     dataset_orig = dataset_orig.dropna()
#     dataset_orig = scale_dataset(dataset_orig)
#     # print(dataset_orig[['interest_rate', 'loan_to_value_ratio']])
#
#     ####---------------Reset Indexes----------------
#     dataset_orig.reset_index(drop=True, inplace=True)
#
#     return dataset_orig
#
#
# ###---------Call Preprocessing to Create Processed_Scaled_Df and Added in Extra Datapoints---------------
# processed_scaled_df = preprocessing(dataset_orig)
# processed_scaled_shape = processed_scaled_df.shape
# # #ADDITION DATASET ADDED HERE
# # added_df = pd.read_csv(add_file, dtype=object)
# # added_df_2 = pd.read_csv(add_file_2, dtype=object)
# # processed_scaled_df = pd.concat([processed_scaled_df, added_df, added_df_2])
# # processed_scaled_df = processed_scaled_df.apply(pd.to_numeric)
# # processed_scaled_df = processed_scaled_df.sample(frac=1)
# # processed_scaled_df.reset_index(drop=True, inplace=True)
# # processed_scaled_df.to_csv(process_scale_file, index=False)
# # filterinfDataframeMale = processed_scaled_df[(processed_scaled_df['derived_ethnicity'] == 0) & (processed_scaled_df['derived_race'] == 1) & (processed_scaled_df['derived_sex'] == 0.5) & (processed_scaled_df['action_taken'] == 0)]
# # filterinfDataframeFemale = processed_scaled_df[(processed_scaled_df['derived_ethnicity'] == 0) & (processed_scaled_df['derived_race'] == 1) & (processed_scaled_df['derived_sex'] == 0) & (processed_scaled_df['action_taken'] == 0)]
# #
# # print(filterinfDataframeMale)
# # print(filterinfDataframeFemale)
# ##------------------Check beginning Measures----------------------
#
# processed_scaled_df["derived_sex"] = pd.to_numeric(processed_scaled_df.derived_sex, errors='coerce')
# processed_scaled_df["derived_race"] = pd.to_numeric(processed_scaled_df.derived_race, errors='coerce')
# processed_scaled_df["derived_ethnicity"] = pd.to_numeric(processed_scaled_df.derived_ethnicity, errors='coerce')
# # processed_scaled_df["interest_rate"] = pd.to_numeric(processed_scaled_df.interest_rate, errors='coerce')
# # processed_scaled_df["loan_to_value_ratio"] = pd.to_numeric(processed_scaled_df.loan_to_value_ratio, errors='coerce')
# processed_scaled_df["action_taken"] = pd.to_numeric(processed_scaled_df.action_taken, errors='coerce')
#
# np.random.seed(0)
#
# # Divide into Train Set, Validation Set, Test Set
# processed_scaled_train, processed_and_scaled_test = train_test_split(processed_scaled_df, test_size=0.2, random_state=0,
#                                                                      shuffle=True)
# print(processed_scaled_train)
# print(processed_and_scaled_test)
# X_train, y_train = processed_scaled_train.loc[:, processed_scaled_train.columns != 'action_taken'], \
#                    processed_scaled_train['action_taken']
# X_test, y_test = processed_and_scaled_test.loc[:, processed_and_scaled_test.columns != 'action_taken'], \
#                  processed_and_scaled_test['action_taken']
#
# clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100)
# clf.fit(X_train, y_train)
# print("Accuracy:", clf.score(X_test, y_test))
#
# ##-----------------------------------------------------------------------------------------------------------------------------------------------------
#
#
# ###===============Part 2: Working w/ Processed_Scaled_Df=================
# ind_cols = ['derived_ethnicity', 'derived_race', 'derived_sex']
# dep_col = 'action_taken'
#
#
# def get_unique_df(processed_scaled_df, ind_cols):
#     uniques = [processed_scaled_df[i].unique().tolist() for i in ind_cols]
#     unique_df = pd.DataFrame(product(*uniques), columns=ind_cols)
#     return unique_df
#
#
# def split_dataset(processed_scaled_df, ind_cols):
#     unique_df = get_unique_df(processed_scaled_df, ind_cols)
#     combination_df = [pd.merge(processed_scaled_df, unique_df.iloc[[i]], on=ind_cols, how='inner') for i in
#                       range(unique_df.shape[0])]
#     return combination_df
#
#
# global_unique_df = get_unique_df(processed_scaled_df, ind_cols)
# print(global_unique_df)
# combination_df = split_dataset(processed_scaled_df, ind_cols)
#
#
# def get_median_val(combination_df):
#     current_total = 0
#     array_of_bars = []
#     for df in combination_df:
#         pos_count = len(df[(df['action_taken'] == 1)])
#         neg_count = len(df[(df['action_taken'] == 0)])
#         current_total += (pos_count + neg_count)
#         array_of_bars.append(pos_count)
#         array_of_bars.append(neg_count)
#
#     print(array_of_bars)
#     median_val = median(array_of_bars)
#     print(median_val)
#
#     return round(median_val)
#
#
# mean_val = get_median_val(combination_df)  # HYPER
# print("Target MEDIAN Value:", mean_val)
#
#
# def RUS_balance(dataset_orig):
#     if dataset_orig.empty:
#         return dataset_orig
#     # print('imbalanced data:\n', dataset_orig['action_taken'].value_counts())
#     action_df = dataset_orig['action_taken'].value_counts()
#     maj_label = action_df.index[0]
#     min_label = action_df.index[-1]
#     if maj_label == min_label:
#         return dataset_orig
#     df_majority = dataset_orig[dataset_orig.action_taken == maj_label]
#     df_minority = dataset_orig[dataset_orig.action_taken == min_label]
#
#     df_majority_downsampled = resample(df_majority,
#                                        replace=False,  # sample without replacement
#                                        n_samples=len(df_minority.index),  # to match minority class
#                                        random_state=123)
#     # Combine minority class with down sampled majority class
#     df_downsampled = pd.concat([df_majority_downsampled, df_minority])
#
#     df_downsampled.reset_index(drop=True, inplace=True)
#
#     # print('balanced data:\n', df_downsampled['action_taken'].value_counts())
#     # print('processed data: ' + str(df_downsampled.shape))
#
#     return df_downsampled
#
#
# # TODO: simplify function
# def smote_balance(comb_df):
#     def apply_smote(df):
#         df.reset_index(drop=True, inplace=True)
#         cols = df.columns
#         smt = smote(df)
#         df = smt.run()
#         df.columns = cols
#         return df
#
#     X_train, y_train = comb_df.loc[:, comb_df.columns != 'action_taken'], comb_df['action_taken']
#
#     train_df = X_train
#     train_df['action_taken'] = y_train
#     train_df["action_taken"] = y_train.astype("category")
#
#     train_df = apply_smote(train_df)
#
#     return train_df
#
#
# def apply_balancing(combination_df, mean_val):
#     smoted_list = []
#     RUS_list = []
#     for c in combination_df:
#         pos_count = len(c[(c['action_taken'] == 1)])
#         neg_count = len(c[(c['action_taken'] == 0)])
#
#         current_max, current_min = max(pos_count, neg_count), min(pos_count, neg_count)
#
#         if current_max < mean_val:
#             print('current_max', current_max)
#             print(c[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']])
#             smoted_df = smote_balance(c)
#             smoted_list.append(smoted_df)
#         elif (current_max > mean_val) and (current_min < mean_val):
#             print('current_max2', current_max, current_min)
#             print(c[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']])
#             diff_to_max = current_max - mean_val
#             diff_to_min = mean_val - current_min
#             if diff_to_max < diff_to_min:
#                 smoted_df = smote_balance(c)
#                 RUS_list.append(smoted_df)
#             else:
#                 RUS_df = RUS_balance(c)
#                 smoted_list.append(RUS_df)
#         elif (current_max > mean_val) and (current_min > mean_val):
#             print('current_max3', current_max, current_min)
#             print(c[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']])
#             RUS_df = RUS_balance(c)
#             RUS_list.append(RUS_df)
#
#     super_balanced_RUS = []
#     for df in RUS_list:
#         num_decrease_of_0 = len(df[(df['action_taken'] == 0)]) - mean_val
#         num_decrease_of_1 = len(df[(df['action_taken'] == 1)]) - mean_val
#
#         print('Before Distribution\n', df['action_taken'].value_counts())
#
#         df = delete_samples(num_decrease_of_0, df, 0)
#         df = delete_samples(num_decrease_of_1, df, 1)
#
#         print('After Distribution\n', df['action_taken'].value_counts())
#         super_balanced_RUS.append(df)
#
#     super_balanced_smote = []
#
#     for df in smoted_list:
#         num_increase_of_0 = mean_val - len(df[(df['action_taken'] == 0)])
#         num_increase_of_1 = mean_val - len(df[(df['action_taken'] == 1)])
#
#         print("Num of Increase:", num_increase_of_0, num_increase_of_1)
#         print('Before Distribution\n', df['action_taken'].value_counts())
#
#         df_zeros, df_ones = generate_samples(num_increase_of_0, num_increase_of_1, df, 'HMDA')
#         df_added = pd.concat([df_zeros, df_ones])
#         concat_df = pd.concat([df, df_added])
#         concat_df = concat_df.sample(frac=1).reset_index(drop=True)
#         print('After Distribution\n', concat_df['action_taken'].value_counts())
#         super_balanced_smote.append(concat_df)
#
#     def concat_and_shuffle(smote_version, RUS_version):
#         concat_smote_df = pd.concat(smote_version)
#         concat_RUS_df = pd.concat(RUS_version)
#         total_concat_df = pd.concat([concat_RUS_df, concat_smote_df])
#         total_concat_df = total_concat_df.sample(frac=1).reset_index(drop=True)
#
#         print('Shuffle:', total_concat_df.head(50))
#         return total_concat_df
#
#     return concat_and_shuffle(super_balanced_smote, super_balanced_RUS)


y1 = []
y2 = []
# for c in combination_df:
#     pos_count = len(c[(c['action_taken'] == 1)])
#     neg_count = len(c[(c['action_taken'] == 0)])
#     # total_count = pos_count + neg_count
#     # pos_count = pos_count / total_count
#     # neg_count = neg_count / total_count
#     y1.append(pos_count)
#     y2.append(neg_count)

print(y1)
print(y2)
# importing package
import matplotlib.pyplot as plt
y1 = [78, 83, 87, 227, 1042]
y2 = [23, 17, 17, 57, 397]
print(y1)
print(y2)

plt.rcParams["font.family"] = "sans-serif"
# create data
x = ['JWF', 'JBM', 'HOLJJ', 'JJJ', 'NHOLBJ']
# plot bars in stack manner

fig, ax = plt.subplots()

ax.bar(x, y1, color='#696969', label='Accepted Loan Applicants')
ax.bar(x, y2,  bottom=y1, color='#ffccc9', label='Denied loan Applicants')

ax.grid(axis = 'y', color='#F1F1F1')
ax.tick_params(direction="in", length=10)
ax.set_axisbelow(True)  # This line added.

plt.xlabel("Sub-datasets")
plt.ylabel("Dataset Points")
plt.legend()
plt.show()

plt.close()
#
# new_dataset_orig = apply_balancing(combination_df, mean_val)
# # new_dataset_orig.to_csv(other_file, index=False)
#
# combination_df_2 = split_dataset(new_dataset_orig, ind_cols)
#
# y1_2 = []
# y2_2 = []
# for c in combination_df_2:
#     pos_count = len(c[(c['action_taken'] == 1)])
#     neg_count = len(c[(c['action_taken'] == 0)])
#     # total_count = pos_count + neg_count
#     # pos_count = pos_count / total_count
#     # neg_count = neg_count / total_count
#     y1_2.append(pos_count)
#     y2_2.append(neg_count)

# importing package
import matplotlib.pyplot as plt


y1_2 = [85,85,85,85,85]
y2_2 = [85,85,85,85,85]


print(y1_2)
print(y2_2)


plt.rcParams["font.family"] = "sans-serif"
# create data
labels = ['JWF', 'JBM', 'HOLJJ', 'JJJ', 'NHOLBJ']
# plot bars in stack manner

fig, ax = plt.subplots()

ax.bar(labels, y1_2, color='#696969', label='Accepted Loan Applicants')
ax.bar(labels, y2_2,  bottom=y1_2, color='#ffccc9', label='Denied loan Applicants')

ax.grid(axis = 'y', color='#F1F1F1')
ax.tick_params(direction="in", length=10)
ax.set_axisbelow(True)  # This line added.

plt.ylim(0, (np.max(y1) + np.max(y2)))
plt.xlabel("Sub-datasets")
plt.ylabel("Dataset Points")
plt.legend()
plt.show()


#balanced