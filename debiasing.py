import os
import sys
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from itertools import product
from balancing import balance

sys.path.append(os.path.abspath('..'))

'''This file is the main file to debias a dataset. Before using this file, make sure your dataset is balanced
as you will get a very bad debiased dataset with very low number of rows. You can begin to debias the dataset
by chaning line 18 where it says BalancedCTHMDA. Lastly, you can save it at the end (line 837) doing the same
process'''
base_path = str(sys.path[0])
input_file = base_path + '/data/processed_state_WY.csv'
interm_file = base_path + '/data/scaled_state_WY.csv'
output_file = base_path + '/data/debiased_state_WY.csv'


# str(sys.path[0]) + '\\Data\\' + 'BalancedCTHMDA.csv'

class EmptyList(Exception):
    pass


processed_df = pd.read_csv(input_file, dtype=object)


def scale_dataset(processed_df):
    #####------------------Scaling------------------------------------
    scaler = MinMaxScaler()
    scaled_df = pd.DataFrame(scaler.fit_transform(processed_df), columns=processed_df.columns)

    return scaled_df


scaled_df = scale_dataset(processed_df)
scaled_df.to_csv(interm_file, index=False)

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


def split_dataset(scaled_df, ind_cols):
    uniques = [scaled_df[i].unique().tolist() for i in ind_cols]
    unique_df = pd.DataFrame(product(*uniques), columns=ind_cols)
    print(unique_df)
    combination_df = [balance(pd.merge(scaled_df, unique_df.iloc[[i]], on=ind_cols, how='inner')) for i in
                      range(unique_df.shape[0])]

    return combination_df


combination_df = split_dataset(scaled_df, ind_cols)

combination_names = ['nhol_w_m', 'nhol_w_f', 'nhol_w_j',
                     'nhol_j_m', 'nhol_j_f', 'nhol_j_j',
                     'nhol_b_m', 'nhol_b_f', 'nhol_b_j',
                     'hol_w_m', 'hol_w_f', 'hol_w_j',
                     'hol_j_m', 'hol_j_f', 'hol_j_j',
                     'hol_b_m', 'hol_b_f', 'hol_b_j',
                     'j_w_m', 'j_w_f', 'j_w_j',
                     'j_j_m', 'j_j_f', 'j_j_j',
                     'j_b_m', 'j_b_f','j_b_j']
# uncomment to see which name is associated with which combination
# for n, c in zip(combination_names, combination_df):
#     print(n, c)

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

    X_train, y_train = comb_df.loc[:, comb_df.columns != 'action_taken'], comb_df['action_taken']
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
            dataset_orig.drop(index)
        elif num_unique_vals == 0:
            raise EmptyList

    return dataset_orig


debias_df = debias_dataset(scaled_df, classifiers)
debias_df.to_csv(output_file, index=False)
