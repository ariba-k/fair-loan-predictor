import numpy as np
import pandas as pd
import os
import sys

import numpy as np
from itertools import product

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample

sys.path.append(os.path.abspath('..'))
###======================Part 1: Importing Datasets Begins======================
base_path = str(sys.path[0])

input_file = base_path + '\\Data\\raw_state_CA.csv'
interm_file = base_path + '\\Data\\FirstBalancedCA.csv'
output_file = base_path + '\\Data\\DoubleBalancedCA.csv'


dataset_orig = pd.read_csv(input_file, dtype=object)
print('Data', dataset_orig.shape)
print(dataset_orig[['derived_msa-md', 'derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']])

