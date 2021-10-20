#-------------------Imports---------------------------
import copy
import os
import sys
import random

import numpy as np
from itertools import product

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample

from imblearn.under_sampling import RandomUnderSampler

base_path = str(sys.path[0])

input_file = base_path + '\\Data\\this_is_balanced_CT.csv'

dataset_orig = pd.read_csv(input_file, dtype=object)

rus = RandomUnderSampler(random_state=0)
