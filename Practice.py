from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os.path
import time
from random import randint

import numpy as np
import tensorflow as tf
import pandas as pd
from keras import backend as K
from pandas.core.ops import array_ops

array = np.array([], dtype=object)
newArray = np.insert(array, 0, "Hello")
newArray2 = np.insert(newArray, 1, 6)
print(newArray)