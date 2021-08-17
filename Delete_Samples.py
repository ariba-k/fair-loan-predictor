from __future__ import print_function, division

import random


def delete_samples(how_much_cut, df, num_to_cut):
    numCols = len(df.columns) - 1
    numDeleted = 0
    threshold = how_much_cut

    while (numDeleted < threshold):
        numRows = len(df) - 1
        numRandom = random.randint(0, numRows)
        # print(numRandom)
        randomRowActionTaken = df.loc[numRandom].iat[numCols]

        if (randomRowActionTaken == num_to_cut):
            df = df.drop(numRandom)
            df.reset_index(drop=True, inplace=True)
            numDeleted += 1
            print('numdeleted:', numDeleted)

    return df