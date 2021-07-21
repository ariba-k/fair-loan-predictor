
import sklearn
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


col = ['Age','Workclass','Fnlwgt','Education','EducationNum','Marital-status','Occupation', 'Relationship', 'Race','Sex','CaptilGain','CaptilLoss','HoursPerWeek','NativeCountry','Label']
data = pd.read_csv("datasets/adult_5k.csv", names = col)
print(data)
predict = 'Sex'


X = np.array(data.drop([predict], 1))
print('hello this is x: \n', X)
y = np.array(data[predict])
print('This is y: \n', y)


#y = np.array(adult_income_data[9])

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=0)

# all parameters not specified are set to their defaults
logisticRegr = LogisticRegression()

logisticRegr.fit(x_train, y_train)
acc = logisticRegr.score(x_test, y_test)
print(acc)

