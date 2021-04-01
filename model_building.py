# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from imblearn.combine import SMOTEENN
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

data = pd.read_csv("VowelA_High_latest.csv")


data['label'].value_counts()


data=data.replace({"label":{"Healthy":0,"Unhealthy":1}})
data=data.replace({"G":{"w":0,"m":1}})
data=data.drop(columns=['Unnamed: 0', 'filename', 'ID'])


data.head(5)

X=data
X = X.drop(columns=['label'])
Y=data['label']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.25,random_state=4)



from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
x_train_scaled = pd.DataFrame(sc_X.fit_transform(x_train))
x_train_scaled.columns = x_train.columns.values
x_train_scaled.index = x_train.index.values
x_train = x_train_scaled
x_test_scaled = pd.DataFrame(sc_X.transform(x_test))
x_test_scaled.columns = x_test.columns.values
x_test_scaled.index = x_test.index.values
x_test = x_test_scaled








from xgboost import XGBClassifier


sm = SMOTEENN()
X_resampled1, y_resampled1 = sm.fit_resample(X,Y)

xr_train1,xr_test1,yr_train1,yr_test1=train_test_split(X_resampled1, y_resampled1,test_size=0.25)

model = XGBClassifier()
model.fit(xr_train1,yr_train1)



y_pred = model.predict(xr_test1)
predictions = [round(value) for value in y_pred]


print(classification_report(yr_test1, y_pred))
print(confusion_matrix(yr_test1, y_pred))


#SAVING THE MODEL

import pickle
filename = 'model.sav'
pickle.dump(model, open(filename, 'wb'))



#LOADING THE MODEL ON TEST DATA

import pickle
import pandas as pd

model = pickle.load(open("model.sav", "rb"))
new_df = pd.read_csv('test.csv')

new_df['G'] = 'm'
new_df['A'] = 30

new_df=new_df.drop(columns=['file_name', 'label'])
new_df=new_df.replace({"G":{"w":0,"m":1}})

'''
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
new_df_scaled = pd.DataFrame(sc_X.transform(new_df))
'''

single = model.predict(new_df.tail(1))