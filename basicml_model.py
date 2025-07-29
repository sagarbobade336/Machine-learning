# -*- coding: utf-8 -*-
"""BasicML model.ipynb

import pandas as pd

df = pd.read_csv('music.csv')

df

#creating input and output
x=df[["age","gender"]]
y=df["genre"]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

#training
from sklearn.neighbors import KNeighborsClassifier

model=KNeighborsClassifier()

model.fit(x_train,y_train)

y_pred=model.predict(x_test)

y_pred

y_test

#report
from sklearn.metrics import classification_report

print(classification_report(y_test,y_pred))

