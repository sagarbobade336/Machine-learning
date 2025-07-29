# -*- coding: utf-8 -*-
"""KNN ML MODEL 1.ipynb

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

df = pd.read_csv('/content/Iris.csv')

df

df["Species"].unique()

df.columns

x = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = df['Species']

#training testing splite
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=23)

model=KNeighborsClassifier()

model.fit(x_train,y_train)

y_pred=model.predict(x_test)

y_pred

y_test

