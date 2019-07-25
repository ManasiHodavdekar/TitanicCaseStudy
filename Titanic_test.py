# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 13:39:27 2019

@author: chinmay
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import math
data_titanic=pd.read_excel("E:\\python\\classes\\datasets\\TitanicData.xlsx")
print (data_titanic)
print("Total no. of passengers travelling",str(len(data_titanic)))
sexwise_list=data_titanic["sex"].tolist()
print("Total male passengers",sexwise_list.count("male"))
print("Total female passengers",sexwise_list.count("female"))

#Plotting the data set available using seaborn object----------

fig,ax=plt.subplots(3,2)
#Finding the no. of people who survived!
sns.countplot(x="survived",data=data_titanic,ax=ax[0][0])
#To find the no. of males and females who survived
sns.countplot(x="survived",hue="sex",data=data_titanic,ax=ax[0][1])
#To find the survivors class wise 
sns.countplot(x="survived",hue="pclass",data=data_titanic,ax=ax[1][0])
#To find the survivors embarked wise
sns.countplot(x="survived",hue="embarked",data=data_titanic,ax=ax[1][1])
fig.show()

#==================DATA WRANGLING==========================================

#DATA WRANGLING: To find the existing null values and delete them

print(data_titanic.isnull())
print(data_titanic.isnull().sum())

#Draw heatmap: USED TO FIND THE NULL VALUES (USED IN DATA WRANGLING)
sns.heatmap(data_titanic.isnull(),yticklabels="false",cmap="viridis",ax=ax[2][0])
#heat map drawn.Now dropping the columns with lot of null data
data_titanic.drop("body",axis=1,inplace=True)
data_titanic.drop("cabin",axis=1,inplace=True)
data_titanic.drop("boat",axis=1,inplace=True)
data_titanic.drop("home.dest",axis=1,inplace=True)
data_titanic.drop("age",axis=1,inplace=True)
sns.heatmap(data_titanic.isnull(),yticklabels="false",cmap="viridis",ax=ax[2][1])
print(data_titanic.isnull().sum())

#Replacing non categorial to categorial data
sex_categorial=pd.get_dummies(data_titanic["sex"])
print(sex_categorial)
sex_categorial=pd.get_dummies(data_titanic["sex"],drop_first=True)
#sex_categorial.drop("male",axis=1,inplace=True)
print(sex_categorial)
#similarly changing other columns
embarked_categorial=pd.get_dummies(data_titanic["embarked"],drop_first=True)
pclass_categorial=pd.get_dummies(data_titanic["pclass"],drop_first=True)
print(embarked_categorial)
print(pclass_categorial)
#concatinating above 2 columns to existing data
data_titanic=pd.concat([data_titanic,sex_categorial,embarked_categorial,pclass_categorial],axis=1)
print(data_titanic.head(5))
data_titanic.drop(["sex","embarked","name","pclass"],axis=1,inplace=True)
print(data_titanic.head(5))

#===================TRAINING AND TESTING===============================

y=data_titanic["survived"]
x=data_titanic.drop(["survived","ticket"],axis=1)
#we didnt need this independent data.so dropping
data_titanic.head()
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.33,random_state=1)
X_train.fillna(X_train.mean(),inplace=True)
Y_train.fillna(Y_train.mean(),inplace=True)
from sklearn.linear_model import LogisticRegression
logmodel= LogisticRegression()
logmodel.fit(X_train,Y_train)

#now make predictions
predictions=logmodel.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(Y_test,predictions))
from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test,predictions))
