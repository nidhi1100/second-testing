#import pandas for dataframes
import pandas as pd

#import numpy for numerical operations
import numpy as np

#import seaborn for visualization
import seaborn as sns

#to partition data
from sklearn.model_selection import train_test_split

#import library for logistical regression
from sklearn.linear_model import LogisticRegression

#import performance metrics- accuracy score and confusion
from sklearn.metrics import accuracy_score,confusion_matrix

#import data
import os
os.chdir("C:\pandas")
data_income=pd.read_csv('income.csv')

#creating copy of data_income
data=data_income.copy()

"""
#exploratory data analysis

#1. getting to know data
#2. dealing with missing values( data preprocessing)
#3. cross tables and data visualization

"""

#getting to know data
# checking data types of variables
print(data.info())

#checking for null values
data.isnull().sum()

#summary of numerical values
summary_num=data.describe()
print(summary_num)

#summary of categorial variables
summary_cate=data.describe(include='O')
print(summary_cate)

#frequency of each categories
data['JobType'].value_counts()
data['occupation'].value_counts()

#checking for unique classes
print(np.unique(data['JobType']))
print(np.unique(data['occupation']))

#reading na_values=[' ?'] with nan value
data=pd.read_csv("income.csv",na_values=[' ?'])


#data preprocessing
#checking for null values
data.isnull().sum()

missing=data[data.isnull().any(axis=1)]
#axis=1 : to consider at least one column value is missing



"""points to remember
#1. missing values in JobType=1809
#2. missing values in occupation type=1816
#3. there are two specific columns jobtype 
and occupation where 1809 values are missing
#4. also there are 7 more rows in jobtype where 
occupation is unfilled as job type is never worked
"""
#dropping missing rows
data2=data.dropna(axis=0)

#relationship between independent variables
correlation=data2.corr()

#cross table and data visualization

#extracting column names
data2.columns

#gender proportion table
gender=pd.crosstab(index=data2['gender'],columns='count',normalize=True)
print(gender)


#gender vs salary status
gender_salstat=pd.crosstab(index=data2['gender'],columns=data2['SalStat'],margins=True,normalize='index')
print(gender_salstat)



#frequency distribution of salary status
SalStat=sns.countplot(data2['SalStat'])
# 75% people<=50000
# 25% people>50000

#histogram of age
sns.displot(data2['age'],bins=10,kde=False)
#people age bw 20-45 are high in frequency


#box plot : age vs salary status
sns.boxplot(x='SalStat',y=data2['age'],data=data2)
data2.groupby('SalStat')['age'].median()
#people with age 25-35 <=50000 
#people with age 35-55> 50000

sns.countplot(y='JobType',data=data2)
sns.countplot(y='EdType',data=data2)
sns.countplot(y='occupation',data=data2)
sns.countplot(y='capitalgain',data=data2)
sns.countplot(y='capitalloss',data=data2)

sns.boxplot(x='SalStat',y=data2['hoursperweek'],data=data2)
data2.groupby('SalStat')['hoursperweek'].median()



#logical regression
#reindexing the salary stat names to 0,1
data2['SalStat']=data2['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])

new_data=pd.get_dummies(data2, drop_first=True)

#storing column names
columns_list=list(new_data.columns)
print(columns_list)

#separating input names from data
features=list(set(columns_list)-set(['SalStat']))
print(features)

#storing output value in y
y=new_data['SalStat'].values
print(y)

#storing the values from input features
x=new_data[features].values
print(x)

#splitting the data into train and test
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)

#make an instance of model
logistic=LogisticRegression()

#fitting the values of x and y
logistic.fit(train_x,train_y)
logistic.coef_
logistic.intercept_

#prediction from test_data
prediction=logistic.predict(test_x)
print(prediction)

#confusion matrix
confusion_matrix=confusion_matrix(test_y,prediction)
print(confusion_matrix)

#calculating accuracy
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)


#printing the misclassified values from prediction
print("missclassified prediction: %d" %(test_y != prediction).sum())


#logistical regression: removing insignificant variables
# Reindexing the salary status names to 0,1
data2['SalStat']=data2['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])

cols=['gender','nativecountry','race','JobType']
new_data=data2.drop(cols,axis=1)
new_data=pd.get_dummies(data2, drop_first=True)

#storing column names
columns_list=list(new_data.columns)
print(columns_list)

#separating input names from data
features=list(set(columns_list)-set(['SalStat']))
print(features)

#storing output value in y
y=new_data['SalStat'].values
print(y)

#storing the values from input features
x=new_data[features].values
print(x)

#splitting the data into train and test
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)

#make an instance of model
logistic=LogisticRegression()

#fitting the values of x and y
logistic.fit(train_x,train_y)
logistic.coef_
logistic.intercept_

#prediction from test_data
prediction=logistic.predict(test_x)
print(prediction)

#calculating accuracy
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)


#printing the misclassified values from prediction
print("missclassified prediction: %d" %(test_y != prediction).sum())



# Reindexing the salary status names to 0,1
data2['SalStat']=data2['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])

cols = ['gender','nativecountry','race','JobType']
new_data = data2.drop(cols,axis = 1)

new_data=pd.get_dummies(new_data, drop_first=True)

# Storing the column names 
columns_list2=list(new_data.columns)
print(columns_list2)

# Separating the input names from data
features2=list(set(columns_list2)-set(['SalStat']))
print(features2)

# Storing the output values in y
y2=new_data['SalStat'].values
print(y2)

# Storing the values from input features
x2 = new_data[features2].values
print(x2)

# Splitting the data into train and test
train_x2,test_x2,train_y2,test_y2 = train_test_split(x2,y2,test_size=0.3, random_state=0)

# Make an instance of the Model
logistic2 = LogisticRegression()

# Fitting the values for x and y
logistic2.fit(train_x2,train_y2)

# Prediction from test data
prediction2 = logistic2.predict(test_x2)

# Printing the misclassified values from prediction
print('Misclassified samples: %d' % (test_y2 != prediction2).sum())


#KNN

#importing knn library
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

#storing the k nearest neighbors classifier
KNN_classifier=KNeighborsClassifier(n_neighbors=5)

#fitting the values of x and y
KNN_classifier.fit(train_x,train_y)


#predicting the test values with model
prediction=KNN_classifier.predict(test_x)

# Performance metric check
confusionMmatrix = confusion_matrix(test_y, prediction)
print(confusionMmatrix)


# Calculating the accuracy
accuracy_score=accuracy_score(test_y, prediction)
print(accuracy_score)

print('Misclassified samples: %d' % (test_y != prediction).sum())


#Effect of K value on classifier

Misclassified_sample = []
# Calculating error for K values between 1 and 20
for i in range(1, 20):  
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(train_x, train_y)
    pred_i = knn.predict(test_x)
    Misclassified_sample.append((test_y != pred_i).sum())

print(Misclassified_sample)