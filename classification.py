import pandas as pd
#import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection  import train_test_split
from sklearn.tree import DecisionTreeClassifier

## set random seed
seed = 726
## read data
adult = pd.read_csv('adult.csv')
adult = adult.drop(columns='fnlwgt')
x_train, x_test, y_train, y_test = train_test_split(adult.iloc[:,0:-1],adult.iloc[:,-1],test_size=0.15, random_state=seed)
adult_train = pd.concat([x_train,y_train],axis=1)
adult_test = pd.concat([x_test,y_test],axis=1)

## Question 1
missing_value = adult[adult.isnull().values==True].shape[0]
row_with_missing_value = adult.isnull().any(axis=1).sum()

print '\n Answer to Question 1: '
print '  (i) Number of instances: %d'% adult.shape[0]
print '  (ii) Number of missing values: %d'% missing_value
print '  (iii) fraction of missing values over all attribute values: %.2f%%'% (100*float(missing_value)/(adult.shape[0]*adult.shape[1]))
print '  (iv) Number of instances with missing value: %d'% row_with_missing_value
print '  (v) Fraction of instances with missing values over all instances: %.2f%%'% (100*float(row_with_missing_value)/float(adult.shape[0]))

## Question 2
le = LabelEncoder()
adult_encode = pd.DataFrame()
for i in adult.columns:
    adult_encode[i] = le.fit_transform(adult[i])

print '\n Answer to Question 2: '
print '  The set of all possible discrete values for each attribute: '
for i in adult.columns:
    print '  ',i,': ',adult_encode[i].unique()

## Question 3
data = adult_encode[adult.notnull().all(axis=1)]
adult_train_encode = pd.DataFrame()
adult_test_encode = pd.DataFrame()
for i in adult.columns:
    adult_train_encode[i] = le.fit_transform(adult_train[i])
    adult_test_encode[i] = le.fit_transform(adult_test[i])

dt = DecisionTreeClassifier(random_state=seed)
adult_train_drop = adult_train_encode[adult.notnull().all(axis=1)]
adult_test_drop = adult_test_encode[adult.notnull().all(axis=1)]
dt.fit(adult_train_drop.iloc[:,0:-1],adult_train_drop.iloc[:,-1])
score = dt.score(adult_test_drop.iloc[:,0:-1],adult_test_drop.iloc[:,-1])

print '\n Answer to Question 3: '
print '  Error rate: %.2f%%'% (100*(1-score))

## Question 4
D_with_nan = adult_encode[adult.isnull().any(axis=1)]
half_num = len(D_with_nan)
D_without_nan = adult_encode[adult.notnull().all(axis=1)]
random_instances = D_without_nan.sample(n=half_num,random_state=seed)
# D1
D_1 = pd.concat([D_with_nan,random_instances],axis=0)
dt.fit(D_1.iloc[:,0:-1],D_1.iloc[:,-1])
score1 = dt.score(adult_test_encode.iloc[:,0:-1],adult_test_encode.iloc[:,-1])
print '\n Answer to Question 4: '
print '  Error rate of D1: %.2f%%'% (100*(1-score1))

# D2
values = {}
for i in adult.columns:
    values[i] = adult[i].mode()[0]
adult_fillwithcommon = adult.fillna(value=values)
le1 = LabelEncoder()
D2 = pd.DataFrame()
for i in adult.columns:
    D2[i] = le1.fit_transform(adult_fillwithcommon[i])
dt.fit(D2.iloc[:,0:-1],D2.iloc[:,-1])

adult_test_fillna_encode = pd.DataFrame()
adult_test_fillna = adult_test.fillna(value=values)
for i in adult.columns:
    adult_test_fillna_encode[i] = le1.fit_transform(adult_test_fillna[i])
score2 = dt.score(adult_test_fillna_encode.iloc[:,0:-1],adult_test_fillna_encode.iloc[:,-1])
print '  Error rate of D2: %.2f%%'% (100*(1-score2))
