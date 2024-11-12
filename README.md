# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages
2. Import the dataset to operate on
3. Split the dataset.
4. Predict the required output

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Padmavathi M
RegisterNumber: 212223040141
*/
```
```
import pandas as pd
data=pd.read_csv("spam.csv",encoding='latin-1')
data.head()
data.info()
data.isnull().sum()
x=data["v1"].values
y=data["v2"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extractiaon.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```
## Output:
## Data Head:
![image](https://github.com/user-attachments/assets/684b0b3c-660e-4fff-8a53-c74e66f5b09d)

## Data Info:
![image](https://github.com/user-attachments/assets/8b7fba5d-28a6-48e8-9d1d-e51998af96e8)

## Data isnull():
![image](https://github.com/user-attachments/assets/d9dd6aec-b75e-4a4e-86fe-ac4a2cb0e7d2)

## y_pred:
![image](https://github.com/user-attachments/assets/f2d3934a-d216-4860-84ce-1a4e3be442e1)

## Accuracy:
![image](https://github.com/user-attachments/assets/6e3905c5-e7b1-479a-82d9-2d9d09345ee4)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
