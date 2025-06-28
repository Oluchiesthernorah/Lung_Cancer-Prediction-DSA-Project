# Lung_Cancer-Prediction-DSA-Project

# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# For ignoring warning
import warnings
warnings.filterwarnings("ignore")

# input/lung_cancer/lung_cancer.csv
# YES = 2 and No = 1
df=pd.read_csv('lung_cancer.csv')

df.shape
![Capture 3](https://github.com/user-attachments/assets/22aa17c4-885e-4076-af07-15d2b59f6b2b)


# To Check for Duplicates
df.duplicated().sum()
![Capture 4](https://github.com/user-attachments/assets/cd2ccc94-e7e6-4989-a94f-2d9eac4bfbc4)


# To Remove Duplicates
df=df.drop_duplicates()
![Capture 5](https://github.com/user-attachments/assets/09b2837b-8366-4615-9a08-b0cdddebac91)


# To Check for null values
df.isnull().sum()


#Display index, columns and data
df.info()
![Capture 7](https://github.com/user-attachments/assets/21afcc86-7bee-42c1-9114-7e27be7fd1be)

#Summary Statistics
df.describe()
![Capture 8](https://github.com/user-attachments/assets/09c933f1-1924-4def-adbe-721b9ccce0a2)

# Display the first 5 rows
df.head()
![Capture 9](https://github.com/user-attachments/assets/8b604379-c333-4a46-8b86-5475f8773c18)

# Display the first 5 rows
df.tail()
![Capture 10](https://github.com/user-attachments/assets/ba432050-7977-40e5-ba66-6ff1e4ae9527)



# convert gender to numeric values using LabelEncoder from sklearn. , Yes(MALE) = 1 and No(FEMALE)= 0
from sklearn import preprocessing
le=preprocessing.LabelEncoder()
df['GENDER']=le.fit_transform(df['GENDER'])
df['LUNG_CANCER']=le.fit_transform(df['LUNG_CANCER'])
df['SMOKING']=le.fit_transform(df['SMOKING'])
df['YELLOW_FINGERS']=le.fit_transform(df['YELLOW_FINGERS'])
df['ANXIETY']=le.fit_transform(df['ANXIETY'])
df['PEER_PRESSURE']=le.fit_transform(df['PEER_PRESSURE'])
df['CHRONIC DISEASE']=le.fit_transform(df['CHRONIC DISEASE'])
df['FATIGUE ']=le.fit_transform(df['FATIGUE '])
df['ALLERGY ']=le.fit_transform(df['ALLERGY '])
df['WHEEZING']=le.fit_transform(df['WHEEZING'])
df['ALCOHOL CONSUMING']=le.fit_transform(df['ALCOHOL CONSUMING'])
df['COUGHING']=le.fit_transform(df['COUGHING'])
df['SHORTNESS OF BREATH']=le.fit_transform(df['SHORTNESS OF BREATH'])
df['SWALLOWING DIFFICULTY']=le.fit_transform(df['SWALLOWING DIFFICULTY'])
df['CHEST PAIN']=le.fit_transform(df['CHEST PAIN'])
df['LUNG_CANCER']=le.fit_transform(df['LUNG_CANCER'])

![Capture 11](https://github.com/user-attachments/assets/95c0cb2d-795b-4712-914b-1aeaab060b35)


# To check
df
![Capture 12](https://github.com/user-attachments/assets/6431a40f-68ad-4d76-ba15-7833675d9a3a)





#
df.info()
![Capture 13](https://github.com/user-attachments/assets/cc1b9109-5660-4025-820b-b970abfc02ca)



# To check the distributaion of Target variable.
sns.countplot(x='LUNG_CANCER', data=df,)
plt.title('Target Distribution');
![Capture 14](https://github.com/user-attachments/assets/2b65b28c-ab27-4744-8682-656ff841c141)



df['LUNG_CANCER'].value_counts()
![Capture 15](https://github.com/user-attachments/assets/d5d4e07a-adbc-4207-81f3-994576f632e4)


# To visualize data
def plot(col, df=df):
    return df.groupby(col)['LUNG_CANCER'].value_counts(normalize=True).unstack().plot(kind='bar', figsize=(8,5))


plot('GENDER')
![Capture 16](https://github.com/user-attachments/assets/75cd049a-1fb6-47d4-b3b6-7f96a3aee723)

plot('AGE')
![Capture 17](https://github.com/user-attachments/assets/ca97cb3e-e5c3-4af7-90ac-8845724f44e7)

plot('SMOKING')
![Capture 18](https://github.com/user-attachments/assets/7e3f7129-318d-4d65-a71f-733fb742b800)

plot('YELLOW_FINGERS')
![Capture 19](https://github.com/user-attachments/assets/72792b52-3711-405d-bbc0-528c520b01fc)

plot('ANXIETY')
![Capture 20](https://github.com/user-attachments/assets/ce01f178-b1ea-4fcc-8b17-daac08bd6b9e)

plot('PEER_PRESSURE')
![Capture 21](https://github.com/user-attachments/assets/2c7174a1-1da6-4f75-8818-fb24b0910965)

plot('CHRONIC DISEASE')
![Capture 22](https://github.com/user-attachments/assets/b63c8654-4776-4477-8daa-b1dcd5edcb78)

plot('FATIGUE ')
![Capture 23](https://github.com/user-attachments/assets/fed0b24f-8062-467c-9017-1bb3b7ec49e7)

plot('ALLERGY ')
![Capture 24](https://github.com/user-attachments/assets/64fff87f-5aaa-40bd-81de-bd74bebcd2b3)

plot('WHEEZING')
![Capture 25](https://github.com/user-attachments/assets/eee27bfe-26df-4e68-961e-d735da5c49a6)

plot('ALCOHOL CONSUMING')
![Capture 26](https://github.com/user-attachments/assets/cf613c9a-f3be-49d9-8baa-550313d7c8f3)

plot('COUGHING')
![Capture 27](https://github.com/user-attachments/assets/a5c46b8f-a375-4be5-810f-9da918fdfbda)

plot('SHORTNESS OF BREATH')
![Capture 28](https://github.com/user-attachments/assets/3c05a3e8-2d63-43b8-bfb5-9807328f10f9)

plot('SWALLOWING DIFFICULTY')
![Capture 29](https://github.com/user-attachments/assets/a104f67d-5ebd-42da-83b1-237e0906bc92)

plot('CHEST PAIN')
![Capture 30](https://github.com/user-attachments/assets/f11e4985-5b85-4ca2-b450-0fc360a5432f)



#GENDER, AGE and SHORTNESS OF BREATH dont have that much relationship with LUNG CANCER. 
#Hence we drop those features to make the dataset more clean.
df_new=df.drop(columns=['GENDER','AGE', 'SHORTNESS OF BREATH'])
df_new
![Capture 31](https://github.com/user-attachments/assets/53cb5ec4-99fd-4988-9e37-1c07ab9e8c93)


CORRELATION
#Finding Correlation
cn=df_new.corr()




#Correlation 
cmap=sns.diverging_palette(260,-10,s=50, l=75, n=6,
as_cmap=True)
plt.subplots(figsize=(18,18))
sns.heatmap(cn,cmap=cmap,annot=True, square=True)
plt.show()
![Capture 32](https://github.com/user-attachments/assets/11d2986e-5b91-4b0b-ad5d-5b78c02fdff2)


![Capture 33](https://github.com/user-attachments/assets/32c62808-f3a8-40e1-94ab-d0a14f4944cb)



kot = cn[cn>=.40]
plt.figure(figsize=(12,8))
sns.heatmap(kot, cmap="Purples")
![Capture 34](https://github.com/user-attachments/assets/24bd5fba-086b-435d-abac-cbee908aa799)



![Capture 35](https://github.com/user-attachments/assets/80555beb-89a0-4553-8492-4ba9f4617b1b)


#The correlation matrix shows that ANXIETY and YELLOW_FINGERS are correlated more than 50%
#To create a new feature combiningthem
df_new['ANXYELFIN']=df_new['ANXIETY']*df_new['YELLOW_FINGERS'] 
df_new
![Capture 36](https://github.com/user-attachments/assets/f8405dfb-c90c-43e6-bd66-0552dc7d6644)




#Splitting independent and dependent variables
X = df_new.drop('LUNG_CANCER', axis =1)
y = df_new['LUNG_CANCER']



#Target Distribution Imbalance Handling
from imblearn.over_sampling import ADASYN
adasyn = ADASYN(random_state=42)
X, y = adasyn.fit_resample(X, y)


len(X)

![Capture 37](https://github.com/user-attachments/assets/3b491c03-41f6-46da-af37-fbd778408c8d)


#Logistic Regression
#Splitting data for training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size= 0.25, random_state=0)

#Fitting training data to the model
from sklearn.linear_model import LogisticRegression
lr_model=LogisticRegression(random_state=0)
lr_model.fit(X_train, y_train)

![Capture 38](https://github.com/user-attachments/assets/a527cdf5-7965-4a26-9eda-09206176a5eb)


#Predicting result using testing data
y_lr_pred= lr_model.predict(X_test)
y_lr_pred
![Capture 39](https://github.com/user-attachments/assets/38036be8-0522-4bf8-8764-54bf642d435b)


#Model accuracy
from sklearn.metrics import classification_report, accuracy_score, f1_score
lr_cr=classification_report(y_test, y_lr_pred)
print(lr_cr)
![Capture p1](https://github.com/user-attachments/assets/e2225e25-71ee-4cae-a340-15c15a33f58d)





#Random Forest
#Training
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
RandomForestClassifier()
![Capture p2](https://github.com/user-attachments/assets/e8391c3b-4572-4373-a458-ef0a0fd753a2)




#Predicting result using testing data
y_rf_pred= rf_model.predict(X_test)
y_rf_pred
![Capture p3](https://github.com/user-attachments/assets/3a31404c-e1cb-46f3-b8c7-9e05d5a4a2e6)




#Model accuracy
rf_cr=classification_report(y_test, y_rf_pred)
print(rf_cr)
![Capture p4](https://github.com/user-attachments/assets/9c38be7b-5a8b-4a47-890a-dd2f073a91a4)





#XGBoost
from xgboost import XGBClassifier
xgb_model = XGBClassifier()
xgb_model.fit(X_train, y_train)
![Capture 40](https://github.com/user-attachments/assets/a6a5b7c1-37c4-48cd-ac12-bcb972888a5a)


#Predicting result using testing data
y_xgb_pred= xgb_model.predict(X_test)
y_xgb_pred

![Capture 41](https://github.com/user-attachments/assets/f3ff1c53-4efb-4c2a-9326-e122e38ebcb2)


#Model accuracy
xgb_cr=classification_report(y_test, y_xgb_pred)
print(xgb_cr)
![Capture 42](https://github.com/user-attachments/assets/825a7393-6a4d-45e4-ad9b-2432fa31e6ca)

# K-Fold Cross Validation

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

k = 10
kf = KFold(n_splits=k, shuffle=True, random_state=42)


# Logistic regerssion model
lr_model_scores = cross_val_score(lr_model,X, y, cv=kf)

# Decision tree model
dt_model_scores = cross_val_score(dt_model,X, y, cv=kf)

print("Logistic regression models' average accuracy:", np.mean(lr_model_scores))
print("Random forest models' average accuracy:", np.mean(rf_model_scores))
print("XGBoost models' average accuracy:", np.mean(xgb_model_scores))
![Capture 43](https://github.com/user-attachments/assets/a206b0aa-396b-4b08-9326-77e8ca4b85d3)







