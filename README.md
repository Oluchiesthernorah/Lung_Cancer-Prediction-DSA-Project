# Lung_Cancer-Prediction-DSA-Project

#Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#For ignoring warning
import warnings
warnings.filterwarnings("ignore")

# input/lung_cancer/lung_cancer.csv
# YES = 2 and No = 1
df=pd.read_csv('lung_cancer.csv')

df.shape


#To Check for Duplicates
df.duplicated().sum()


#To Remove Duplicates
df=df.drop_duplicates()


#To Check for null values
df.isnull().sum()

#
df.info()

#
df.describe()

# Display the first 5 rows
df.head()

# Display the first 5 rows
df.tail()





#convert gender to numeric values using LabelEncoder from sklearn. , Yes(MALE) = 1 and No(FEMALE)= 0
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



#To check
df





#
df.info()



#To check the distributaion of Target variable.
sns.countplot(x='LUNG_CANCER', data=df,)
plt.title('Target Distribution');



df['LUNG_CANCER'].value_counts()


# To visualize data
def plot(col, df=df):
    return df.groupby(col)['LUNG_CANCER'].value_counts(normalize=True).unstack().plot(kind='bar', figsize=(8,5))


plot('GENDER')

plot('AGE')

plot('SMOKING')

plot('YELLOW_FINGERS')

plot('ANXIETY')

plot('PEER_PRESSURE')

plot('CHRONIC DISEASE')

plot('FATIGUE ')

plot('ALLERGY ')

plot('WHEEZING')

plot('ALCOHOL CONSUMING')

plot('COUGHING')

plot('SHORTNESS OF BREATH')

plot('SWALLOWING DIFFICULTY')

plot('CHEST PAIN')
