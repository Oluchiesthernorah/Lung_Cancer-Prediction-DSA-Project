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


# To Check for Duplicates
df.duplicated().sum()


# To Remove Duplicates
df=df.drop_duplicates()


# To Check for null values
df.isnull().sum()

#
df.info()

#
df.describe()

# Display the first 5 rows
df.head()

# Display the first 5 rows
df.tail()



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



# To check
df





#
df.info()



# To check the distributaion of Target variable.
sns.countplot(x='LUNG_CANCER', data=df,)
plt.title('Target Distribution');



df['LUNG_CANCER'].value_counts()


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




![Capture 31](https://github.com/user-attachments/assets/53cb5ec4-99fd-4988-9e37-1c07ab9e8c93)

