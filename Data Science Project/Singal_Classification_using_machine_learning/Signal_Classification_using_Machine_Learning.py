# Step1: Import the necessary Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Step2 : Load the data set
df = pd.read_csv("ECGCvdata.csv")
print(df.head())

# Step3 : Data information
print(df.info())

# Step4 : Find out the data set columns and rows
print(df.shape)

# Step5 :  Data Describe
print(df.describe())

#Observation : In data description its take numerical data only

# Let read the Columns Names in the data set
Column_names = df.columns
print(Column_names)

# let find out the categorical data and column name
Categorical_Feature = [feature for feature in df.columns if df[feature].dtypes == 'O']
print('Number of Categorical_Feature_columns:', len(Categorical_Feature))

# let find out the categorical data and column name
Categorical_Feature = [feature for feature in df.columns if df[feature].dtypes == 'O']
print('Number of Categorical_Feature_columns:', Categorical_Feature)

# now check out the number of missing values in the columns
Missing_value = [feature for feature in df.columns if df[feature].isnull().sum()>1]
print('Number of Missing Value in dataset column:', Missing_value)

Missing_value = [feature for feature in df.columns if df[feature].isnull().sum()>1]
print('Number of Missing Value in dataset column:', len(Missing_value))

Missing_value = [feature for feature in df.columns if df[feature].isnull().sum()>1]
print(Missing_value)

# we will check the missing value total
print(df.isnull().sum())

print(df['STslope'].head())

print(df['STslope'].tail())

#Observation: we fing out the missing value are in the middle and some case are either top or bottom of dataset

# Now check out the duplicated rows are present or not
print(df.duplicated())

# let's find out toral number of Duplicated rows if there are present
print(df.duplicated().sum())

# We will check the number of nuique in ECG_Signal Coulmn
ECG_Signal = df['ECG_signal'].unique()
print(ECG_Signal)

#Observation: We observed four major symptoms in the ECG_signal column.

# lets find out the how many 'ARR' value in ECG_Signal
ECG_Signal = [feature for feature in df['ECG_signal'] if 'ARR' in feature]
print('Number of ARR in ECG column:', ECG_Signal)

# lets find out the how many 'ARR' value in ECG_Signal
ECG_Signal = [feature for feature in df['ECG_signal'] if 'ARR' in feature]
print('Number of ARR in ECG column:', len(ECG_Signal))

# lets find out the how many 'ARR' value in ECG_Signal
ECG_Signal = [feature for feature in df['ECG_signal'] if 'AFF' in feature]
print('Number of AFF in ECG column:', len(ECG_Signal))

ECG_Signal = [feature for feature in df['ECG_signal'] if 'CHF' in feature]
print('Number of CHF in ECG column:', len(ECG_Signal))

ECG_Signal = [feature for feature in df['ECG_signal'] if 'CHF' in feature]
print('Number of CHF in ECG column:', len(ECG_Signal))

'''
#Observation: We observed that most of the values in the ECG_Signal column are evenly distributed across each specified category. #Upon analyzing the ECG_Signal column, it was observed that the values are fairly evenly distributed across the defined categories. 
Based on this distribution, the ECG_Signal column is selected as the target feature, as it may help in identifying the type of symptoms present in future input datasets.
'''

# IN simple way
print(df['ECG_signal'].value_counts())

import seaborn as sns
import matplotlib.pyplot as plt

# Example for a few main features:
main_features = ['hbpermin', 'Pseg', 'QRSseg', 'QTseg']

for feature in main_features:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x='ECG_signal', y=feature, data=df, palette='rainbow')
    plt.title(f'{feature} vs ECG_signal')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

'''
#Observation

Chat1 (Heartbeat per minute vs ECG_Signal):

CHF (Congestive Heart Failure) patients consistently show higher heart rates compared to all other groups.
AFF (Atrial Fibrillation) and ARR (Arrhythmia) exhibit the most variability, especially with extreme values (indicative of unstable cardiac rhythms).
NSR (Normal Sinus Rhythm) shows a tighter distribution, confirming regular, normal rhythms.
PSegment (Pseg):

Corresponds to the P wave, which reflects atrial depolarization (atrial activity in the heart).
QRS Complex:

Represents ventricular depolarization, the electrical activity that triggers ventricular contraction.
Components:
Q wave – Initial negative deflection.
R wave – Sharp upward peak.
S wave – Negative deflection after the R wave.
QT Segment (Qtseg):

Measures the time between the start of the Q wave and the end of the T wave.
Indicates the total time for ventricular electrical activity, including both depolarization and repolarization.
'''

import seaborn as sns
import matplotlib.pyplot as plt

# Example for a few main features as barplot:
main_features = ['hbpermin', 'Pseg', 'QRSseg', 'QTseg']

for feature in main_features:
    plt.figure(figsize=(6, 4))
    sns.barplot(x='ECG_signal', y=feature, data=df, palette = 'Set1')
    plt.title(f'{feature} vs ECG_signal')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# We can use the Heatmap for missing values
plt.figure(figsize=(10, 8))
sns.heatmap(df.isnull(), yticklabels = False, cbar= False, cmap = "viridis")
plt.title("Heatmap of Feature Missing")
plt.show()

#We will compare the difference between all Hbpermin with ecg_signal
main_features = ['hbpermin','Pseg', 'QRSseg', 'QTseg']  # Example features

for feature in main_features:
    plt.figure(figsize=(6, 4))
    plt.scatter(df[feature], df['ECG_signal'])
    plt.xlabel(feature)
    plt.ylabel('ECG_signal')
    plt.title(f'{feature} vs ECG_signal')
    plt.show()

#distribution of main feature test data
import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(data=df, x='Pseg', kde=True, color='red')
plt.title('Distribution of Pseg')
plt.xlabel('Pseg')
plt.ylabel('Frequency')
plt.show()


sns.histplot(data=df, x='hbpermin', kde=True, color='red') #'hbpermin', 'Pseg', 'QRSseg', 'QTseg'
plt.title('Distribution of HeartBeat per Minute')
plt.xlabel('HeartBeat per Minute')
plt.ylabel('Frequency')
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

features = ['hbpermin', 'Pseg', 'QRSseg', 'QTseg']

for feature in features:
    plt.figure(figsize=(6, 4))
    sns.histplot(data=df, x=feature, kde=True, color='skyblue')
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

Missing_value = [feature for feature in df.columns if df[feature].isnull().sum()>1]
print('Number of Missing Value in dataset column:', Missing_value)

# we will fix the missing the value in the feature columns

from sklearn.impute import SimpleImputer

imputer_object = SimpleImputer(strategy = "median")
df["QRtoQSdur"] = imputer_object.fit_transform(df[["QRtoQSdur"]])
df["QRtoQSdur"].head(30)

from sklearn.impute import SimpleImputer

feature = ['RStoQSdur','PonPQang','PQRang','QRSang','STToffang','RSTang','QRslope','RSslope']
imputer_object = SimpleImputer(strategy = "mean")
df[feature] = imputer_object.fit_transform(df[feature])
df[feature].tail()

# check the null data is replaced
print(df[feature].isnull().sum())

sns.heatmap(df.isnull(), yticklabels = False, cbar= False, cmap = "viridis")
plt.show()

# let look the enteires data set
plt.figure(figsize = (12,8))
sns.countplot(x = "hbpermin", hue = "ECG_signal", data =df, palette ="BrBG" )
plt.show()

sns.set_style("whitegrid")
g = sns.pairplot(data=df, hue="ECG_signal", palette="rainbow")
g._legend.set_title("ECG Signal Type")
new_labels = ["Arrhythmia (ARR)", "Atrial Fibrillation (AFF)",
              "Normal Sinus Rhythm (NSR)", "Congestive Heart Failure (CHF)"]
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()


print(df['ECG_signal'].unique())

import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x="ECG_signal", data=df, palette="pastel")
plt.title("Count of Each ARR Category")
plt.xlabel("ECG_signal")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()

# let look the corelation
plt.figure(figsize = (12,10))
sns.heatmap(df.corr(), cmap='coolwarm')
plt.show()

'''
Converting the Categorical Feature
We'll need to convert categorical features to dummy variables using pandas! Otherwise our machine learning algorithm '
   won't be able to directly take in those features as inputs.
'''

print(df.info())

from sklearn.preprocessing import LabelEncoder

label_object = LabelEncoder()
df["ECG_signal"] = label_object.fit_transform(df["ECG_signal"])
df["ECG_signal"].head()

print(df['ECG_signal'].unique())


'''
Building a Logistic Regression model
Let's start by splitting our data into a training set and test set (there is another test.csv file that you can play 
around with in case you want to use all this data for training).
'''

# set feature and target
x =df.drop('ECG_signal',axis= True)
y =df['ECG_signal']

# splitting the data into train and test for training the model
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size =0.3 , random_state = 42)

# Model Calling
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(x_train, y_train)

# prediction
prediction  = model.predict(x_test)
print(prediction)

# accuarcy
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
accuarcy = confusion_matrix(y_test, prediction)
print(accuarcy)

score = accuracy_score(y_test, prediction)
print(score)

from sklearn.metrics import classification_report

print(classification_report(y_test,prediction))

y_pred = model.predict(x_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_object.classes_))

print("\nAccuracy Score:", accuracy_score(y_test, y_pred))

print("Class Labels:", label_object.classes_)

import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_object.classes_, yticklabels=label_object.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, x, y, cv=5)
print("Cross-validation accuracy:", scores.mean())


