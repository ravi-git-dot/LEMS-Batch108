#!/usr/bin/env python
# coding: utf-8

# ## Flight Price Predection(EDA + Feature Engineering)

# In[1]:


# import Libraies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df_train = pd.read_excel('Data_Train.xlsx')
df_train.head()


# In[3]:


df_test = pd.read_excel('Test_set.xlsx')
df_test


# In[4]:


final_df = pd.concat([df_train, df_test], ignore_index = True)
final_df.head()


# In[5]:


# read the  columns
df_columns = final_df.columns
df_columns


# In[6]:


# read the total data set like rows and columns
final_df.shape


# In[7]:


# data information
final_df.info()


# In[8]:


# data describe
final_df.describe()


# In[9]:


final_df.tail()


# In[10]:


final_df['Airline'].unique()


# # feature Engineering process

# In[11]:


## doing the split of data_of_jounary
final_df['Date_of_Journey'].str.split('/')


# In[12]:


# we can split the data only
final_df['Date_of_Journey'].str.split('/').str[0]


# In[13]:


# # Now split the data of Jounary 
# final_df['Data'] = final_df['Date_of_Journey'].str.split('/').str[0]
# final_df['Month'] = final_df['Date_of_Journey'].str.split('/').str[1]
# final_df['Year'] = final_df['Date_of_Journey'].str.split('/').str[2]


# In[14]:


final_df.head()


# In[15]:


## apply lambda function
final_df["Date"]=final_df['Date_of_Journey'].apply(lambda x:x.split("/")[0])
final_df["Month"]=final_df['Date_of_Journey'].apply(lambda x:x.split("/")[1])
final_df["Year"]=final_df['Date_of_Journey'].apply(lambda x:x.split("/")[2])


# In[16]:


final_df.info()


# In[17]:


final_df.head()


# In[18]:


# converting the data type
final_df['Date'] = final_df['Date'].astype(int)
final_df['Month'] = final_df['Month'].astype(int)
final_df['Year'] = final_df['Year'].astype(int)


# In[19]:


final_df.info()


# In[20]:


# drop the data_of_jounery
final_df.drop('Date_of_Journey', axis = 1, inplace = True)


# In[21]:


final_df.head()


# In[22]:


# let focus on the arival time 
final_df['Arrival_Time'].str.split(' ')


# In[23]:


final_df['Arrival_Time'].str.split(' ').str[0]


# In[24]:


# Appy lambda function
final_df['Arrival_Time'].apply(lambda x:x.split(' ')[0])


# In[25]:


final_df['Arrival_Time'] = final_df['Arrival_Time'].apply(lambda x:x.split(' ')[0])


# In[26]:


final_df.head()


# In[27]:


# find out the null value 
final_df.isnull().sum()


# In[28]:


# check out the duplicated values or rows
final_df.duplicated()


# In[29]:


df_duplicated = final_df.duplicated().sum()
df_duplicated


# In[30]:


# find out the duplicated rows columns

[features for features in final_df.columns if final_df[features].duplicated().sum() > 0]


# In[31]:


final_df.drop_duplicates(keep=False, inplace=True) 


# In[32]:


final_df.duplicated().sum()


# In[33]:


print(final_df.shape)

# In[34]:


final_df.head()


# In[35]:


# split the arrival time into time and hours
final_df['Arrival_hour'] =final_df['Arrival_Time'].apply(lambda x : x.split(':')[0])
final_df['Arrival_min'] =final_df['Arrival_Time'].apply(lambda x : x.split(':')[1])


# In[36]:


final_df.head()


# In[37]:


final_df['Arrival_hour'] = final_df['Arrival_hour'].astype(int)
final_df['Arrival_min'] = final_df['Arrival_min'].astype(int)


# In[38]:


final_df.info()


# In[39]:


# final drop the arival time columns
final_df.drop('Arrival_Time', axis =1 ,inplace =True)


# In[40]:


final_df.head()


# In[41]:


# know focus on Dep_Time
final_df['Dep_Hour'] = final_df['Dep_Time'].apply(lambda x : x.split(':')[0])
final_df['Dep_Min'] = final_df['Dep_Time'].apply(lambda x : x.split(':')[1])


# In[42]:


final_df['Dep_Hour'] =final_df['Dep_Hour'].astype(int)
final_df['Dep_Min'] = final_df['Dep_Min'].astype(int)


# In[43]:


final_df.info()


# In[44]:


final_df.head()


# In[45]:


final_df.drop('Dep_Time',axis = 1, inplace =True)


# In[46]:


final_df.head(1)


# In[47]:


final_df['Total_Stops'].unique()


# In[48]:


final_df['Total_Stops'].isnull().sum()


# In[49]:


# # check out which data is nan value
# final_df[final_df['Total_Stops'].isnull()]


# In[50]:


final_df['Total_Stops'].fillna('1 stop', inplace=True)


# In[51]:


final_df['Total_Stops'].isnull().sum()


# In[52]:


# from sklearn.preprocessing import LabelEncoder
# label_object =LabelEncoder() 
# final_df['Total_Stops'] = label_object.fit_transform(final_df['Total_Stops'])
# final_df.head()


# In[53]:


final_df['Total_Stops'] = final_df['Total_Stops'].map({'non-stop': 0, '2 stops':1, '1 stop':2, '3 stops':3, '4 stops':4})


# In[54]:


final_df.head()


# In[55]:


# now we can drop route columns
final_df.drop('Route', axis =1 , inplace = True)


# In[56]:


final_df.head()


# In[57]:


final_df['Additional_Info'].unique()


# In[58]:


final_df.info()


# In[59]:


# convert the duration form hours to minutes
final_df['Duration'].str.split(' ').str[0]


# In[60]:


type(final_df['Duration'].str.split(' ').str[0])


# In[61]:


final_df['Duration'].str.split(' ').str[0].str.split('h')


# In[62]:


final_df['Duration'].str.split(' ').str[0].str.split('h').str[0]


# In[63]:


final_df['Duration_hours'] = final_df['Duration'].str.split(' ').str[0].str.split('h').str[0]


# In[64]:


final_df.head()


# In[65]:


final_df[final_df['Duration'] == '5m']


# In[66]:


# probably to drop the above two rows
final_df.drop(6474, axis =0, inplace = True)
final_df.drop(13343, axis = 0, inplace =True)
final_df.reset_index(drop=True, inplace=True)


# In[67]:


final_df['Duration_hours'] = final_df['Duration_hours'].astype(int)


# In[68]:


final_df[final_df['Duration'] == '5m']


# In[69]:


#checkout any missing value
final_df[final_df['Duration'].isnull()]


# In[70]:


final_df['Duration'].str.split(' ').str[1]


# In[71]:


final_df['Duration_Min'] = final_df['Duration'].str.split(' ').str[1]


# In[72]:


final_df['Duration_Min'] = final_df['Duration'].str.split(' ').str[1].str.split('m')


# In[73]:


final_df.head()


# In[74]:


final_df['Duration_Min'] = final_df['Duration'].str.split(' ').str[1].str.split(',')


# In[75]:


final_df.head()


# In[76]:


final_df['Duration_Min'] = final_df['Duration'].str.split(' ').str[1].str.split('m').str[0]


# In[77]:


final_df.head()


# In[78]:


final_df['Duration_Min'].isnull().sum()


# In[79]:


final_df['Duration_Min'].fillna(0, inplace = True)


# In[80]:


final_df['Duration_Min'].isnull().sum()


# In[81]:


final_df.info()


# In[82]:


final_df['Duration_Min'] = final_df['Duration_Min'].astype(int)


# In[83]:


final_df.head()


# In[85]:


# now drop the duration column
final_df.drop('Duration', axis =1, inplace =True)


# In[86]:


final_df.head(1)


# In[87]:


final_df['Source'].unique()


# In[88]:


final_df['Airline'].unique()


# In[89]:


from sklearn.preprocessing import LabelEncoder
label_object =LabelEncoder()
final_df['Airline'] = label_object.fit_transform(final_df['Airline'])
final_df['Source'] = label_object.fit_transform(final_df['Source'])
final_df['Additional_Info'] = label_object.fit_transform(final_df['Additional_Info'])
final_df['Destination'] = label_object.fit_transform(final_df['Destination'])


# In[90]:


final_df.head()


# In[91]:


final_df.info()


# In[92]:


# this is also converting categorical data into numerical data
# from sklearn.preprocessing import OneHotEncoder 
# ohe =OneHotEncoder()
# final_df['Airline'] = ohe.fit_transform(final_df['Airline'])


# In[93]:


pd.get_dummies(final_df, columns = ['Airline', 'Source', 'Destination','Additional_Info'], drop_first= True)


# In[94]:


final_df.head()


# In[98]:


import matplotlib.pyplot as plt
import seaborn as sns

# Check missing values
print("Missing values in Price column:", final_df['Price'].isnull().sum())

# Plot distribution
plt.figure(figsize=(8, 5))
sns.histplot(final_df['Price'], bins=30, kde=True, color='skyblue', edgecolor='black')
plt.title('Distribution of Price Column', fontsize=16)
plt.xlabel('Price', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.grid(True)
plt.show()


# In[100]:


from sklearn.impute import SimpleImputer

# Use median strategy for right-skewed numeric data
medianImputer = SimpleImputer(strategy='median')
final_df['Price'] = medianImputer.fit_transform(final_df[['Price']])
final_df.tail()


# In[101]:


print('Missing value in price columns', final_df['Price'].isnull().sum())


# In[102]:


# let set feature and target data set

x = final_df.drop('Price', axis =1)
y =final_df['Price']


# In[103]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size =0.2, random_state = 42)


# In[104]:


# let train the model 
model = LinearRegression()
model.fit(x_train, y_train)


# In[107]:


from sklearn.metrics import mean_squared_error, r2_score
y_pred = model.predict(x_test)
print("R2 Score:", r2_score(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred))


# In[108]:


## Observation poor accarcy try another model
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators = 100, random_state =42)
model.fit(x_train, y_train)


# In[109]:


y_pred = model.predict(x_test)
print("R2 Score:", r2_score(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred))


# In[110]:


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators = 100, random_state =42)
model.fit(x_train, y_train)


# In[111]:


y_pred = model.predict(x_test)
print("R2 Score:", r2_score(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred))


# In[112]:


get_ipython().system('pip install lightgbm')


# In[117]:


get_ipython().system('pip install xgboost')


# In[120]:


from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor(random_state=42)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

print("R2 Score:", r2_score(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred))







