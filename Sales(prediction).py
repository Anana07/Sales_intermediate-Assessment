#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


import warnings
warnings.filterwarnings("ignore")


# In[3]:


df=pd.read_csv("/content/train_v9rqX0R.csv")
df.head()


# In[4]:


DF=df.copy()


# In[5]:


df.shape


# In[6]:


df.tail()


# In[7]:


df.info()


# In[8]:


df.describe()


# In[9]:


plt.figure(figsize=(7,7))
sns.heatmap(df.corr(),annot=True,cmap="Blues")
plt.show()


# In[10]:


duplicated_df=df[df.duplicated()]
duplicated_df.shape


# No duplicate rows are present in the data set.

# In[11]:


df.columns


# ## **Pre Processing**

# In[12]:


col=df.select_dtypes(include=['object'])
col.columns


# In[13]:


from scipy.stats import chi2_contingency

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))

def find_object_correlation(dataframe):
    object_columns = dataframe.select_dtypes(include=['object']).columns
    correlation_matrix = pd.DataFrame(np.zeros((len(object_columns), len(object_columns))), index=object_columns, columns=object_columns)

    for i in range(len(object_columns)):
        for j in range(i, len(object_columns)):
            feature1 = dataframe[object_columns[i]]
            feature2 = dataframe[object_columns[j]]
            correlation = cramers_v(feature1, feature2)
            correlation_matrix.loc[object_columns[i], object_columns[j]] = correlation
            correlation_matrix.loc[object_columns[j], object_columns[i]] = correlation

    return correlation_matrix

# Example usage
# Assuming you have a DataFrame called 'df' with your data
correlation_matrix = find_object_correlation(df)
print(correlation_matrix)


# In[14]:


df.isnull().sum()


# **Handling missing values in Item_Weight**

# In[15]:


df.value_counts('Item_Weight')


# In[16]:


df['Item_Weight'].nunique()


# In[17]:


df['Item_Weight'].describe()


# In[18]:


df['Item_Weight'].fillna(df.groupby('Item_Type')['Item_Weight'].transform('median'), inplace=True)


# In[19]:


DF=df.copy()


# In[20]:


df.isnull().sum()


# In[21]:


Nan_df=df.loc[df.isnull().any(axis=1)]
Nan_df


# In[22]:


Nan_df.groupby(['Outlet_Type'])['Outlet_Location_Type'].value_counts()


# ## **Working with the copy of data for handling the missing value in Outlet_Size.**

# In[23]:


import statistics


# In[24]:


mode_size=DF.groupby(['Outlet_Type','Outlet_Location_Type'])['Outlet_Size'].agg(lambda x: statistics.mode(x) if len(x) > 0 else 'Unknown')


# In[25]:


print(mode_size)


# In[26]:


DF.isnull().sum()


# In[27]:


DF_NO_Nan=DF.dropna()


# In[28]:


DF_NO_Nan.isnull().sum()


# In[29]:


mode_size_DF_without_Nan=DF_NO_Nan.groupby(['Outlet_Type','Outlet_Location_Type'])['Outlet_Size'].agg(lambda x: statistics.mode(x) if len(x) > 0 else 'Unknown')
print(mode_size_DF_without_Nan)


# In[30]:


DF.groupby(['Outlet_Type','Outlet_Location_Type'])['Outlet_Size'].value_counts()


# I realised that all of the Outlet_Size values for Grocery Stores in Tier 3 are missing values.As a result, we are compelled to manage the lacking in this arrangement with the Tier 1 grocery store mode.
# 
# We handle the missing value "Small" in the Supermarket Type 1 Tier 2 situation after the mode drops missing values in "Small."

# ## ** original data**

# In[31]:


df.groupby(["Outlet_Type",'Outlet_Location_Type'])['Outlet_Size'].apply(lambda x: x.mode())


# In[32]:


df=df.fillna(df.groupby(["Outlet_Type",'Outlet_Location_Type'])['Outlet_Size'].apply(lambda x: x.mode()).iloc[0])


# In[33]:


df.isnull().sum()


# In[34]:


df.value_counts('Outlet_Size')


# **Handling Outliers**

# In[35]:


df.plot(subplots=True,kind="box",figsize=(20,6),title='Outlier Visualization')
plt.show()


# In[36]:


#handling outliers in Item_Visibility
plt.title("Boxplot of Item_Visibility")
plt.boxplot(df['Item_Visibility'])
plt.show()


# In[37]:


df['Item_Visibility'].describe()


# In[38]:


Q1=np.percentile(df['Item_Visibility'],25,interpolation='midpoint')
Q3=np.percentile(df['Item_Visibility'],75,interpolation='midpoint')
IQR=Q3-Q1
Max=Q3+(1.5*IQR)
Min=Q1-(1.5*IQR)
l=[]
for i in df['Item_Visibility']:
  if(i>Max)or(i<Min):
    l.append(i)

len(l)


# Due to the fact that there are only 144 outliers in Item_Visibility, which is quite little compared to the total number of data points. therefore discarding such anomalies.

# In[39]:


rows_to_drop = []
for index, value in enumerate(df['Item_Visibility']):
    if (value > Max) or (value < Min):
        rows_to_drop.append(index)

df = df.drop(rows_to_drop)


# In[40]:


df.shape


# In[41]:


#handling outliers in Item_Outlet_Sales
plt.title("Boxplot of Item_Outlet_Sales")
plt.boxplot(df['Item_Outlet_Sales'])
plt.show()


# Values in Item_Outlet_Sales cannot be considered as Outliers .So leave as it is.
# 
# 
# 
# 

# In[42]:


df.columns


# ## **ENCODING**

# In[43]:


df.dtypes


# In[44]:


df.Item_Fat_Content.nunique()


# In[45]:


df.Item_Identifier.nunique()


# In[46]:


df.Item_Type.nunique()


# In[47]:


df.Outlet_Identifier.nunique()


# In[48]:


df.Outlet_Size.nunique()


# In[49]:


df.Outlet_Location_Type.nunique()


# In[50]:


df.Outlet_Type.nunique()


# In[51]:


from sklearn.preprocessing import LabelEncoder


# In[52]:


le=LabelEncoder()
for i in ['Item_Identifier','Item_Fat_Content','Item_Type','Outlet_Identifier']:
  df[i]=le.fit_transform(df[i])


# In[53]:


for i in ['Outlet_Size','Outlet_Location_Type','Outlet_Type']:
  df=pd.get_dummies(df,columns=[i])


# In[54]:


df


# In[55]:


df.dtypes


# In[56]:


y=df['Item_Outlet_Sales']
x=df.drop('Item_Outlet_Sales',axis=1)


# **Splitting the data into train and test data**

# In[57]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)


# In[58]:


x_train.shape


# In[59]:


x_test.shape


# In[60]:


y_train.shape


# ## **MODELLING**

# In[61]:


from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score


# In[62]:


#Linear Regression
lr=linear_model.LinearRegression()
model=lr.fit(x_train,y_train)
predictions=model.predict(x_test)
print('MSE is :',mean_squared_error(y_test,predictions))
print('R squared value is :',r2_score(y_test,predictions))


# In[63]:


#decision tree regression
dr = DecisionTreeRegressor()
dr.fit(x_train,y_train)
predictions=dr.predict(x_test)
print('MSE is :',mean_squared_error(y_test,predictions))
print('R squared value is :',r2_score(y_test,predictions))


# MSE value of linear regression is less than MSE value of Decision tree regression ,so the better model is Linear regresssion.

# ## **Preprocessing the test data.**

# In[64]:


test_df=pd.read_csv("/content/test_AbJTz2l.csv")
test_df.head()


# In[65]:


test_DF=test_df.copy()


# In[66]:


test_df.shape


# In[67]:


test_df.info()


# In[68]:


test_df.describe()


# In[69]:


duplicated_df=test_df[test_df.duplicated()]
duplicated_df.shape


# In[70]:


test_df.isnull().sum()


# In[71]:


test_df.dtypes


# In[72]:


test_df['Item_Weight'].describe()


# In[73]:


test_df['Item_Weight'].fillna(test_df.groupby('Item_Type')['Item_Weight'].transform('median'), inplace=True)


# In[74]:


test_df.isnull().sum()


# In[75]:


Nan_df=test_df.loc[test_df.isnull().any(axis=1)]
Nan_df


# In[76]:


Nan_df.groupby(['Outlet_Type'])['Outlet_Location_Type'].value_counts()


# # Working with the copy of data for handling the missing value in Outlet_Size.

# In[77]:


test_DF=test_df.copy()


# In[78]:


mode_size=test_DF.groupby(['Outlet_Type','Outlet_Location_Type'])['Outlet_Size'].agg(lambda x: statistics.mode(x) if len(x) > 0 else 'Unknown')


# In[79]:


print(mode_size)


# This the same case as in the train data

# In[80]:


test_df=test_df.fillna(test_df.groupby(["Outlet_Type",'Outlet_Location_Type'])['Outlet_Size'].apply(lambda x: x.mode()).iloc[0])


# In[81]:


test_df.isnull().sum()


# ## **Handling Ouliers**

# In[82]:


test_df.plot(subplots=True,kind="box",figsize=(20,6),title='Outlier Visualization')
plt.show()


# In[83]:


#handling outliers in Item_Visibility
plt.title("Boxplot of Item_Visibility")
plt.boxplot(df['Item_Visibility'])
plt.show()


# In[84]:


df['Item_Visibility'].describe()


# In[85]:


Q1=np.percentile(test_df['Item_Visibility'],25,interpolation='midpoint')
Q3=np.percentile(test_df['Item_Visibility'],75,interpolation='midpoint')
IQR=Q3-Q1
Max=Q3+(1.5*IQR)
Min=Q1-(1.5*IQR)
l1=[]
for i in test_df['Item_Visibility']:
  if(i>Max)or(i<Min):
    l1.append(i)

len(l1)


# In[86]:


rows_to_drop = []
for index, value in enumerate(test_df['Item_Visibility']):
    if (value > Max) or (value < Min):
        rows_to_drop.append(index)

test_df =test_df.drop(rows_to_drop)


# In[87]:


test_df.shape


# In[88]:


test_df


# ## **ENCODING**

# In[89]:


le=LabelEncoder()
for i in ['Item_Identifier','Item_Fat_Content','Item_Type','Outlet_Identifier']:
  test_df[i]=le.fit_transform(test_df[i])


# In[90]:


for i in ['Outlet_Size','Outlet_Location_Type','Outlet_Type']:
  test_df=pd.get_dummies(test_df,columns=[i])


# **Now we predict the Item_Outlet_Sales of test data with the linear regression model**

# In[91]:


lr=LinearRegression()
model=lr.fit(x_train,y_train)
test_pred=model.predict(test_df)


# In[92]:


test_df['Item_Outlet_Sales']=test_pred


# In[93]:


test_df


# In[93]:




