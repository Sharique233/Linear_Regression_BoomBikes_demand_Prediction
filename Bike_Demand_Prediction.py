#!/usr/bin/env python
# coding: utf-8

# ## Problem_Statement: You are required to model the demand for shared bikes with the available independent variables. It will be used by the management to understand how exactly the demands vary with different features. They can accordingly manipulate the business strategy to meet the demand levels and meet the customer's expectations. Further, the model will be a good way for management to understand the demand dynamics of a new market. 

# ##  Importing libraries

# In[38]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# In[2]:


# Load the dataset
data = pd.read_csv('day(1).csv')
data


# In[3]:


data.head()


# ## Data Understanding and Preparation

# In[4]:


print(data.info())


# In[5]:


# Check for missing values
print(data.isnull().sum())


# In[6]:


# Handle missing values
data = data.dropna()
data


# In[7]:


# Check for duplicate rows
print(f"Duplicate rows: {data.duplicated().sum()}")


# In[8]:


#Handle Categorical Variable-> Convert categorical variables to dummies
data = pd.get_dummies(data, drop_first=True)
data


# ## Data Preprocessing:

# In[9]:


#Drop irrelevant columns (if any exist)
columns_to_drop = ['instant', 'dteday']  # Assuming these columns are irrelevant
data.drop(columns=columns_to_drop, inplace=True, errors='ignore')
data


# In[10]:


# Handle categorical variables
categorical_cols = ['season', 'yr', 'mnth', 'weathersit']
encoder = OneHotEncoder(drop='first', sparse=False)
categorical_encoded = pd.DataFrame(encoder.fit_transform(data[categorical_cols]), 
                                   columns=encoder.get_feature_names_out(categorical_cols))
data = pd.concat([data, categorical_encoded], axis=1)
data.drop(columns=categorical_cols, inplace=True)
data


# In[11]:


# Standardize numerical columns
scaler = StandardScaler()
numeric_cols = ['temp', 'atemp', 'hum', 'windspeed']
data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
data[numeric_cols]


# ## Data Visualization:

# #### Understand the Target Variable (cnt):

# In[16]:


sns.histplot(data['cnt'], kde=True)
plt.title('Distribution of Total Bike Rentals')
plt.show()


# #### Correlation Analysis:

# #### Visualize Relationships:

# In[17]:


# Analyze correlation between features
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# In[19]:


sns.pairplot(data, vars=['temp', 'hum', 'windspeed', 'cnt'])


# ## Model Building:

# In[20]:


# Define target and features
X = data.drop(columns=['casual', 'registered', 'cnt'])
y = data['cnt']


# In[21]:


# Split data
X = data.drop(columns=['cnt', 'casual', 'registered'])
y = data['cnt']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[22]:


# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[23]:


# Build the model
model=lr = LinearRegression()
lr.fit(X_train_scaled, y_train)


# In[24]:


# Feature selection using RFE
rfe = RFE(lr, n_features_to_select=10)
rfe.fit(X_train, y_train)
selected_features = X_train.columns[rfe.support_]
print(f"Selected Features: {list(selected_features)}")


# In[25]:


# Refit model with selected features
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]
lr.fit(X_train_selected, y_train)


# In[26]:


# Predictions and evaluation
y_pred_train = lr.predict(X_train_selected)
y_pred_test = lr.predict(X_test_selected)


# In[27]:


# Evaluate model performance
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)
train_rmse = mean_squared_error(y_train, y_pred_train, squared=False)
test_rmse = mean_squared_error(y_test, y_pred_test, squared=False)


# In[28]:


print(f"Training R2: {train_r2}")
print(f"Test R2: {test_r2}")
print(f"Training RMSE: {train_rmse}")
print(f"Test RMSE: {test_rmse}")


# ## Model Interpretation
# 

# ### Residual Analysis: Checked residual distribution and scatter plots for assumptions of linear regression.

# In[33]:


# Residual Analysis
residuals = y_test - y_pred_test
plt.figure(figsize=(8, 5))
sns.histplot(residuals, kde=True)
plt.title("Residual Distribution")
plt.show()


# In[34]:


#Distribution of Residuals:
sns.histplot(residuals, kde=True)
plt.title('Distribution of Residuals')
plt.show()


# In[35]:


plt.figure(figsize=(8, 5))
plt.scatter(y_test, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.title("Residuals vs Predicted Values")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.show()


# In[36]:


# Variance Inflation Factor (VIF) for multicollinearity check
vif_data = pd.DataFrame()
vif_data['Feature'] = X_train_selected.columns
vif_data['VIF'] = [variance_inflation_factor(X_train_selected.values, i) for i in range(X_train_selected.shape[1])]
print(vif_data.sort_values(by='VIF', ascending=False))


# In[39]:


# Final Model Interpretation
final_model = sm.OLS(y_train, sm.add_constant(X_train_selected)).fit()
print(final_model.summary())


# In[ ]:




