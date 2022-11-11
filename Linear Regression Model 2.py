#!/usr/bin/env python
# coding: utf-8

# ### Aim
# 
# To build a multiple linear regression model in python that predicts the score of the players in premier soccer league.
# 
# ### Data Description
# 
# The dataset used is the soccer player dataset. It has information about various players from different clubs, and it provides data over ten features with a number of goals as the target variable.

# ### Approach
# Import the required libraries and dataset
# 
# Check for the correlation between features
# 
# Plot a graph for correlations
# 
# Remove the weakly correlated and highly multicollinear variables
# 
# Perform train test split on the dataset
# 
# Fit the multiple linear regression model
# 
# Convert categorical variables into dummy/indicator variables
# 
# Plot the results

# In[1]:


# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import seaborn as sns


# In[3]:


#Import the required dataset
df = pd.read_csv('EPL_Soccer_MLR_LR.csv')


# ### Explore the dataset and check for the correlation between features

# In[6]:


df.columns


# In[7]:


#Get basic description of data, looking for the spread of the different variable, along with abrupt changes between
#the minimum, 25th, 50th, 75th and max.
df.describe()


# In[8]:


#Check the For correlation between the columns. the closer to 1 the more correlation determined between columns
corr = df.corr()
corr


# In[11]:


#create a colored graph to visual represent correlation
ax = sns.heatmap(
    corr,
    vmin = -1, vmax = 1, center = 0,
    cmap = sns.diverging_palette(20, 220, n=200),
    square = True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation = 45,
    horizontalalignment = "right"
);


# Identify weakly correlated variables: weight, height
# Remove: Height and Weight
# 
# Identify the multicollinear predictors: Minutes to Goal Ratio, Shots Per game
# Remove: Minutes to Goal Ratio

# In[20]:


columns = df.columns
columns


# In[21]:


#Extract predictor variables
X = df[['DistanceCovered(InKms)', 'Goals','ShotsPerGame', 'AgentCharges', 'BMI', 'Cost','PreviousClubCost',]]
y = df[['Score']]


# In[22]:


#Splitting with 75% trainign, 25% testing data
x_train, x_test, y_train, y_test = train_test_split(X, y, train_size = 0.75,
                                                   test_size = 0.25, random_state = 100)


# In[23]:


#Force intercept term
x_train_with_intercept = sm.add_constant(x_train)
lr = sm.OLS(y_train, x_train_with_intercept).fit()
lr.summary()


# In[24]:


#After looking at P>|t| values determined that I can trim down some variables to see if this is a better m,odel.
X = df[['DistanceCovered(InKms)', 'BMI', 'Cost','PreviousClubCost']]

#Retrain the data with the changes
x_train, x_test, y_train, y_test = train_test_split(X, y, train_size = 0.75,
                                                   test_size = 0.25, random_state = 100)

#Force intercept term
x_train_with_intercept = sm.add_constant(x_train)
lr = sm.OLS(y_train, x_train_with_intercept).fit()
lr.summary()


# In[25]:


#Let's try throwing in club
clubs = set(df.Club)
clubs #CHE, MUN, LIV


#to encode text into a number that is easier for the machine learnign maodel to understand
nominal_features = pd.get_dummies(df['Club'])
nominal_features


# In[26]:


#Concat is when you are combining two dataframes. Only works when you have the same number of row in each dataset and should join side by side.
df_encoded = pd.concat([df, nominal_features], axis = 1)
df_encoded


# In[28]:


X = df_encoded[['DistanceCovered(InKms)', 'BMI', 'Cost','PreviousClubCost', 'CHE','MUN','LIV']]


# In[29]:


#Retrain the data with the changes
x_train, x_test, y_train, y_test = train_test_split(X, y, train_size = 0.75,
                                                   test_size = 0.25, random_state = 100)

#Force intercept term
x_train_with_intercept = sm.add_constant(x_train)
lr = sm.OLS(y_train, x_train_with_intercept).fit()
lr.summary()


# In[30]:


#Look at the model plot
x_test_with_intercept = sm.add_constant(x_test)
y_test_fitted = lr.predict(x_test_with_intercept)

plt.scatter(y_test_fitted, y_test)
#plt.plot(x_test, y_test_fitted, 'r')
plt.show()


# In[ ]:




