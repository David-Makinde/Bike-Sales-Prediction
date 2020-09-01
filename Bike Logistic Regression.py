#!/usr/bin/env python
# coding: utf-8

# In[14]:


from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.model_selection import train_test_split
import pandas as pd


# In[4]:


data = pd.read_csv('C:/Users/user/Documents/Data Science/biketrain_preprocessed.csv')


# In[5]:


data.columns


# In[8]:


data = data[[ 'BikeBuyer', 'HomeOwnerFlag', 'NumberCarsOwned', 'NumberChildrenAtHome',
       'TotalChildren', 'YearlyIncome', 'AveMonthSpend',
       'CountryRegionName_Australia', 'CountryRegionName_Canada',
       'CountryRegionName_France', 'CountryRegionName_Germany',
       'CountryRegionName_United Kingdom', 'CountryRegionName_United States',
       'Education_Graduate', 'Education_High School', 'Occupation_Clerical',
       'Occupation_Management', 'Occupation_Manual', 'Occupation_Professional',
       'Gender_F', 'Gender_M', 'MaritalStatus_M', 'MaritalStatus_S', 'Age']]


# In[9]:


data.head()


# In[10]:


#Creating feature array
X = data.iloc[:,1:]


# In[11]:


X.head()


# In[12]:


Y = data.iloc[:,0]


# In[13]:


#Output array
Y.head()


# In[15]:


#Classifier
classifier = LogReg(solver='lbfgs',random_state=0)


# In[16]:


classifier.fit(X,Y)


# In[17]:


#Checking Accuracy
'Accuracy:{:2f}'.format(classifier.score(X,Y))


# In[23]:


#Lets predict for the test data
X2 = pd.read_csv('C:/Users/user/Documents/Data Science/biketest_preprocessed.csv')


# In[24]:


X2.head()


# In[25]:


predicted_y = classifier.predict(X2)


# In[26]:


predicted_y


# In[27]:


#Checking accuracy
'Accuracy:{:2f}'.format(classifier.score(X2,predicted_y))


# In[30]:


Y2 = pd.DataFrame(predicted_y, columns=['BikeBuyer']).to_csv('C:/Users/user/Documents/Data Science/bikesales_prediction.csv')


# In[31]:


#Saving the Model
import pickle
pickle.dump(classifier, open('C:/Users/user/Documents/Data Science/bike_prediction_model.sav', 'wb'))


# In[32]:


##To open te model later
model = pickle.load(open('C:/Users/user/Documents/Data Science/bike_prediction_model.sav','rb'))


# In[ ]:




