#!/usr/bin/env python
# coding: utf-8

# In[24]:


#Importing Dependencies
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[3]:


#Data collection and processing
heart=pd.read_csv('heart.csv')


# In[4]:


heart.head()


# In[5]:


heart.tail()


# In[6]:


#No of rows and columns in dataset
heart.shape


# In[7]:


heart.info


# In[8]:


#Missing Values
heart.isnull().sum()


# In[9]:


#statical measures of data
heart.describe()


# In[11]:


#Distribution of Trget variables
heart['target'].value_counts()


# In[13]:


#Splitting features
X=heart.drop(columns='target',axis=1)
Y=heart['target']


# In[14]:


print(X)


# In[15]:


Y


# In[19]:


#Treain-Test Split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=.20,stratify=Y,random_state=2)


# In[20]:


X_train.shape


# In[21]:


X_test.shape


# In[25]:


#Model Training : Logistic Regression

model=LogisticRegression()


# In[26]:


model.fit(X_train,Y_train)


# In[32]:


#Model Evaluation
#Accuracy Score
X_test_pred=model.predict(X_test)
train_acc=accuracy_score(X_test_pred,Y_test)


# In[33]:


train_acc


# In[38]:


#Buliding Predictive System
input_data=(59,1,0,164,176,1,0,90,0,1,1,2,1)

#Change input data to a numpy array
input_data_as_np=np.asarray(input_data)

#Reshaping numpy array for single prediction
input_data_reshape=input_data_as_np.reshape(1,-1)

prediction=model.predict(input_data_reshape)
prediction

if(prediction[0]==0):
    print("Person is free from disease")
else:
    print("Person is suffering from heart disease")

