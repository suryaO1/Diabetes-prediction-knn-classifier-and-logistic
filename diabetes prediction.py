#!/usr/bin/env python
# coding: utf-8

# In[1]:


import scipy as stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv('diabetes.csv')


# In[3]:


df.head()


# In[4]:


df.isnull().values.any()


# In[5]:


df.isnull().sum()


# In[6]:


df.info()


# In[7]:


df.shape


# In[8]:


df.describe()


# In[9]:


df.skew()


# In[10]:


df.columns


# In[11]:


df.Outcome.value_counts()


# In[12]:


correlationmatrix = df.corr()
correlationmatrix


# In[13]:


plt.figure(figsize=(10,10))
Heatmp = sns.heatmap(correlationmatrix,annot=True,square=True)
Heatmp


# In[14]:


x = df.drop('Outcome',axis='columns')


# In[15]:


y = df[['Outcome']]


# In[16]:


from imblearn.over_sampling import SMOTE
oversampling = SMOTE(sampling_strategy='minority')
x,y=oversampling.fit_resample(x,y)


# In[17]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)


# In[18]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x = pd.DataFrame(scaler.fit_transform(x),columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'Pedigree', 'Age'])
x.head()


# In[19]:


from sklearn.linear_model import LogisticRegression
clff = LogisticRegression()
clff.fit(x_train,y_train)


# In[20]:


y_pred=clff.predict(x_test)


# In[22]:


clff.score(x_test,y_test)


# In[23]:


from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_pred,y_test)
cm


# In[24]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x = pd.DataFrame(scaler.fit_transform(x),columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'Pedigree', 'Age'])
x.head()


# In[25]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)


# In[26]:


x_train.shape


# In[61]:


from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=5,metric='minkowski')
clf.fit(x_train,y_train)


# In[62]:


y_pred = clf.predict(x_test)


# In[63]:


clf.score(x_test,y_test)


# In[64]:


from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_pred,y_test)
cm


# In[ ]:





# In[ ]:





# In[ ]:




