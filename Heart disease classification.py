#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB


# In[3]:


dataset=pd.read_csv(r"C:\Users\DELL\Downloads\archive.zip")


# In[4]:


dataset


# In[5]:


dataset.info()


# In[146]:


dataset.isnull().sum()


# In[147]:


#part of preprocessing and no null values


# In[148]:


x=dataset.iloc[:,0:13]


# In[149]:


x


# In[150]:


y=dataset.iloc[:,13:14]


# In[151]:


y


# In[152]:


xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.20)


# In[153]:


xtrain.shape


# In[154]:


xtest.shape


# In[155]:


model1=RandomForestClassifier()


# In[156]:


model1.fit(xtrain,ytrain)


# In[157]:


prediction1=model1.predict(xtest)


# In[158]:


prediction1


# In[159]:


ac1=accuracy_score(ytest,prediction1)


# In[160]:


ac1


# In[161]:


model2=DecisionTreeClassifier()


# In[162]:


model2.fit(xtrain,ytrain)


# In[163]:


prediction2=model2.predict(xtest)


# In[164]:


prediction2


# In[165]:


ac2=accuracy_score(ytest,prediction2)


# In[166]:


ac2


# In[167]:


model3=SVC()


# In[168]:


model3.fit(xtrain,ytrain)


# In[169]:


prediction3=model3.predict(xtest)


# In[170]:


prediction3


# In[171]:


ac3=accuracy_score(prediction3,ytest)


# In[172]:


ac3


# In[173]:


model4=GaussianNB()


# In[174]:


model4.fit(xtrain,ytrain)


# In[175]:


prediction4=model4.predict(xtest)


# In[176]:


prediction4


# In[177]:


ac4=accuracy_score(prediction4,ytest)


# In[178]:


ac4


# In[179]:


#Confusion_marix


# In[181]:


cm=confusion_matrix(ytest,prediction2)


# In[182]:


cm


# In[185]:


report=classification_report(ytest,prediction2)


# In[187]:


print(report)


# In[ ]:




