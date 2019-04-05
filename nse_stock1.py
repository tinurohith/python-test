
# coding: utf-8

# In[1]:


import os


# In[2]:


import pandas as pd
from pandas import DataFrame
import numpy as np


# In[3]:


import urllib
from urllib.request import urlretrieve


# In[4]:


url="https://github.com/swapniljariwala/nsepy"


# In[5]:


from nsepy import get_history
from datetime import date


# In[6]:


nse_tcs=get_history(symbol="TCS",start=date(2015,1,1),end=date(2015,12,31))


# In[7]:


nse_tcs.columns


# In[8]:


nse_infy=get_history(symbol="INFY",start=date(2015,1,1),end=date(2015,12,31))


# In[9]:


nse_infy.columns


# In[10]:


nifty_it=get_history(symbol="NIFTY IT",start=date(2015,1,1),end=date(2015,12,31))


# In[11]:


nifty_it.columns


# In[12]:


nse_infy.head()


# In[13]:


nse_tcs.head()


# In[14]:


nifty_it.head()


# In[15]:


nse_infy.shape


# In[16]:


nse_tcs.shape


# In[17]:


nse_infy.describe()


# In[18]:


nse_tcs.describe()


# In[19]:


nse_infy.isnull().sum()


# In[20]:


nse_tcs.isnull().sum()


# In[21]:


nse_infy.info()


# In[22]:


nse_tcs.info()


# In[23]:


nifty_it.shape


# In[24]:


nse_infy['Series'].unique()


# In[25]:


nse_tcs['Series'].unique()


# In[26]:


def movingaverage(x,w):
   return pd.Series(x.rolling(window=w,min_periods=0).mean())


# In[27]:


nse_infy['4weeks']=movingaverage(nse_infy['Close'],20)
nse_infy['16weeks']=movingaverage(nse_infy['Close'],80)
nse_infy['28weeks']=movingaverage(nse_infy['Close'],140)
nse_infy['40weeks']=movingaverage(nse_infy['Close'],200)
nse_infy['52weeks']=movingaverage(nse_infy['Close'],260)
nse_tcs['4weeks']=movingaverage(nse_tcs['Close'],20)
nse_tcs['16weeks']=movingaverage(nse_tcs['Close'],80)
nse_tcs['28weeks']=movingaverage(nse_tcs['Close'],140)
nse_tcs['40weeks']=movingaverage(nse_tcs['Close'],200)
nse_tcs['52weeks']=movingaverage(nse_tcs['Close'],260)


# In[28]:


nse_infy[['Close','4weeks','16weeks','28weeks','40weeks','52weeks']].tail()


# In[29]:


nse_tcs[['Close','4weeks','16weeks','28weeks','40weeks','52weeks']].tail()


# In[30]:


nse_infy.tail()


# In[31]:


nse_tcs.tail()


# In[32]:


def volumeshocks(data):
    data['PreviousVolume']=data['Volume'].shift(1)
    data['VolumeShocks'] = (data['Volume']>data['PreviousVolume']*0.1+data['PreviousVolume']).map({True:0,False:1})
    return data


# In[33]:


nse_infy=volumeshocks(nse_infy)
nse_tcs=volumeshocks(nse_tcs)


# In[34]:


nse_infy[['Volume','PreviousVolume','VolumeShocks']].head()


# In[35]:


nse_tcs[['Volume','PreviousVolume','VolumeShocks']].head()


# In[36]:


def priceshocks(data):
    data['T']=data['Close'].shift(1)
    data['PriceShocks'] = (data['Close']-data['T']>0.20*(data['Close']-data['T'])).map({True:0,False:1})
    return data


# In[37]:


nse_infy=priceshocks(nse_infy)
nse_tcs=priceshocks(nse_tcs)


# In[38]:


nse_infy[['Close','PriceShocks']].head()


# In[39]:


nse_infy[['Close','PriceShocks']].head()


# In[40]:


def blackswan(data):
    data['T1']=data['Prev Close'].shift(1)
    data['BlackSwanPrice'] = (data['Prev Close']-data['T1']>0.20*(data['Prev Close']-data['T1'])).map({True:0,False:1})
    return data


# In[41]:


nse_infy=blackswan(nse_infy)
nse_tcs=blackswan(nse_tcs)


# In[42]:


nse_infy[['Prev Close','BlackSwanPrice']].head()


# In[43]:


nse_tcs[['Prev Close','BlackSwanPrice']].head()


# In[44]:


def priceshocknovolshock(data):
    data['notvolshock']  = (~(data['VolumeShocks'].astype(bool))).astype(int)
    data['PriceshockNovolumeshocks'] = data['notvolshock'] & data['PriceShocks']
    return data


# In[45]:


nse_infy=priceshocknovolshock(nse_infy)
nse_tcs=priceshocknovolshock(nse_tcs)


# In[46]:


nse_infy[['VolumeShocks','PriceShocks','PriceshockNovolumeshocks']].head()


# In[47]:


nse_tcs[['VolumeShocks','PriceShocks','PriceshockNovolumeshocks']].head()


# In[48]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[49]:


nse_infy.Close.plot(figsize=(20,10),linewidth=5,fontsize=20,grid=True)
plt.title("Close price of INFY")


# In[50]:


nse_tcs.Close.plot(figsize=(20,10),linewidth=5,fontsize=20,grid=True)
plt.title("Close price of TCS")


# In[51]:


nse_infy.Close.plot(figsize=(20,10),linewidth=5,fontsize=20,grid=True)
nse_tcs.Close.plot(figsize=(20,10),linewidth=5,fontsize=20,grid=True)
plt.title("Close price of INFY and TCS")
plt.show()


# In[52]:


nse_infy[['4weeks','16weeks','28weeks','40weeks','52weeks']].plot(grid=True,figsize=(20,10),linewidth=5,fontsize=20)
plt.title("Moving Average Of INFY")


# In[53]:


nse_tcs[['4weeks','16weeks','28weeks','40weeks','52weeks']].plot(grid=True,figsize=(20,10),linewidth=5,fontsize=20)
plt.title("Moving Average Of TCS")


# In[54]:


plt.hist(nse_infy.VolumeShocks,color='red')
plt.title("Volume Shock Of INFY")


# In[55]:


plt.hist(nse_tcs.VolumeShocks,color='red')
plt.title("Volume Shock Of TCS")


# In[56]:


nse_infy['52weeks'].plot(grid=True,color='blue',figsize=(20,10),linewidth=5,fontsize=20)
nse_tcs['52weeks'].plot(grid=True,color='green')
plt.title("Moving Average-52weeks of INFY and TCS")


# In[57]:


nse_infy[['Close','Volume']].plot(secondary_y=['Volume'],grid=True,figsize=(20,10),linewidth=5,fontsize=20)


# In[58]:


nse_tcs[['Close','Volume']].plot(secondary_y=['Volume'],grid=True,figsize=(20,10),linewidth=5,fontsize=20)


# In[59]:


from pandas.tools.plotting import autocorrelation_plot
autocorrelation_plot(np.log(nse_infy['Volume']))


# In[60]:


autocorrelation_plot(np.log(nse_tcs['Volume']))


# In[61]:


autocorrelation_plot(np.log(nse_infy['Close']))


# In[62]:


autocorrelation_plot(np.log(nse_tcs['Close']))


# In[63]:


nse_infy=nse_infy.drop(['T','notvolshock','PreviousVolume','T1'],axis=1).head()


# In[64]:


nse_tcs=nse_tcs.drop(['T','notvolshock','PreviousVolume','T1'],axis=1).head()


# In[65]:


x=nse_infy.drop(['Close','Symbol','Series'],axis=1)
y=nse_infy['Close']


# In[66]:


import sklearn.model_selection as model_selection
x_train,x_test,y_train,y_test=model_selection.train_test_split(x,y,test_size=0.20,random_state=200)


# In[67]:


import sklearn.linear_model as linear_model


# In[68]:


reg=linear_model.Ridge(normalize=True,fit_intercept=True)
reg=reg.fit(x_train,y_train)


# In[69]:


greg=model_selection.GridSearchCV(reg,param_grid={'alpha':np.arange(0.1,100,1)})
greg=greg.fit(x_train,y_train)
greg.best_params_


# In[70]:


reg=linear_model.Ridge(normalize=True,fit_intercept=True,alpha=99.1)
reg=reg.fit(x_train,y_train)
reg.coef_


# In[71]:


reg.intercept_


# In[72]:


import sklearn.metrics as metrics
import sklearn.preprocessing as preprocessing


# In[73]:


x_test=preprocessing.normalize(x_test)


# In[76]:


print('MAE:',metrics.mean_squared_error(y_test,greg.predict(x_test)))


# In[80]:


x1=nse_tcs.drop(['Close','Symbol','Series'],axis=1)
y1=nse_tcs['Close']
x1_train,x1_test,y1_train,y1_test=model_selection.train_test_split(x1,y1,test_size=0.20,random_state=200)


# In[81]:


reg1=linear_model.Lasso(max_iter=1000,normalize=True)
reg1=reg1.fit(x1_train,y1_train)


# In[82]:


greg1=model_selection.GridSearchCV(reg1,param_grid={'alpha':np.arange(0.1,100,1).tolist()})
greg1=greg1.fit(x1_train,y1_train)
greg1.best_params_


# In[83]:


reg1=linear_model.Lasso(max_iter=1000,normalize=True,alpha=14.1)
reg1=reg1.fit(x1_train,y1_train)
reg1.coef_


# In[84]:


reg1.intercept_


# In[85]:


x1_test=preprocessing.normalize(x1_test)


# In[86]:


print('MAE:',metrics.mean_squared_error(y1_test,greg1.predict(x1_test)))


# In[87]:


import sklearn.tree as tree
from sklearn.ensemble import GradientBoostingClassifier
x2=nse_infy.drop(['VolumeShocks','Symbol','Series'],axis=1)
x2=pd.get_dummies(x2)
y2=nse_infy['VolumeShocks']
x2_train,x2_test,y2_train,y2_test=model_selection.train_test_split(x2,y2,test_size=0.20,random_state=200)


# In[88]:


clf=GradientBoostingClassifier(n_estimators=80,random_state=400)
clf.fit(x2_train,y2_train)


# In[89]:


clf.score(x2_test,y2_test)


# In[90]:


clf.feature_importances_


# In[91]:


feature=pd.Series(clf.feature_importances_,index=x.columns)
feature.sort_values(ascending=False)
feature.sort_values(ascending=False).plot(kind='bar',figsize=(10,5),grid=True)


# In[92]:


x3=nse_tcs.drop(['VolumeShocks','Symbol','Series'],axis=1)
x3=pd.get_dummies(x3)
y3=nse_tcs['VolumeShocks']
x3_train,x3_test,y3_train,y3_test=model_selection.train_test_split(x3,y3,test_size=0.20,random_state=200)


# In[93]:


clf1=GradientBoostingClassifier(n_estimators=80,random_state=400)
clf1.fit(x3_train,y3_train)


# In[94]:


clf1.score(x3_test,y3_test)


# In[95]:


clf1.feature_importances_


# In[98]:


feature1=pd.Series(clf1.feature_importances_,index=x.columns)
feature1.sort_values(ascending=False)
feature1.sort_values(ascending=False).plot(kind='bar',figsize=(10,5),grid=True)

