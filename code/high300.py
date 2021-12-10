
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
from pyts.classification import LearningShapelets
from pyts.utils import windowed_view
from pyts.transformation import ShapeletTransform
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import math
import joblib
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from scipy import  stats
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima_model import ARIMA  
from statsmodels.tsa.arima_model import ARMA  


# In[2]:


shapelet = ShapeletTransform(n_shapelets=5,window_sizes=[3,4,5], random_state=42,sort=True)


# In[3]:


high=pd.read_csv("high.csv")


# In[4]:


high300=high.iloc[:,2]


# In[9]:


def GenerateSequences(high300):
    high300=np.array(high300)
    Xh300=[]
    for i in range(30,len(high300)-1):
        s=[]
        for j in range(i-30,i):
            s.append(high300[j])
        Xh300.append(s)
    return Xh300
def GenerateSet(Xh300,high300):
    x_train=Xh300[:-5]
    x_test=Xh300[-5:]
    y_train=high300[31:-5]
    y_test=high300[-5:]
    return x_train,x_test,y_train,y_test
def status(high300):
    Y=[]
    for i in range(31,len(high300)-5):
        if high300[i]/high300[i]<1:
            s=0
            Y.append(s)
        else:
            s=1
            Y.append(s)
    return Y
    


# In[6]:


Xh300=GenerateSequences(high300)


# In[10]:


xh300_train,xh300_test,yh300_train,yh300_test=GenerateSet(Xh300,high300)


# In[22]:


model_h300 = joblib.load("model_h300")
xh300t=model_h300.transform(Xh300)
close=pd.DataFrame(np.array(high300[30:-1]))
xh300t=pd.DataFrame(xh300t)
xh300_feature=pd.concat((close,xh300t),axis=1)


# In[24]:


xh300t=pd.DataFrame(xh300t).iloc[:,:-1]


# In[25]:


xh300_feature=pd.concat((close,xh300t),axis=1)


# In[23]:


model_h300.shapelets_


# In[26]:


x1_train=xh300_feature[:-5]
x1_test=xh300_feature[-5:]


# In[27]:


forest_shapelet1=RandomForestRegressor(n_estimators=150,random_state=100)
forest_shapelet1.fit(x1_train,yh300_train)
yh300_pred=forest_shapelet1.predict(x1_test)


# In[28]:


def evaluate(y_pred,y_test):
    s=0
    ae=0
    ape=0
    for i in range (0,len(y_pred)):
        s+=(y_pred[i]-y_test[i])**2
        ae+=abs(y_pred[i]-y_test[i])
        ape+=abs(y_pred[i]-y_test[i])/y_test[i]
    MSE=s/len(y_pred)
    MAE=ae/len(y_pred)
    MAPE=ape/len(y_pred)
    print("predict:",y_pred)
    print("really:",y_test)
    return MSE,MAE,MAPE


# In[29]:


evaluate(yh300_pred,np.array(yh300_test))


# In[20]:


forest=RandomForestRegressor(n_estimators=150,random_state=100)
forest.fit(xh300_train,yh300_train)
yh300_pred=forest.predict(xh300_test)


# In[21]:


evaluate(yh300_pred,np.array(yh300_test))

