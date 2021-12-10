
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


# In[4]:


low=pd.read_csv("low.csv")
low300=low.iloc[:,1]
Xl300=GenerateSequences(low300)
xl300_train,xl300_test,yl300_train,yl300_test=GenerateSet(Xl300,low300)


# In[5]:


y_train=status(low300)
model_l300=shapelet.fit(xl300_train, y_train)
save_model = "model_l300"
joblib.dump(model_l300,save_model) 
xl300t=model_l300.transform(Xl300)
close=pd.DataFrame(np.array(low300[30:-1]))
xl300t=pd.DataFrame(xl300t)
xl300_feature=pd.concat((close,xl300t),axis=1)


# In[6]:


model_l300.shapelets_


# In[7]:


x1_train=xl300_feature[:-5]
x1_test=xl300_feature[-5:]
forest_shapelet1=RandomForestRegressor(n_estimators=150,random_state=100)
forest_shapelet1.fit(x1_train,yl300_train)
yl300_pred=forest_shapelet1.predict(x1_test)


# In[8]:


evaluate(yl300_pred,np.array(yl300_test))


# In[9]:


forest=RandomForestRegressor(n_estimators=150,random_state=100)
forest.fit(xl300_train,yl300_train)
yl300_pred=forest.predict(xl300_test)
evaluate(yl300_pred,np.array(yl300_test))

