
# coding: utf-8

# In[2]:


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


# In[4]:


low=pd.read_csv("low.csv")
low50=low.iloc[:,2]
Xl50=GenerateSequences(low50)
xl50_train,xl50_test,yl50_train,yl50_test=GenerateSet(Xl50,low50)


# In[5]:


shapelet = ShapeletTransform(n_shapelets=5,window_sizes=[3,4,5], random_state=42,sort=True)
y_train=status(low50)
model_l50=shapelet.fit(xl50_train, y_train)
save_model = "model_l50"
joblib.dump(model_l50,save_model) 
xl50t=model_l50.transform(Xl50)
close=pd.DataFrame(np.array(low50[30:-1]))
xl50t=pd.DataFrame(xl50t)
xl50_feature=pd.concat((close,xl50t),axis=1)


# In[6]:


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


# In[7]:


x1_train=xl50_feature[:-5]
x1_test=xl50_feature[-5:]
forest_shapelet1=RandomForestRegressor(n_estimators=150,random_state=100)
forest_shapelet1.fit(x1_train,yl50_train)
yl50_pred=forest_shapelet1.predict(x1_test)
evaluate(yl50_pred,np.array(yl50_test))


# In[8]:


forest=RandomForestRegressor(n_estimators=150,random_state=100)
forest.fit(xl50_train,yl50_train)
yl50_pred=forest.predict(xl50_test)
evaluate(yl50_pred,np.array(yl50_test))

