
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math


# In[2]:


hx=pd.read_excel("600015.xlsx")


# In[3]:


hx.head()


# In[4]:


hx_data=hx.iloc[:,1:10]


# In[5]:


hx_data.head()


# PCA

# In[6]:


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# In[7]:


scaler=StandardScaler()
scaler.fit(hx_data)
hxs=scaler.transform(hx_data)


# In[8]:


pca=PCA(n_components=2)
pca.fit(hxs)


# In[9]:


hx_pca=pca.transform(hxs)


# In[10]:


len(hx_pca)


# GRNN

# In[96]:


x_train=hx_pca[0:-6]


# In[97]:


y_train=np.array(hx.iloc[:-6,-2])


# In[98]:


x_test=hx_pca[-6:-1]


# In[99]:


y_test=hx.iloc[-6:-1,-2]


# In[108]:


x_test[:,1]


# In[ ]:


def transform(x,y):
    


# In[17]:


len(x_train)


# In[91]:


scaler1=StandardScaler()
scaler1.fit(x_train)
x_train=scaler1.transform(x_train)


# In[92]:


x_test=scaler1.transform(x_test)


# In[94]:


u=np.mean(y_train)


# In[95]:


np.std(y_train)


# In[109]:


close=list(hx.iloc[:,2])
X=[]
for i in range(30,len(close)):
    s=[]
    for j in range(i-30,i):
        s.append(close[j])
    X.append(s)


# In[123]:


X_train=np.array(X[0:-6])
X_test=np.array(X[-6:-1])


# In[68]:


def distance(X,Y):
    return np.sqrt(np.sum(np.square(X-Y)))

def distance_mat(trainX,testX):
    m,n = np.shape(trainX)
    p = np.shape(testX)[0]
    Euclidean_D = np.mat(np.zeros((p,m)))
    for i in range(p):
        for j in range(m):
            Euclidean_D[i,j] = distance(testX[i,:],trainX[j,:])
    return Euclidean_D

def Gauss(Euclidean_D,sigma):
    m,n = np.shape(Euclidean_D)
    Gauss = np.mat(np.zeros((m,n)))
    for i in range(m):
        for j in range(n):
            Gauss[i,j] = math.exp(- Euclidean_D[i,j] / (2 * (sigma ** 2)))
    return Gauss


# In[ ]:


def sum_layer(Gauss,x_train,trY):
    m=len(x_train)
    l=len(trY)
    n = 1
    sum_mat = np.mat(np.zeros((m,n+1)))
    ## 对所有模式层神经元输出进行算术求和
    for i in range(m):
        sum_mat[i,0] = np.sum(Gauss[i,:],axis = 1) ##sum_mat的第0列为每个测试样本Gauss数值之和
    ## 对所有模式层神经元进行加权求和
    for i in range(m):             
        for j in range(n):
            total = 0.0
            for s in range(l):
                total += Gauss[i,s] * trY[s,j]
            sum_mat[i,j+1] = total           ##sum_mat的后面的列为每个测试样本Gauss加权之和            
    return sum_mat

def output_layer(sum_mat):
    m,n = np.shape(sum_mat)
    output_mat = np.mat(np.zeros((m,n-1)))
    for i in range(n-1):
        output_mat[:,i] = sum_mat[:,i+1] / sum_mat[:,0]
    return output_mat


# In[125]:


Euclidean_D = distance_mat(X_train,X_test)


# In[129]:


Gauss1 = Gauss(Euclidean_D,0.1)


# In[130]:


Gauss1


# In[131]:


l=len(X_train)
m=len(y_test)
n = 1
sum_mat = np.mat(np.zeros((m,n+1)))
## 对所有模式层神经元输出进行算术求和
for i in range(m):
    sum_mat[i,0] = np.sum(Gauss1[i,:],axis = 1) ##sum_mat的第0列为每个测试样本Gauss数值之和
## 对所有模式层神经元进行加权求和
for i in range(m):             
    total = 0.0
    for s in range(l):
        total += Gauss1[i,s] * y_train[s]
    sum_mat[i,1] = total           ##sum_mat的后面的列为每个测试样本Gauss加权之和            


# In[132]:


sum_mat


# In[133]:


for i in range(0,5):
    print(sum_mat[i,1]/sum_mat[i,0])

