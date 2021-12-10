
# coding: utf-8

# In[20]:


import matplotlib.pyplot as plt
import numpy as np
from pyts.classification import LearningShapelets
from pyts.utils import windowed_view
from pyts.transformation import ShapeletTransform
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC


# In[2]:


from sklearn.model_selection import train_test_split


# In[3]:


import pandas as pd


# In[4]:


from sklearn.tree import DecisionTreeClassifier


# In[5]:


from sklearn.model_selection import train_test_split,GridSearchCV


# In[6]:


from pyts.datasets import load_gunpoint


# In[7]:


hangseng=pd.read_excel("hangseng.xlsx")


# In[21]:


hangseng.head()


# In[22]:


hangseng.columns=['Date','close','chg','pct_chg','Y']


# In[23]:


Y=hangseng['Y']


# In[24]:


x=list(hangseng['pct_chg'])


# In[25]:


m=len(x)


# In[26]:


X=[]
for i in range(30,len(x)):
    s=[]
    for j in range(i-30,i):
        s.append(x[j])
    X.append(s)


# In[27]:


len(X)


# In[28]:


Y=Y[30:]


# In[29]:


X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.1)


# In[82]:


clf = LearningShapelets(random_state=42, tol=0.01)
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))


# In[31]:


shapelet = ShapeletTransform(n_shapelets=8, window_sizes=[3], random_state=42)


# In[32]:


tree = DecisionTreeClassifier()
clf1 = make_pipeline(shapelet, tree)
clf1.fit(X_train, y_train)
clf1.score(X_test, y_test)


# In[32]:


shapelet = ShapeletTransform(n_shapelets=8, window_sizes=[3], random_state=42)


# In[33]:


model=shapelet.fit(X_train, y_train)


# In[34]:


model.shapelets_


# In[35]:


t=model.transform(X_train)


# In[37]:


from sklearn.tree import DecisionTreeClassifier


# In[38]:


t=model.transform(X)


# In[93]:


pdt=pd.DataFrame(t)


# In[95]:


pdt.to_csv("st.csv")


# In[41]:


p=hangseng['pct_chg']
p=p[30:]


# In[61]:


from sklearn.ensemble import RandomForestRegressor


# In[62]:


forest=RandomForestRegressor(random_state=100)


# In[45]:


X_train, X_test, y_train, y_test = train_test_split(t,Y,test_size=0.1)


# In[55]:


tree=DecisionTreeClassifier(max_depth=10,random_state=0)
tree.fit(X_train,y_train)


# In[57]:


tree.score(X_test,y_test)


# In[59]:


X_train, X_test, y_train, y_test = train_test_split(t,p,test_size=0.1)


# In[63]:


forest.fit(X_train,y_train)


# In[64]:


y_pred=forest.predict(X_test)


# In[76]:


y_test


# In[75]:


y_pred


# In[71]:


y_test=np.array(y_test)


# In[74]:


s=0
ae=0
for i in range (0,len(y_pred)):
    s+=(y_pred[i]-y_test[i])**2
    ae+=abs(y_pred[i]-y_test[i])
MSE=s/len(y_pred)
MAE=ae/len(y_pred)
print(MSE,MAE)


# In[77]:


c=hangseng['close']


# In[78]:


c=c[31:]


# In[79]:


x=X[:-1]


# In[81]:


x=model.transform(x)


# In[82]:


X_train, X_test, y_train, y_test = train_test_split(x,c,test_size=0.1)


# In[87]:


forest=RandomForestRegressor(n_estimators=100,random_state=100)


# In[88]:


forest.fit(X_train,y_train)
y_pred=forest.predict(X_test)


# In[89]:


y_test=np.array(y_test)


# In[91]:


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
print(MSE,MAE,MAPE)


# Code without packages

# In[17]:


def generate_candidates(data, max_len=5, min_len=2):
    candidates, m = [], max_len
    while m >= min_len:
        for i in range(len(data)):
            time_serie = data[i]
            for k in range(len(time_serie)-m+1): 
                candidates.append((time_serie[k:k+m]))
        m -= 1
    return candidates


# In[18]:


candidates=generate_candidates(X)


# In[20]:


type(candidates)


# In[27]:


def check_candidate(data, shapelet):
    histogram = {} 
    for i in range(0,len(data)):
        # TODO: entropy pre-pruning in each iteration
        time_serie = data[i]
        d, idx = subsequence_dist(time_serie, shapelet)
        if d is not None:
            histogram[d] = [(time_serie, label)] if d not in histogram else histogram[d].append((time_serie, label))
    return find_best_split_point(histogram)


# In[24]:


def subsequence_dist(time_serie, sub_serie):
    if len(sub_serie) < len(time_serie):
        min_dist, min_idx = float("inf"), 0
        for i in range(len(time_serie)-len(sub_serie)+1):
            dist = manhattan_distance(sub_serie, time_serie[i:i+len(sub_serie)], min_dist)
            if dist is not None and dist < min_dist: min_dist, min_idx = dist, i
        return min_dist, min_idx
    else:
        return None, None


# In[26]:


len(candidates)


# In[29]:


def calculate_dict_entropy(data):
    counts = {}
    for entry in data:
        if entry[1] in counts: counts[entry[1]] += 1
        else: counts[entry[1]] = 1
    return calculate_entropy(np.divide(list(counts.values()), float(sum(list(counts.values())))))


# In[ ]:


def find_best_split_point(histogram):
    histogram_values = list(itertools.chain.from_iterable(list(histogram.values())))
    prior_entropy = calculate_dict_entropy(histogram_values)
    best_distance, max_ig = 0, 0
    best_left, best_right = None, None
    for distance in histogram:
        data_left = []
        data_right = []
        for distance2 in histogram:
            if distance2 <= distance: data_left.extend(histogram[distance2])
            else: data_right.extend(histogram[distance2])
        ig = prior_entropy - (float(len(data_left))/float(len(histogram_values))*calculate_dict_entropy(data_left) +              float(len(data_right))/float(len(histogram_values)) * calculate_dict_entropy(data_right))
        if ig > max_ig: best_distance, max_ig, best_left, best_right = distance, ig, data_left, data_right
    return max_ig, best_distance, best_left, best_right


# In[30]:


def manhattan_distance(a, b, min_dist=float('inf')):
    dist = 0
    for x, y in zip(a, b):
        dist += np.abs(float(x)-float(y))
        if dist >= min_dist: return None
    return dist

def calculate_entropy(probabilities):
    return sum([-prob * np.log(prob)/np.log(2) if prob != 0 else 0 for prob in probabilities])


# In[ ]:


def find_shapelets_bf(data, max_len=100, min_len=1, plot=True, verbose=True):
    candidates = generate_candidates(data, max_len, min_len)
    bsf_gain, bsf_shapelet = 0, None
    if verbose: candidates_length = len(candidates)
    for idx, candidate in enumerate(candidates):
        gain, dist, data_left, data_right = check_candidate(data, candidate[0])
        if verbose: print(idx, '/', candidates_length, ":", gain, dist)
        if gain > bsf_gain:
            bsf_gain, bsf_shapelet = gain, candidate[0]
            if verbose:
                print('Found new best shapelet with gain & dist:', bsf_gain, dist, [x[1] for x in data_left],                                                                                    [x[1] for x in data_right])
            if plot:
                plt.plot(bsf_shapelet)
                plt.show()
            plt.show()
    return bsf_shapelet

