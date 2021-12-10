
# coding: utf-8

# In[7]:


import itertools
import numpy as np
import os
import matplotlib.image as mpimg       # reading images to numpy arrays

import matplotlib.pyplot as plt        # to plot any graph

from skimage import measure            # to find shape contour
import scipy.ndimage as ndi            # to determine shape centrality

from pylab import rcParams
rcParams['figure.figsize'] = (6, 6)      # setting default size of plots


# In[8]:


X = [[1, 2, 2, 1, 2, 3, 2],
     [0, 2, 0, 2, 0, 2, 3],
      [0, 1, 2, 2, 1, 2, 2]]
y = [0, 1, 0]


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


def check_candidate(data, shapelet):
    histogram = {} 
    for i in range(0,len(data)):
        # TODO: entropy pre-pruning in each iteration
        time_serie = data[i]
        d, idx = subsequence_dist(time_serie, shapelet)
        if d is not None:
            histogram[d] = [(time_serie, label)] if d not in histogram else histogram[d].append((time_serie, label))
    return find_best_split_point(histogram)


def calculate_dict_entropy(data):
    counts = {}
    for entry in data:
        if entry[1] in counts: counts[entry[1]] += 1
        else: counts[entry[1]] = 1
    return calculate_entropy(np.divide(list(counts.values()), float(sum(list(counts.values())))))


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


def manhattan_distance(a, b, min_dist=float('inf')):
    dist = 0
    for x, y in zip(a, b):
        dist += np.abs(float(x)-float(y))
        if dist >= min_dist: return None
    return dist

def calculate_entropy(probabilities):
    return sum([-prob * np.log(prob)/np.log(2) if prob != 0 else 0 for prob in probabilities])


def subsequence_dist(time_serie, sub_serie):
    if len(sub_serie) < len(time_serie):
        min_dist, min_idx = float("inf"), 0
        for i in range(len(time_serie)-len(sub_serie)+1):
            dist = manhattan_distance(sub_serie, time_serie[i:i+len(sub_serie)], min_dist)
            if dist is not None and dist < min_dist: min_dist, min_idx = dist, i
        return min_dist, min_idx
    else:
        return None, None


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


# In[9]:


def extract_shapelets(data, min_len=10, max_len=15, verbose=1):
    _classes = np.unique([x[1] for x in data])
    shapelet_dict = {}
    for _class in _classes:
        print('Extracting shapelets for', _class)
        transformed_data = []
        for entry in data:
            time_serie, label = entry[0], entry[1]
            if label == _class: transformed_data.append((time_serie, 1))
            else: transformed_data.append((time_serie, 0))
        shapelet_dict[_class] = find_shapelets_bf(transformed_data, max_len=max_len, min_len=min_len, plot=0, verbose=1)
    return shapelet_dict


# In[18]:


find_shapelets_bf(X)

