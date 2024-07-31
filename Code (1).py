#!/usr/bin/env python
# coding: utf-8

# #### Importing Libraries :

# In[90]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix
import seaborn as sns


# ## Reading Data into Memory

# In[88]:


data = pd.read_csv('Datas.csv')
data.head(10)


# #### Data before Pre-Processing

# In[92]:


plt.figure(figsize=(10, 6))
plt.plot(data['Time'], data['Fz'], color='blue', linestyle='-')
plt.xlabel('Time')
plt.ylabel('Fz')
plt.title('Original Data')
plt.grid(True)
plt.show()


# ## Preprocessing the Data

# In[73]:


data.drop(columns=['Time'], inplace=True)


# In[93]:


#Removing Noise

data = data[data['Fz'] > 0.4]
data.reset_index(inplace=True, drop=True)

#Preprocessed Data
print(data.head())


# #### Data After Pre-Processing

# In[75]:


plt.figure(figsize=(10, 6))
plt.plot(data.index, data['Fz'], color='blue', linestyle='-')
plt.xlabel('Serial Number')
plt.ylabel('Fz')
plt.title('Filtered Fz Data')
plt.grid(True)
plt.show()


# ## IsolationForest Model Training

# In[76]:


from sklearn.ensemble import IsolationForest

isolation_forest = IsolationForest(contamination=0.05)  
isolation_forest.fit(data)


anomaly_predictions = isolation_forest.predict(data)


anomalies = data[anomaly_predictions == -1]


# #### All the Anamalies in the Data

# In[94]:


print("Anomalies:", anomalies)


# ### Possible Failure Points Detected

# In[77]:


plt.figure(figsize=(10, 6))
plt.plot(data.index, data['Fz'], color='blue', linestyle='-', label='Machine will work without fail in this Zone')
plt.plot(anomalies.index, anomalies['Fz'], color='red', label='Machine might fail in this Zone')
plt.xlabel('Serial Number')
plt.ylabel('Fz')
plt.title('Possible Failure Points')
plt.legend()
plt.grid(True)
plt.show()


# #### Anomaly Score Distribution

# In[96]:


plt.figure(figsize=(10, 6))
sns.histplot(anomaly_scores, kde=True, color='blue')
plt.xlabel('Anomaly Score')
plt.ylabel('Frequency')
plt.title('Anomaly Score Distribution')
plt.show()


# ## Predicting the Safe Zone

# In[78]:


#Enter the Force Value

new_data_point = [[3]] 
prediction = isolation_forest.predict(new_data_point)

if prediction == -1:
    print("Machine might get damaged")
else:
    print("Machine is in safe zone")


# ## Accuracy and Efficiency

# In[97]:


num_anomalies = len(anomalies)
total_samples = len(data)
percentage_anomalies = (num_anomalies / total_samples) * 100
print("Number of anomalies:", num_anomalies)
print("Percentage of anomalies:", percentage_anomalies)


# In[98]:


# Precision-Recall Curve

precision, recall, _ = precision_recall_curve(np.where(anomaly_predictions == -1, 1, 0), anomaly_scores)
plt.figure()
plt.step(recall, precision, color='b', alpha=0.2, where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall Curve')
plt.show()


# In[99]:


# Confusion Matrix

cm = confusion_matrix(np.where(anomaly_predictions == -1, 1, 0), np.where(anomaly_predictions == -1, 1, 0))
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


# In[ ]:


import numpy as np

class IsolationTreeEnsemble:
    def __init__(self, sample_size, n_trees=10):
        self.sample_size = sample_size
        self.n_trees = n_trees
        self.trees = []
        
    def fit(self, X):
        num_samples, num_features = X.shape
        max_height = np.ceil(np.log2(self.sample_size))
        
        for _ in range(self.n_trees):
            idx = np.random.choice(num_samples, size=self.sample_size, replace=False)
            X_subset = X[idx]
            tree = IsolationTree(max_height)
            tree.fit(X_subset)
            self.trees.append(tree)
            
    def anomaly_score(self, X):
        num_samples = X.shape[0]
        scores = np.zeros(num_samples)
        
        for tree in self.trees:
            scores += tree.anomaly_score(X)
            
        return scores / self.n_trees
    
class IsolationTree:
    def __init__(self, max_height):
        self.max_height = max_height
        self.root = None
        
    def fit(self, X, current_height=0):
        num_samples, num_features = X.shape
        
        if current_height >= self.max_height or num_samples <= 1:
            self.root = {
                'is_leaf': True,
                'size': num_samples
            }
            return
        
        feature_index = np.random.randint(num_features)
        split_value = np.random.uniform(X[:, feature_index].min(), X[:, feature_index].max())
        
        left_indices = X[:, feature_index] < split_value
        right_indices = ~left_indices
        
        self.root = {
            'is_leaf': False,
            'feature_index': feature_index,
            'split_value': split_value
        }
        
        self.left = IsolationTree(self.max_height)
        self.left.fit(X[left_indices], current_height + 1)
        
        self.right = IsolationTree(self.max_height)
        self.right.fit(X[right_indices], current_height + 1)
        
    def anomaly_score(self, X, current_height=0):
        if self.root['is_leaf']:
            if self.root['size'] == 1:
                return np.zeros(X.shape[0])
            else:
                return current_height + 1
            
        left_indices = X[:, self.root['feature_index']] < self.root['split_value']
        right_indices = ~left_indices
        
        scores = np.zeros(X.shape[0])
        scores[left_indices] = self.left.anomaly_score(X[left_indices], current_height + 1)
        scores[right_indices] = self.right.anomaly_score(X[right_indices], current_height + 1)
        
        return scores

# Using the IsolationTreeEnsemble
sample_size = 256  
n_trees = 100  

data = np.random.randn(1000, 2) 


ensemble = IsolationTreeEnsemble(sample_size, n_trees)
ensemble.fit(data)

anomaly_scores = ensemble.anomaly_score(data)
anomalies_indices = np.where(anomaly_scores >= np.percentile(anomaly_scores, 95))[0]
anomalies = data[anomalies_indices]

print("Anomalies:", anomalies)

