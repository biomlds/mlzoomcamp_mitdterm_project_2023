#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pycaret
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
# from sklearnex import patch_sklearn
# patch_sklearn()
from pycaret.classification import *


# #### Load data

# In[2]:


train = pd.read_csv('../data/train.csv')
train.shape


# #### Check data stats, missing values

# In[3]:


train.describe().T


# In[4]:


train.info()


# #### Pycaret will generate polynomial features. Let's add several extra features

# In[5]:


train['bmi'] = 10000*train['weight(kg)']/train['height(cm)']/train['height(cm)']
train['liver_enz'] = train['AST'] + train['ALT'] + train['Gtp']
train['totalDL'] = train['HDL'] + train['LDL']


# #### Check for categorical features

# In[6]:


for ft in train.columns:
    if len(train[ft].value_counts())<10:
        print(45*'=')
        print(ft,)
        print(train[ft].value_counts())
print(45*'=')
        


# `dental caries` is the binary feature. The rest can be treated as numerical ones.

# In[7]:


ignore_features = ['id']
categorical_features = ['dental caries']
TARGET = 'smoking'

numeric_features = train.columns.values.tolist()
for fetaure in categorical_features+ignore_features+[TARGET]:
    numeric_features.remove(fetaure)



# #### Setup pycaret experiment

# In[8]:


s = setup(train,
            # numeric_features=numeric_features,
            # categorical_features=categorical_features,
            # ordinal_features=ordinal_features,
            # bin_numeric_features = bin_numeric_features,
            ignore_features=ignore_features,
            # remove_multicollinearity=True,
            # low_variance_threshold = 0.1,
            train_size=0.8, 
            fold = 5,
            normalize=True, 
            normalize_method='robust',
            # transformation=True,  
            polynomial_features=True, 
            polynomial_degree=2,
            feature_selection_estimator=True,
            fix_imbalance=True,
            target = TARGET, 
            session_id = 111, 
            )


# #### Let's run a set of models and pick the best one

# In[9]:


best = compare_models()


# `Catboost` is the best model. Proceed with it.

# #### Tune the best model

# In[10]:


tuned_best = tune_model(best, search_library = 'scikit-learn')


# In[11]:


print(tuned_best)


# In[12]:


plot_model(tuned_best)


# #### Top 10 important features

# In[13]:


plot_model(tuned_best, plot = 'feature')


# #### Here is the confusion matrix
# The model is good enough.

# In[14]:


plot_model(tuned_best, plot = 'confusion_matrix', plot_kwargs = {'percent' : True})


# #### Deploy the model as API in a docker container

# In[16]:


# create api
create_api(tuned_best, 'smoking_clf_api')



# In[17]:


# create docker
create_docker('smoking_clf_api')

