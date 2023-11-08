#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pycaret
import pandas as pd
import matplotlib.pyplot as plt
# from sklearnex import patch_sklearn
# patch_sklearn()
from pycaret.classification import *


# In[3]:


pycaret.__version__


# #### Load data

# In[4]:


train = pd.read_csv('../data/train.csv')
train.shape


# #### Check data stats, missing values

# In[5]:


train.describe().T


# In[6]:


train.info()


# #### Pycaret will generate polynomial features. Let's add several extra features

# In[7]:


train['bmi'] = 10000*train['weight(kg)']/train['height(cm)']/train['height(cm)']
train['liver_enz'] = train['AST'] + train['ALT'] + train['Gtp']
train['totalDL'] = train['HDL'] + train['LDL']


# #### Check for categorical features

# In[8]:


for ft in train.columns:
    if len(train[ft].value_counts())<10:
        print(45*'=')
        print(ft,)
        print(train[ft].value_counts())
print(45*'=')
        


# `dental caries` is the binary feature. The rest can be treated as numerical ones.

# In[9]:


ignore_features = ['id']
categorical_features = ['dental caries']
TARGET = 'smoking'

numeric_features = train.columns.values.tolist()
for fetaure in categorical_features+ignore_features+[TARGET]:
    numeric_features.remove(fetaure)



# #### Setup pycaret experiment

# In[10]:


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

# In[11]:


models()


# In[12]:


best = compare_models(include = ['catboost', 'lightgbm', 'lr', 'dummy'])
best = compare_models()


# `Catboost` is the best model. Proceed with it.

# #### Tune the best model

# In[14]:


tuned_best = tune_model(best)


# In[16]:


print(tuned_best)


# In[15]:


plot_model(tuned_best)


# #### Top 10 important features

# In[16]:


plot_model(tuned_best, plot = 'feature')


# #### Here is the confusion matrix
# The model is good enough.

# In[17]:


plot_model(tuned_best, plot = 'confusion_matrix', plot_kwargs = {'percent' : True})


# #### Deploy the model as API in a docker container

# In[33]:


import joblib
joblib.dump(tuned_best, 'tuned_best.pkl')
joblib.dump(tuned_best, 'best.pkl')


# In[19]:


#!pip install pycaret[mlops]
# create api
create_api(best, 'smoking_clf_api')



# In[20]:


# create docker
create_docker('smoking_clf_api')

