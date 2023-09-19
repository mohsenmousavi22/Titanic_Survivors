#!/usr/bin/env python
# coding: utf-8

# # Overview
# 
# In this project, my primary objective is to predict the survival of passengers aboard the RMS Titanic, a monumental task that requires keen analysis and the application of robust machine learning algorithms.
# 
# Drawing insights from the historical data available, I aim to create a model that can accurately determine whether a given passenger would have survived the tragic sinking of the ship.
# 
# 
# To achieve this, i have employed various classification models, each with its unique strengths:
# 
# 
# 1. **Logistic Regression**: This model is a fundamental algorithm for binary classification tasks. Given its simplicity and interpretability, it serves as a robust baseline against which we can gauge the performance of more complex models.
# 
# 
# 2. **Decision Tree Classification**: A flowchart-like tree structure where an internal node represents a feature(or attribute), the branch represents a decision rule, and each leaf node represents an outcome. This model provides a clear visual representation of the decision-making process, making it easier to interpret and understand.
# 
# 
# 
# 3. **Random Forest Classification**: A powerful ensemble method that uses multiple decision trees and aggregates their results. The Random Forest Classifier is renowned for its ability to handle large datasets with higher dimensionality and can output the importance of different features, thereby providing insights into the critical factors determining survival.
# 
# 
# 
# Throughout the project, I emphasize data preprocessing, model training, optimization, and validation to ensure the highest accuracy and reliability of  predictive models. The findings and insights derived from these models can provide a deeper understanding of the factors that influenced survival during the Titanic disaster and demonstrate the power of machine learning in predicting outcomes based on historical data.

# # Import libraries 

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


# # Data Preprocessing

# In[2]:


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# In[3]:


train.info()


# In[4]:


test.info()


# In[5]:


train.isnull().sum()


# In[6]:


test.isnull().sum()


# In[7]:


train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis = 1, inplace = True)
test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis = 1, inplace = True)


# In[8]:


train['Age'].fillna(train['Age'].mean(), inplace=True)


# In[9]:


train['Embarked'].fillna(train['Embarked'].mode().iloc[0], inplace=True)


# In[10]:


for col in ['Age', 'Fare']:
    test[col].fillna(train[col].mean(), inplace=True)


# In[11]:


print('missing value in train set is:', train.isnull().sum().sum())
print('missing value in test set is:', test.isnull().sum().sum())


# In[12]:


train_encoded = pd.get_dummies(train, columns=['Sex', 'Embarked'], prefix={'Sex': '', 'Embarked': ''}, prefix_sep='')


# In[13]:


test_encoded = pd.get_dummies(test, columns=['Sex', 'Embarked'], prefix={'Sex': '', 'Embarked': ''}, prefix_sep='')


# In[14]:


X_train = train_encoded.drop('Survived', axis=1)
y_train = train_encoded['Survived']
X_test = test_encoded


# In[15]:


param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 
    'penalty': ['l1', 'l2'], 
    'solver': ['liblinear'] 
}

logistic = LogisticRegression()
grid_search = GridSearchCV(logistic, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("Best Hyperparameters:", grid_search.best_params_)
print("Best Cross-Validation Accuracy:", grid_search.best_score_)

y_pred = grid_search.predict(X_test)
print("Sample Predictions:", y_pred[:10])


# In[16]:


param_grid = {
    'criterion': ['gini', 'entropy'],  
    'splitter': ['best', 'random'],
    'max_depth': [None, 5, 10, 15, 20],
    'min_samples_split': [2, 5, 10], 
    'min_samples_leaf': [1, 2, 5, 10]
}

decision_tree = DecisionTreeClassifier()
grid_search = GridSearchCV(decision_tree, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("Best Hyperparameters:", grid_search.best_params_)
print("Best Cross-Validation Accuracy:", grid_search.best_score_)

y_pred = grid_search.predict(X_test)
print("Sample Predictions:", y_pred[:10])


# In[17]:


param_grid = {
    'n_estimators': [50, 100, 200],  
    'criterion': ['gini', 'entropy'],  
    'max_depth': [None, 5, 10, 15, 20],  
    'min_samples_split': [2, 5, 10],  
    'min_samples_leaf': [1, 2, 5, 10],  
    'bootstrap': [True, False]
}

random_forest = RandomForestClassifier()
grid_search = GridSearchCV(random_forest, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("Best Hyperparameters:", grid_search.best_params_)
print("Best Cross-Validation Accuracy:", grid_search.best_score_)

y_pred = grid_search.predict(X_test)
print("Sample Predictions:", y_pred[:10])


# In[18]:


best_params = grid_search.best_params_
final_rf_model = RandomForestClassifier(**best_params)
final_rf_model.fit(X_train, y_train)


# In[19]:


predicted_targets = final_rf_model.predict(X_test)


# In[20]:


test = pd.read_csv('test.csv')
passenger_ids = test['PassengerId'].values


# In[21]:


result_df = pd.DataFrame({
    'PassengerId': passenger_ids,
    'Survived': predicted_targets
})


# In[22]:


print(result_df.head(10))


# In[23]:


result_df.to_csv('predictions.csv', index=False)

