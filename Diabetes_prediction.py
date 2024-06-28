#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Python Libraries 
import pandas as pd #Data Processing and CSV file I/o
import numpy as np #for numeric operations
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

#spliting and scaling the data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Using GridSearchCV to find the best algorithm for this problem 
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

#metric
from sklearn.metrics import classification_report, accuracy_score


# In[3]:


diabetes_df = pd.read_csv('diabetes.csv')
diabetes_df.head(8)


# In[4]:


#lowering #lowercasing all the column names
diabetes_df.columns = diabetes_df.columns.str.lower()


# In[5]:


diabetes_df.columns


# In[6]:


#renaming the column name 
diabetes_df = diabetes_df.rename(columns={'diabetespedigreefunction': 'diabetes_pedigree_function',                                           'bloodpressure': 'blood_pressure', 'skinthickness': 'skin_thickness'})
diabetes_df.columns


# In[7]:


print(f"The Numbers of Rows and Columns in this data set are: {diabetes_df.shape[0]} rows and {diabetes_df.shape[1]} columns.")


# EXPLORATORY DATA ANALYSIS
# 

# In[8]:


#Checking for Data type of columns
diabetes_df.info()


# In[13]:


#statistics summary
diabetes_df.describe().T


# In[12]:


#Checking out the Correlation Matrix
sns.pairplot(diabetes_df, diag_kind="kde");


# In[14]:


#creating correlation matrix
corr = diabetes_df.corr()


# In[15]:


#plotting the correlation matrix
plt.figure(figsize=(16,12))
ax = sns.heatmap(corr, annot=True, square=True, fmt='.3f', linecolor='black')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_yticklabels(ax.get_yticklabels(), rotation=30)
plt.title('Correlation Heatmap')
plt.show();


# In[16]:


corr_matrix = diabetes_df.corr()
corr_matrix['outcome'].sort_values(ascending=False)


# In[17]:


#Checking for outliers
plt.figure(figsize=(16,15)) #(width,height)
plt.subplot(3,3,1) #(row, column, plot_number)
sns.boxplot(x='glucose', data=diabetes_df);
plt.subplot(3,3,2)
sns.boxplot(x='blood_pressure', data=diabetes_df);
plt.subplot(3,3,3)
sns.boxplot(x='skin_thickness', data=diabetes_df);
plt.subplot(3,3,4)
sns.boxplot(x='insulin', data=diabetes_df);
plt.subplot(3,3,5)
sns.boxplot(x='bmi', data=diabetes_df);
plt.subplot(3,3,6)
sns.boxplot(x='diabetes_pedigree_function', data=diabetes_df);
plt.subplot(3,3,7)
sns.boxplot(x='age', data=diabetes_df);


# In[18]:


#counting the missing values in numerical features
diabetes_df.isnull().sum()


# In[19]:


#distribution
diabetes_df.hist(figsize=(15,15));


# FEATURE SCALING
# 

# In[20]:


# segregating the target variable
X = diabetes_df.drop(columns='outcome')
y = diabetes_df['outcome']
#spliting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)


# In[21]:


# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# SELECTING AND TRAINING MODELS

# In[22]:


print(f"In X_train dataset there are: {X_train.shape[0]} rows and {X_train.shape[1]} columns.")
print(f"In X_test dataset there are: {X_test.shape[0]} rows and {X_test.shape[1]} columns.")
print(f"The shape of y_train is: {y_train.shape}")
print(f"The shape of y_test is: {y_test.shape}")


# In[23]:


X_train[:3]


# In[24]:


def best_model(X, y):
    """
    This function is for finding best model for this problem and tell the best parameter along with it.
    """
    models = {
        'logistic_regression': {
            'model': LogisticRegression(solver='lbfgs', multi_class='auto'),
            'parameters': {
                'C': [1,5,10]
               }
        },
        
        'decision_tree': {
            'model': DecisionTreeClassifier(splitter='best'),
            'parameters': {
                'criterion': ['gini', 'entropy'],
                'max_depth': [5,10]
            }
        },
        
        'random_forest': {
            'model': RandomForestClassifier(criterion='gini'),
            'parameters': {
                'n_estimators': [10,15,20,50,100,200]
            }
        },
        
        'svm': {
            'model': SVC(gamma='auto'),
            'parameters': {
                'C': [1,10,20],
                'kernel': ['rbf','linear']
            }
        }

    }
    
    scores = [] 
    cv_shuffle = ShuffleSplit(n_splits=5, test_size=0.20, random_state=0)
        
    for model_name, model_params in models.items():
        gs = GridSearchCV(model_params['model'], model_params['parameters'], cv = cv_shuffle, return_train_score=False)
        gs.fit(X, y)
        scores.append({
            'model': model_name,
            'best_parameters': gs.best_params_,
            'score': gs.best_score_
        })
        
    return pd.DataFrame(scores, columns=['model','best_parameters','score'])

best_model(X_train, y_train)


# In[25]:


# Using cross_val_score for gaining average accuracy
from sklearn.model_selection import cross_val_score
scores = cross_val_score(RandomForestClassifier(n_estimators=200, random_state=42), X_train, y_train, cv=10)
print(f'Average Accuracy : {round(sum(scores)/len(scores) * 100, 3)}%')


# In[26]:


# Creating Random Forest Model
classifier = RandomForestClassifier(n_estimators=20, random_state=42)
classifier.fit(X_train, y_train)
y_train_pred = classifier.predict(X_train)


# In[27]:


print(f"Accuracy on trainning set: {round(accuracy_score(y_train, y_train_pred), 4)*100}%")


# EVALUATING THE ENTIRE SYSTEM ON TEST DATA

# In[28]:


y_test_pred = classifier.predict(X_test)
print(f"Accuracy on trainning set: {round(accuracy_score(y_test, y_test_pred), 4)*100}%")


# In[29]:


#Classification Report
print(classification_report(y_train, y_train_pred))


# In[30]:


plt.scatter(y_test, y_test_pred);


# SAVE THE MODEL

# In[32]:


import pickle 
file = open('model.pkl', 'wb') # open a file, where you ant to store the data
pickle.dump(classifier, file) #dump information to that file


# In[33]:


# Creating a function for prediction
def predict_diabetes(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DPF, Age):
    preg = int(Pregnancies)
    glucose = int(Glucose)
    bp = int(BloodPressure)
    st = int(SkinThickness)
    insulin = int(Insulin)
    bmi = float(BMI)
    dpf = float(DPF)
    age = int(Age)

    x = [[preg, glucose, bp, st, insulin, bmi, dpf, age]]
    x = sc.transform(x)

    return classifier.predict(x)

# Prediction 1
# Input sequence: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DPF, Age
prediction = predict_diabetes(2, 138, 62, 35, 0, 33.6, 0.127, 47)[0]
if prediction:
    print('Oops! You have diabetes.')
else:
    print("Great! You don't have diabetes.")


# In[ ]:




