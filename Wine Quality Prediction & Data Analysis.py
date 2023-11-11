#!/usr/bin/env python
# coding: utf-8

# # "Wine Quality Prediction & Data Analysis"

# # Dataset

# The overall goal is : 
# #### "Wine Quality Prediction & Data Analysis"
# 
# - The datasets are related to red and white variants of the Portuguese "Vinho Verde" wine. 
# - For more details, the reference [Cortez et al., 2009]. 
# - Due to privacy and logistic issues, only physicochemical (inputs) and sensory (the output) variables are available (e.g. there is no data about grape types, wine brand, wine selling price, etc.).
# - These datasets can be viewed as classification or regression tasks. 
# - The classes are ordered and not balanced (e.g. there are much more normal wines than excellent or poor ones). 
# - Outlier detection algorithms could be used to detect the few excellent or poor wines.

# Prerequisites : Jupyter Notebook, Pandas, Numpy and Seaborn.

# ## Here are all the imports that we will require to build this model.

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv("D:\DATA SCIENCE\Wine Dataset\WINE_QUALITY.csv")


# In[3]:


df.head()


# In[4]:


df.describe()


# In[5]:


df.info()


# In[6]:


df.isnull().sum()


# In[7]:


df["fixed acidity"].value_counts()


# In[8]:


mean = df["fixed acidity"].mean()
df["fixed acidity"].fillna(mean,inplace=True)
df["fixed acidity"].isnull().sum()


# In[9]:


mean2 = df["volatile acidity"].mean()
df["volatile acidity"].fillna(mean,inplace=True)
df["volatile acidity"].isnull().sum()


# In[10]:


mean3 = df["citric acid"].mean()
df["citric acid"].fillna(mean,inplace=True)
df["citric acid"].isnull().sum()


# In[11]:


mean4 = df["residual sugar"].mean()
df["residual sugar"].fillna(mean,inplace=True)
df["residual sugar"].isnull().sum()


# In[12]:


mean4 = df["chlorides"].mean()
df["chlorides"].fillna(mean,inplace=True)
df["chlorides"].isnull().sum()


# In[13]:


mean5 = df["pH"].mean()
df["pH"].fillna(mean,inplace=True)
df["pH"].isnull().sum()


# In[14]:


mean6 = df["sulphates"].mean()
df["sulphates"].fillna(mean,inplace=True)
df["sulphates"].isnull().sum()


# In[15]:


df.isnull().sum()


# ## LET's VISUALISE THE DATA

# In[16]:


plt.figure(figsize=(10,7))
plt.scatter(x="alcohol",y="fixed acidity",data =df,marker= 'o',c="r")
plt.xlabel("alcohol",fontsize=15)
plt.ylabel("fixed_acidity",fontsize=15)
plt.show()


# In[17]:


sns.lmplot(x="alcohol",y="fixed acidity",data=df)
plt.plot()


# In[18]:


plt.figure(figsize=(10,7))
plt.scatter(x="volatile acidity",y="alcohol",data =df,marker= 'o',c="m")
plt.xlabel("volatile_acidity",fontsize=15)
plt.ylabel("alcohol",fontsize=15)
plt.show()


# In[19]:


sns.set(style="darkgrid")
sns.countplot(df["quality"],hue="type",data=df)
plt.show()


# In[20]:


sns.set()
sns.distplot(df["quality"],bins=10)
plt.show()


# In[21]:


plt.figure(figsize=(10,7))
sns.regplot(x="citric acid",y="chlorides",data =df,marker= 'o',color="r")
plt.show()


# In[22]:


plt.figure(figsize=(10,7))
sns.regplot(x="fixed acidity",y="volatile acidity",data =df,marker= 'o',color="r")
plt.show()


# In[23]:


sns.set()
sns.pairplot(df)
plt.show()


# In[24]:


sns.set()
plt.figure(figsize=(20,10))
sns.boxplot(data=df,palette="Set3")
plt.show()


# ## REMOVING OUTLIERS
# We can see that there are Some outliers.So now let's remove those Outliers

# In[25]:


lower_limit = df["free sulfur dioxide"].mean() - 3*df["free sulfur dioxide"].std()
upper_limit = df["free sulfur dioxide"].mean() + 3*df["free sulfur dioxide"].std()
print(lower_limit,upper_limit)


# In[26]:


df2 = df[(df["free sulfur dioxide"] > lower_limit) & (df["free sulfur dioxide"] < upper_limit)]
df.shape[0] - df2.shape[0]


# In[27]:


lower_limit = df2['total sulfur dioxide'].mean() - 3*df2['total sulfur dioxide'].std()
upper_limit = df2['total sulfur dioxide'].mean() + 3*df2['total sulfur dioxide'].std()
print(lower_limit,upper_limit)


# In[28]:


df3 = df2[(df2['total sulfur dioxide'] > lower_limit) & (df2['total sulfur dioxide'] < upper_limit)]
df3.head()


# In[29]:


df2.shape[0] - df3.shape[0]


# In[30]:


lower_limit = df3['residual sugar'].mean() - 3*df3['residual sugar'].std()
upper_limit = df3['residual sugar'].mean() + 3*df3['residual sugar'].std()
print(lower_limit,upper_limit)


# In[31]:


df4 = df3[(df3['residual sugar'] > lower_limit) & (df3['residual sugar'] < upper_limit)]
df4.head()


# In[32]:


df3.shape[0] - df4.shape[0]


# In[33]:


df4.isnull().sum()


# In[34]:


dummies = pd.get_dummies(df4["type"],drop_first=True)


# In[35]:


df4 = pd.concat([df4,dummies],axis=1)
df4.drop("type",axis=1,inplace=True)
df4.head()


# In[36]:


df4.quality.value_counts()


# In[37]:


df4.head()


# ## Now lets Change the Categorical 'String' Variables into Numerical Variables

# In[38]:


quaity_mapping = { 3 : "Low",4 : "Low",5: "Medium",6 : "Medium",7: "Medium",8 : "High",9 : "High"}
df4["quality"] =  df4["quality"].map(quaity_mapping)


# In[39]:


df4.quality.value_counts()


# In[40]:


df4.head()


# In[41]:


mapping_quality = {"Low" : 0,"Medium": 1,"High" : 2}
df4["quality"] =  df4["quality"].map(mapping_quality)
df4.head()


# ## Selecting the best Features for our Model

# In[42]:


x = df4.drop("quality",axis=True)
y = df4["quality"]


# In[43]:


from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(x,y)


# In[44]:


print(model.feature_importances_)


# In[45]:


feat_importances = pd.Series(model.feature_importances_,index =x.columns)
feat_importances.nlargest(9).plot(kind="barh")
plt.show()


# ## Now selecting the best Model for our Dataset

# In[46]:


from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


# In[47]:


model_params  = {
    "svm" : {
        "model":SVC(gamma="auto"),
        "params":{
            'C' : [1,10,20],
            'kernel':["rbf"]
        }
    },
    
    "decision_tree":{
        "model": DecisionTreeClassifier(),
        "params":{
            'criterion':["entropy","gini"],
            "max_depth":[5,8,9]
        }
    },
    
    "random_forest":{
        "model": RandomForestClassifier(),
        "params":{
            "n_estimators":[1,5,10],
            "max_depth":[5,8,9]
        }
    },
    "naive_bayes":{
        "model": GaussianNB(),
        "params":{}
    },
    
    'logistic_regression' : {
        'model' : LogisticRegression(solver='liblinear',multi_class = 'auto'),
        'params': {
            "C" : [1,5,10]
        }
    }
    
}


# In[48]:


score=[]
for model_name,mp in model_params.items():
    clf = GridSearchCV(mp["model"],mp["params"],cv=8,return_train_score=False)
    clf.fit(x,y)
    score.append({
        "Model" : model_name,
        "Best_Score": clf.best_score_,
        "Best_Params": clf.best_params_
    })


# In[49]:


df5 = pd.DataFrame(score,columns=["Model","Best_Score","Best_Params"])


# In[50]:


df5.head()


# # So we can see that, we are getting 93% accuracy for "SVM" & "Random Forest".

# In[51]:


from sklearn.model_selection import cross_val_score
clf_svm = SVC(kernel="rbf",C=1)
scores = cross_val_score(clf_svm,x,y,cv=8,scoring="accuracy")


# In[52]:


scores


# In[53]:


scores.mean()


# # So we are getting "93% Accuracy for predicting the Quality of Wine".

# In[54]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)


# In[55]:


clf_svm1 = SVC(kernel="rbf",C=1)
clf_svm1.fit(x_train,y_train)


# In[56]:


y_pred = clf_svm1.predict(x_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy


# ## Now Lets see the Real value and Predicted Value

# In[57]:


accuracy_dataframe = pd.DataFrame({"y_test": y_test, "y_pred": y_pred})
accuracy_dataframe.head()

