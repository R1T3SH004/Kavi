
# coding: utf-8

# In[54]:



import pandas as pd
import numpy as np


# In[55]:


adult_df = pd.read_csv('adult_data.csv',
                       header = None, delimiter=' *, *',engine='python')

adult_df.head()


# In[56]:


pd.set_option('display.max_columns',None)
#pd.set_option('display.max_rows',None)


# In[57]:


adult_df.shape


# In[58]:


adult_df.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
                    'marital_status', 'occupation', 'relationship',
                    'race', 'sex', 'capital_gain', 'capital_loss',
                    'hours_per_week', 'native_country', 'income']

adult_df.head()


# **Pre processing the data**

# In[59]:


adult_df.isnull().sum()


# In[60]:


adult_df=adult_df.replace(['?'], np.nan)


# In[61]:


adult_df.isnull().sum()


# In[62]:


#create a copy of the dataframe
adult_df_rev = pd.DataFrame.copy(adult_df)

#adult_df_rev.describe(include= 'all')


# In[63]:


#replace the missing values with values in the top row of each column
for value in ['workclass', 'occupation',
              'native_country']:
    adult_df_rev[value].fillna(adult_df_rev[value].mode()[0],inplace=True)


# In[64]:


adult_df_rev.workclass.mode()


# In[65]:


"""
for x in adult_df_rev.columns[:]:
    if adult_df_rev[x].dtype=='object':
        adult_df_rev[x].fillna(adult_df_rev[x].mode()[0],inplace=True)
    elif adult_df_rev[x].dtype=='int64':
        adult_df_rev[x].fillna(adult_df_rev[x].mean(),inplace=True)
"""


# In[66]:


adult_df_rev.isnull().sum()
#adult_df_rev.head()


# In[67]:


adult_df_rev.workclass.value_counts()


# In[68]:


colname = ['workclass', 'education',
          'marital_status', 'occupation',
          'relationship','race', 'sex',
          'native_country', 'income']
colname


# In[69]:


# For preprocessing the data
from sklearn import preprocessing

le={}

le=preprocessing.LabelEncoder()

for x in colname:
     adult_df_rev[x]=le.fit_transform(adult_df_rev[x])


# In[70]:


adult_df_rev.head()

#0--> <=50K
#1--> >50K


# In[71]:


adult_df_rev.dtypes


# In[72]:


X = adult_df_rev.values[:,:-1]
Y = adult_df_rev.values[:,-1]



# In[73]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X)

X = scaler.transform(X)
print(X)


# In[74]:


#np.set_printoptions(threshold=np.inf)


# In[75]:


Y=Y.astype(int)


# **Running a basic model**

# In[76]:


from sklearn.model_selection import train_test_split

#Split the data into test and train
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3,
                                                    random_state=10)  


# In[77]:


from sklearn.linear_model import LogisticRegression
#create a model
classifier=LogisticRegression()
#fitting training data to the model
classifier.fit(X_train,Y_train)

Y_pred=classifier.predict(X_test)
print(list(zip(Y_test,Y_pred)))

print(classifier.coef_)
print(classifier.intercept_)


# In[78]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)

print("Classification report: ")

print(classification_report(Y_test,Y_pred))

acc=accuracy_score(Y_test, Y_pred)
print("Accuracy of the model: ",acc)



# **Adjusting the threshold**

# In[28]:


# store the predicted probabilities
y_pred_prob = classifier.predict_proba(X_test)
print(y_pred_prob)


# In[29]:


y_pred_class=[]
for value in y_pred_prob[:,1]:
    if value > 0.46:
        y_pred_class.append(1)
    else:
        y_pred_class.append(0)
print(y_pred_class)


# In[30]:


from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cfm=confusion_matrix(Y_test,y_pred_class)
print(cfm)
acc=accuracy_score(Y_test, y_pred_class)
print("Accuracy of the model: ",acc)
print(classification_report(Y_test, y_pred_class))


# In[31]:



for a in np.arange(0,1,0.01):
    predict_mine = np.where(y_pred_prob[:,1] > a, 1, 0)
    cfm=confusion_matrix(Y_test, predict_mine)
    total_err=cfm[0,1]+cfm[1,0]
    print("Errors at threshold ", a, ":",total_err, " , type 2 error :", 
          cfm[1,0]," , type 1 error:", cfm[0,1])



# **Running model using cross validation**

# In[32]:


#Using cross validation

classifier=(LogisticRegression())

#performing kfold_cross_validation
from sklearn.model_selection import KFold
kfold_cv=KFold(n_splits=10)
print(kfold_cv)

from sklearn.model_selection import cross_val_score
#running the model using scoring metric as accuracy
kfold_cv_result=cross_val_score(estimator=classifier,X=X_train,
                                                 y=Y_train, cv=kfold_cv)
print(kfold_cv_result)
#finding the mean
print(kfold_cv_result.mean())


for train_value, test_value in kfold_cv.split(X_train):
    classifier.fit(X_train[train_value], Y_train[train_value]).predict(X_train[test_value])

    
Y_pred=classifier.predict(X_test)
#print(list(zip(Y_test,Y_pred)))


# In[33]:


from sklearn.metrics import confusion_matrix, accuracy_score,classification_report

confusion_matrix=confusion_matrix(Y_test,Y_pred)
print(confusion_matrix)
print()


print("Classification report: ")

print(classification_report(Y_test,Y_pred))

accuracy_score=accuracy_score(Y_test, Y_pred)
print("Accuracy of the model: ",accuracy_score)


# **Feature selection using Recursive Feature Elimination**

# In[34]:


colname=adult_df_rev.columns[:]


# In[35]:


from sklearn.feature_selection import RFE
rfe = RFE(classifier, 8)
model_rfe = rfe.fit(X_train, Y_train)
print("Num Features: ",model_rfe.n_features_)
print("Selected Features: ") 
print(list(zip(colname, model_rfe.support_)))
print("Feature Ranking: ", model_rfe.ranking_) 


# In[36]:


Y_pred=model_rfe.predict(X_test)
#print(list(zip(Y_test,Y_pred)))


# In[37]:


from sklearn.metrics import confusion_matrix, accuracy_score,classification_report

confusion_matrix=confusion_matrix(Y_test,Y_pred)
print(confusion_matrix)
print()


print("Classification report: ")

print(classification_report(Y_test,Y_pred))

accuracy_score=accuracy_score(Y_test, Y_pred)
print("Accuracy of the model: ",accuracy_score)


# In[38]:


"""new_data=adult_df_rev[['age','workclass','occupation','race','sex','income']]
new_data.head()
new_X=new_data.values[:,:-1]
new_Y=new_data.values[:,-1]
print(new_X)
print(new_Y)
"""


# **Feature selection using Univariate Selection**

# In[39]:


X = adult_df_rev.values[:,:-1]
Y = adult_df_rev.values[:,-1]


# In[40]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


test = SelectKBest(score_func=chi2, k=11)
fit1 = test.fit(X, Y)

print(fit1.scores_)
print(list(zip(colname,fit1.get_support())))
X = fit1.transform(X)

print(X)


# In[41]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X)

X = scaler.transform(X)


# In[42]:


from sklearn.model_selection import train_test_split

#Split the data into test and train
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3,random_state=10)  


# In[43]:


from sklearn.linear_model import LogisticRegression
#create a model
classifier=(LogisticRegression())
#fitting training data to the model
classifier.fit(X_train,Y_train)
Y_pred=classifier.predict(X_test)
#print(list(zip(Y_test,Y_pred)))


# In[44]:


from sklearn.metrics import confusion_matrix, accuracy_score,classification_report

confusion_matrix=confusion_matrix(Y_test,Y_pred)
print(confusion_matrix)
print()


print("Classification report: ")

print(classification_report(Y_test,Y_pred))

accuracy_score=accuracy_score(Y_test, Y_pred)
print("Accuracy of the model: ",accuracy_score)


# ** Variance Threshold**

# In[45]:


X = adult_df_rev.values[:,:-1]
Y = adult_df_rev.values[:,-1]


# In[46]:


from sklearn.feature_selection import VarianceThreshold


# In[53]:


#scaling required
vt = VarianceThreshold()
fit1 = vt.fit(X, Y)
print(fit1.variances_)

features = fit1.transform(X)
print(features)
print(features.shape[1])
print(list(zip(colname,fit1.get_support())))

