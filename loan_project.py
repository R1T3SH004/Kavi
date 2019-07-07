import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn import  preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mode
from sklearn.preprocessing import LabelEncoder


loan =pd.read_csv(r'XYZCorp_LendingData.txt',sep='\t',na_values = 'NaN',low_memory = False)

###df = pd.read_csv("property data.csv", na_values = missing_values)
pd.set_option('display.max_columns',None)
print(loan)
print(loan.isnull().sum())
loan.shape

loan.drop(['id', 'member_id', 'emp_title', 'desc', 'zip_code', 'title'], axis=1, inplace=True)

#Checking the datatypes
loan.info()
# Lets' transform the issue dates by year.
loan['issue_d'].head()
dt_series = pd.to_datetime(loan['issue_d'])
loan['year'] = dt_series.dt.year

plt.figure(figsize=(12,8))
sns.barplot('year', 'loan_amount', data=loan, palette='tab10')
plt.title('Issuance of Loans', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Average loan amount issued', fontsize=14)





#Lets check whether this coulnm has null/missing values or not
loan.emp_length.isnull().sum()
# It has 43061 missing values
loan.emp_length.value_counts()
# Let us make this variable simple integers. We will replace missing value entries with 0,
# and all non numeric entries.
loan.emp_length.fillna(value=0,inplace=True)
loan['emp_length'].replace(to_replace='^<', value=0.0, inplace=True, regex=True)
loan['emp_length'].replace(to_replace='[^0-9]+', value='', inplace=True, regex=True)
loan['emp_length'] = loan['emp_length'].astype(int)


loan.inq_last_6mths=np.where(loan['inq_last_6mths']==0,'A',
           np.where(loan['inq_last_6mths'].between(1,2), 'B',
           np.where(loan['inq_last_6mths'].between(3,6), 'C',
           np.where(loan['inq_last_6mths'].between(7,10), 'D',
           'E'
         ))))

loan.earliest_cr_line.isnull().sum()
#lets just retain the year for simplicity
loan['earliest_cr_line']= loan['earliest_cr_line'].apply(lambda s:int(s[-4:]))
loan['earliest_cr_line'].head()

loan.last_credit_pull_d.isnull().sum()
loan['last_credit_pull_d'].head()
loan.last_credit_pull_d=loan.last_credit_pull_d.fillna(loan['last_credit_pull_d'].value_counts().idxmax())
loan['last_credit_pull_d']= loan['last_credit_pull_d'].apply(lambda s:int(s[-4:]))
loan['last_credit_pull_d'].head()

loan['credit_age']= loan['last_credit_pull_d'] - loan['earliest_cr_line']

loan[['earliest_cr_line','last_credit_pull_d','credit_age']].head()

loan.drop(["last_credit_pull_d"], axis=1, inplace=True)

loan.drop(["mths_since_last_delinq","mths_since_last_record","next_pymnt_d",
           "initial_list_status","mths_since_last_major_derog",
"policy_code","annual_inc_joint","dti_joint","verification_status_joint","open_acc_6m","open_il_6m","open_il_12m",
"open_il_24m","mths_since_rcnt_il","total_bal_il","il_util","open_rv_12m","open_rv_24m","max_bal_bc","all_util","total_rev_hi_lim","inq_fi","total_cu_tl",
"pymnt_plan","inq_last_12m","sub_grade","total_pymnt_inv","out_prncp_inv","total_rec_prncp","collection_recovery_fee","total_rec_late_fee","funded_amnt_inv",
"application_type","acc_now_delinq","delinq_2yrs","pub_rec","revol_bal","addr_state"], axis=1, inplace=True)



#Filling missing values
for value in ['revol_util', 'tot_coll_amt','tot_cur_bal','collections_12_mths_ex_med']:
    loan[value].fillna(loan[value].mean(),inplace=True)

for value in ['last_pymnt_d','emp_length']:
    loan[value].fillna(loan[value].mode()[0],inplace=True)



loan.revol_util.isnull().sum()
# As this Colunm has 446 missing values no
loan.revol_util.fillna(value=0,inplace=True)

#converting the cateogorical data to numbers using the label encoder

obj_list = ['grade','verification_status','purpose','last_pymnt_d',
           'home_ownership','inq_last_6mths','earliest_cr_line','term']

loan.dtypes

le = LabelEncoder()
for i in obj_list:
    le.fit(list(loan[i].values))
    loan[i] = le.transform(list(loan[i]))

loan.columns
loan.info()



loan.issue_d.unique()
loan_copy=loan.copy()
loan_copy.info()



loan['issue_d']=pd.to_datetime(loan['issue_d'])

loan_train=loan.loc[loan['issue_d'] < loan['issue_d'].quantile(0.7)]
loan_train.issue_d.unique()

loan_test=loan.loc[loan['issue_d'] >= loan['issue_d'].quantile(0.7)]
loan_test.issue_d.unique()

Y_train=loan_train['default_ind']
Y_test=loan_test['default_ind']


loan_train.drop('issue_d',axis=1, inplace=True)
loan_test.drop('issue_d',axis=1, inplace=True)

#Now lets drop target variable from both test and train
loan_train.drop(['default_ind'],1, inplace=True)
loan_test.drop(['default_ind'],1, inplace=True)


loan_train.info()
loan_test.info()

X_train=loan_train
X_test=loan_test
X=X_train
y=Y_train
X.info()
X_train.info()


##Logistic
from sklearn.linear_model import LogisticRegression
#create a model
classifier=LogisticRegression()
#fitting training data to the model
classifier.fit(X_train,Y_train)

Y_pred=classifier.predict(X_test)
print(list(zip(Y_test,Y_pred)))

#print(classifier.coef_)
#print(classifier.intercept_)

from sklearn.metrics import confusion_matrix, accuracy_score,classification_report

confusion_matrix=confusion_matrix(Y_test,Y_pred)
print(confusion_matrix)

print(classification_report(Y_test,Y_pred))

accuracy_score=accuracy_score(Y_test, Y_pred)
print("Accuracy of the model: ",accuracy_score)


y_pred_prob = classifier.predict_proba(X_test)
print(y_pred_prob)

for a in np.arange(0,1,0.01):
    predict_mine = np.where(y_pred_prob[:,1] > a, 1, 0)
    cfm=confusion_matrix(Y_test, predict_mine)
    total_err=cfm[0,1]+cfm[1,0]
    print("Errors at threshold ", a, ":",total_err, " , type 2 error :", 
          cfm[1,0]," , type 1 error:", cfm[0,1])




y_pred_class=[]
for value in y_pred_prob[:,1]:
    if value > 0.80:
        y_pred_class.append(1)
    else:
        y_pred_class.append(0)
print(y_pred_class)


from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cfm=confusion_matrix(Y_test,y_pred_class)
print(cfm)
acc=accuracy_score(Y_test, y_pred_class)
print("Accuracy of the model: ",acc)
print(classification_report(Y_test, y_pred_class))


from sklearn.tree import DecisionTreeClassifier
model_Decision_tree =  DecisionTreeClassifier(criterion = "entropy", random_state = 10)
model_Decision_tree.fit(X_train,Y_train)
Y_pred = model_Decision_tree.predict(X_test)


confusion_matrix=confusion_matrix(Y_test,Y_pred)
print(confusion_matrix)


colname=loan_train.columns[:]
colname
from sklearn import tree
with open(r"XYZCorp_LendingData.txt", "w") as f:  
    f = tree.export_graphviz(model_Decision_tree, feature_names= colname[:-1],out_file=f)
#generate the file and upload the code in webgraphviz.com to plot the decision tree
    
# feature importance attribute of decision tree
    print(list(zip(colname,model_Decision_tree.feature_importances_)))



from sklearn.feature_selection import RFE 
rfe = RFE(classifier, 20)
model_rfe = rfe.fit(X_train, Y_train)
print("Num Features: ",model_rfe.n_features_)
print("Selected Features: ") 
print(list(zip(loan_train.columns, model_rfe.support_)))
print("Feature Ranking: ", model_rfe.ranking_) 

Y_pred=model_rfe.predict(X_test)


#predicting using the Random_Forest_Classifier
from sklearn.ensemble import RandomForestClassifier

model_RandomForest=RandomForestClassifier(500)

###
#fit the model on the data and predict the values
model_RandomForest.fit(X_train,Y_train)

Y_pred=model_RandomForest.predict(X_test)

import sklearn.metrics as metrics
# calculate the fpr and tpr for all thresholds of the classification
probs = model_RandomForest.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(Y_test, preds)
roc_auc = metrics.auc(fpr, tpr)

# method I: plt
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


### ROC Curve
from sklearn import metrics
y_pred_proba = model_Decision_tree.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(Y_test,  y_pred_proba)
auc = metrics.roc_auc_score(Y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()
