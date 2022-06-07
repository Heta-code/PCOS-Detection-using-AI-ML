#!/usr/bin/env python
# coding: utf-8

# In[9]:


#Project Name: Detection of PCOS
#Disease Name:PCOS(Polysystic Ovary Syndrome)
#Used Formulation of good Machine Learning model

#Importing required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Reading Dataset for PCOS
pcos = pd.read_csv('PCOS_data.csv')


print(pcos.head())#here head is used for printing first five values of table data         

for i in ['AMH(ng/mL)', 'II    beta-HCG(mIU/mL)']:
    pcos[i] = pd.to_numeric(pcos[i], errors='coerce')
pcos = pcos.drop(['Sl. No', 'Patient File No.', 'Unnamed: 44'], axis =1)#here we removed unnecesarry columns like S1. No, Pateint File No, Unnamed: 44

target = pcos.columns[:1].to_list()
features = pcos.columns[1:].to_list()
print("Total number of Features:", len(features))#calculate features

pcos.isnull().sum()#NULL value column

pcos = pcos.dropna()#Drop the column of null value
#to derive useful insights and prdeict the PCOS from data we're using here visualization.
continous=[
'PRL(ng/mL)', 'FSH/LH',
'II    beta-HCG(mIU/mL)', '  I   beta-HCG(mIU/mL)',
'BP _Diastolic (mmHg)', 'BP _Systolic (mmHg)',
'Avg. F size (L) (mm)', 'Avg. F size (R) (mm)',
'TSH (mIU/L)', 'RBS(mg/dl)',
'Vit D3 (ng/mL)','Cycle length(days)'
]

f, axes = plt.subplots(6, 2, figsize=(16,25))# Using seaborn and matplotlib with subplots
k = 0
for i in range(0,6):
    for j in range(0,2):
        sns.kdeplot(data=pcos, x=continous[k], hue="PCOS (Y/N)", ax = axes[i][j])#Using seaborn for ploting the graph
        k = k + 1
#Second Method to predict PCOS
#Now we use chi-square and SelectKBest to determine important features of data
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

num = 30#most important features

bestfeatures = SelectKBest(score_func=chi2, k=num)
fit = bestfeatures.fit(pcos[features], pcos[target])
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(pcos.columns)

featureScores = pd.concat([dfcolumns, dfscores], axis=1)
featureScores.columns = ['Feature','Score']
featureScores = featureScores.sort_values(by='Score', ascending = False)
featureScores = featureScores[featureScores.Feature != target[0]]
featureScores = featureScores.reset_index(drop = True)
print("TOP 30 Most Important Features:\t\n",featureScores[:num])


#featureScores[:num]

new_features = featureScores['Feature'].to_list()
new_features = new_features[:num]

#now we use ColumnTransformer and Pipeline to perform necessary transformation on the data.
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

numerical_transformer = Pipeline(steps=[('scaler', StandardScaler())])

preprocessor = ColumnTransformer(transformers=[('num', numerical_transformer, new_features)])


#now we train the model importing the sklearn library to perform classification & regression on the given data.
#importing classifiers
from sklearn import svm
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.naive_bayes import GaussianNB

from sklearn import metrics
from sklearn.model_selection import KFold, cross_val_score, train_test_split

train, test = train_test_split(pcos, test_size = 0.2, random_state = 0)

observations = pd.DataFrame()#converting obsevations into the pandas's Data

#Defining the Classifiers
classifiers = [
    'Linear SVM',
    'Radial SVM',
    'LogisticRegression',
    'RandomForestClassifier',
    'KNeighborsClassifier',
    'Gaussian Naive Bayes'
]

#Creating the Model
models = [
    svm.SVC(kernel='linear'),
    svm.SVC(kernel='rbf'),
    LogisticRegression(),
    RandomForestClassifier(n_estimators=200, random_state=0),
    KNeighborsClassifier(),
    GaussianNB()
]

j = 0
for i in models:
    model = i
    cv = KFold(n_splits=5, random_state=0, shuffle=True)
    pipe = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', model)])
    observations[classifiers[j]] = (cross_val_score(pipe, train[new_features], np.ravel(train[target]), scoring='accuracy', cv=cv))
    j = j+1

#now we calculate the scores of 5 folds with thier mean values
mean = pd.DataFrame(observations.mean(), index= classifiers)
observations = pd.concat([observations,mean.T])
observations.index=['Fold 1','Fold 2','Fold 3','Fold 4','Fold 5','Mean']
observations.T.sort_values(by=['Mean'], ascending = False)
print(observations.T.sort_values(by=['Mean'], ascending = False))

#observations.T.sort_values(by=['Mean'], ascending = False)

#performance of RandomForestClassifier with confusion_matrix
from sklearn.metrics import confusion_matrix

ran_model = RandomForestClassifier(n_estimators=200, random_state=0)
ran_pipe = Pipeline(steps=[('preprocessor', preprocessor), ('model', ran_model)])
ran_pipe.fit(train[new_features], np.ravel(train[target]))
pred = ran_pipe.predict(test[new_features])


plt.figure(dpi = 100)
plt.title("Confusion Matrix")
cf_matrix = confusion_matrix(np.ravel(test[target]), pred)
cf_hm = sns.heatmap(cf_matrix, annot=True, cmap = 'rocket_r')

#Result
import sklearn.metrics as metrics

fpr, tpr, threshold = metrics.roc_curve(test[target], pred)#RandForestClassifier with ROC Curve
roc_auc = metrics.auc(fpr, tpr)

plt.figure(dpi = 100)
plt.title('ROC curve for Random Forest Classifier')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])#define x-limit
plt.ylim([0, 1])#define y-limit
plt.ylabel('True Positive Rate(sensitivity)')
plt.xlabel('False Positive Rate(specificity)')
plt.savefig("output.jpg")
plt.show()#plotting ROC_Curve for RandomForestClassifier



# In[ ]:




