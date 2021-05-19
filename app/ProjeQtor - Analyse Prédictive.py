#!/usr/bin/env python
# coding: utf-8

# <center>
#     <img src="http://www.relations-publiques.pro/wp-content/uploads/2015/10/logo-Project.png" width="300" alt="projeqtor logo"  />
# </center>
# 
# # Analyse prédictive
# 
# 
# 

#  <h2>Contenu</h2>
# 
# <div class="alert alert-block alert-info" style="margin-top: 20px">
# <ol>
#     <li><a href="#">Setup librairies</a></li>
#     <li><a href="#">Importer les données</a></li>
#     <li><a href="#">Préparation des données </a></li>
#      <li><a href="#">Sélection des variables</a></li>
#      <li><a href="#">Modéles de  prédiction </a></li>
#     
# </ol>
# 
# </div>
#  
# <hr>

# ## 1. Setup librairies

# <p>Importation des librairies : </p>

# In[599]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
import sqlite3
from sqlite3 import OperationalError
import psycopg2
from pandas import DataFrame
import pandas_profiling as pp
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import learning_curve
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
import pandas_profiling as pp
from sklearn.tree import DecisionTreeClassifier

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler # i will use Min Max Scaler 
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import OrdinalEncoder


# ## 2.Importer les données 

# In[600]:

dfc= pd.read_csv("data.csv", na_values=missing_values , index_col=None)


def get_data():
    pd.set_option('display.max_columns', None)
    missing_values = ["n/a", "na", "--", " ",""] 
    dfc= pd.read_csv("data.csv", na_values=missing_values , index_col=None)
    return dfc


# ## 3. Préparation des données

# In[601]:


df1 = dfc[['idProject','Category','Premier Delay','projectDuration','CLient','Date','paymentDone','nb_phases']]
for i in range(len(df1)-1):
    if(df1['idProject'][i]==df1['idProject'][i+1] and df1['nb_phases'][i] < df1['nb_phases'][i+1]  ):
        df1.drop(labels=[i],axis=0,inplace=True)


# In[602]:


df1.groupby('idProject')
df1.head()
#df1['idProject'].value_counts()


# In[603]:


df=df1.drop(df1[['idProject'  , 'projectDuration','nb_phases' ]],axis=1)


# In[604]:


df.head()


# In[605]:


from sklearn.preprocessing import LabelEncoder

number = LabelEncoder()
df['Category'] = number.fit_transform(df1['Premier Delay'].astype('str'))
df['Premier Delay'] = number.fit_transform(df1['Premier Delay'].astype('str'))
df['CLient'] = number.fit_transform(df1['CLient'].astype('str'))
df['Date'] = number.fit_transform(df1['Date'].astype('str'))

df.tail()


# ## 4. Selection des variables 

# <h3>BoxPlot</h3>

# In[639]:


sns.boxplot(x="Category", y="projectDuration", data=df1)


# La distribution de la durée de projet entre les secteurs sont différents et distinct ce qui nous aidera à voir l'influence du secteur sur la durée du projet .

# In[640]:


sns.boxplot(x="Category", y="nb_phases", data=df1)


# La distribution des nombres dephases par projet entre les secteurs sont similaires  donc la durée du dépassement d'une activité ne peut pas être reliée au secteur .

# <h3>Chi-Square</h3>

# In[647]:



dataset = df.values
X = dataset[:, :-1]
y = dataset[:,-1]
# format all fields as string
X = X.astype(str)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

from sklearn.feature_selection import SelectKBest
from  scipy.stats import chi2 
# feature selection


dataset1 = df1.values
X = dataset1[:, :-1]
y = dataset1[:,-1]
# format all fields as string
X = X.astype(str)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

fs = SelectKBest(score_func=chi2, k='all')
fs.fit(X_train, y_train)
X_train_fs = fs.transform(X_train)
X_test_fs = fs.transform(X_test)


# what are scores for the features
for i in range(len(fs.scores_)):
	print('Feature %d: %f' % (i, fs.scores_[i]))
# plot the scores
pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
pyplot.show()
# what are scores for the features
for i in range(len(fs.scores_)):
	print('Feature %d: %f' % (i, fs.scores_[i]))
# plot the scores
pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
pyplot.show()

# what are scores for the features
for i in range(len(fs.scores_)):
	print('Feature %d: %f' % (i, fs.scores_[i]))
# plot the scores
pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
pyplot.show()
# ## 5. Modèles de prédiction

# <h3> KNN </h3>

# In[607]:


y=df1['Premier Delay']
x=df1.drop(['Premier Delay','Category','CLient','Date'], axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=23)


# In[608]:


robust = RobustScaler()
x_train=robust.fit_transform(x_train)
x_test=robust.transform(x_test)


# from sklearn.model_selection import GridSearchCV
# 
# #List Hyperparameters that we want to tune.
# leaf_size = list(range(1,30))
# n_neighbors = list(range(1,40))
# p=[1,2]
# #Convert to dictionary
# hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)
# #Create new KNN object
# knn_2 = KNeighborsClassifier()
# #Use GridSearch
# clf = GridSearchCV(knn_2, hyperparameters, cv=10)
# #Fit the model
# best_model = clf.fit(x,y)
# #Print The value of best Hyperparameters
# print('Best leaf_size:', best_model.best_estimator_.get_params()['leaf_size'])
# print('Best p:', best_model.best_estimator_.get_params()['p'])
# print('Best n_neighbors:', best_model.best_estimator_.get_params()['n_neighbors'])
# 

# In[609]:


error = []

# Calculating error for K values between 1 and 12
for i in range(1, 12):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train, y_train)
    pred_i = knn.predict(x_test)
    error.append(np.mean(pred_i != y_test))


# In[610]:


plt.figure(figsize=(12, 6))
plt.plot(range(1, 12), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')


# In[611]:


#print("Preliminary model score:")
#print(KNN_model.score(x_test,y_test))

no_neighbors = np.arange(1, 16)
train_accuracy = np.empty(len(no_neighbors))
test_accuracy = np.empty(len(no_neighbors))

for i, k in enumerate(no_neighbors):
    # We instantiate the classifier
    knn = KNeighborsClassifier(n_neighbors=k)
    # Fit the classifier to the training data
    knn.fit(x_train,y_train)
    
    # Compute accuracy on the training set
    train_accuracy[i] = knn.score(x_train, y_train)

    # Compute accuracy on the testing set
    test_accuracy[i] = knn.score(x_test, y_test)
    
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(no_neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(no_neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()


# In[612]:


KNN_model = KNeighborsClassifier(n_neighbors=4, metric='minkowski',algorithm='auto',leaf_size=1,p=1,weights='uniform')
KNN_model.fit(x_train, y_train)
print(KNN_model.score(x_test, y_test))
print(KNN_model.score(x_train, y_train))


# In[613]:


y_pred = KNN_model.predict(x_test)


# In[614]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# # Decision Tree

# In[615]:


Tree_model = DecisionTreeClassifier(splitter= 'random',
                            max_depth=8,
                            criterion='gini',
                            random_state=5)
Tree_model.fit(x_train, y_train)
print(Tree_model.score(x_test, y_test))
print(Tree_model.score(x_train, y_train))


# In[616]:


clf = DecisionTreeClassifier()
cv_score = cross_val_score(clf, x_train, y_train,scoring = 'accuracy',
                            cv = 12,
                            n_jobs = -1,
                            verbose = 5)
cv_score


# In[617]:


clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)


# In[618]:


labels=['Delayed','OnTime']
plot_confusion_matrix(Tree_model, x_test, y_test,display_labels=labels, cmap=plt.cm.Blues)
plt.show()


# In[619]:


dt_pred_prob =Tree_model.predict_proba(x_test)[:,1]
dt_auroc = roc_auc_score(y_test, dt_pred_prob)
print("DecisionTree AUROC: {}".format(dt_auroc))
dt_y_pred = Tree_model.predict(x_test)
print(classification_report(y_test, dt_y_pred,digits=6))


# ## Regression logistique 

# In[620]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics
logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(x_test, y_test)))


# In[621]:


from sklearn.metrics import confusion_matrix
from sklearn import metrics
labels=['Delayed','OnTime']
plot_confusion_matrix(logreg, x_test, y_test,display_labels=labels, cmap=plt.cm.Blues)
plt.show()


# In[622]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# <h3> Naive bai </h3>

# In[637]:


#Cross Validation
nb = {'gaussian': GaussianNB(),
      'bernoulli': BernoulliNB()}
scores = {}
for key, model in nb.items():
    s = cross_val_score(model, x_train, y_train, cv=5, scoring='accuracy')
    scores[key] = np.mean(s)
scores


# In[624]:


#model
modele = BernoulliNB()
modele.fit(x2_train, y2_train)
print(modele.fit(x2_train,y2_train))
print(modele.score(x2_train,y2_train))


# In[625]:


#sample prediction
echantillon= [x2_test.iloc[3, :]]
print ("echantillon: ", echantillon)


# In[626]:


#print "classe réelle: ", classe_reel
print ("classe prédite: ", modele.predict(echantillon))
print ("probabilités: ", modele.predict_proba(echantillon))


# In[627]:


modele.class_count_


# In[628]:


67/x_train.shape[0]


# In[629]:


modele.predict_proba(echantillon)


# In[630]:


#Les prédictions
y_pred = modele.predict(x_test)

#évaluation de notre modèle
print ("précision: ", accuracy_score(y_test, y_pred))
print ("précision: ", modele.score(x_test, y_test))


# In[631]:


# matrice de confusion
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))
labels=['Delayed','OnTime']
plot_confusion_matrix(logreg, x_test, y_test,display_labels=labels, cmap=plt.cm.Blues)
plt.show()


# In[632]:


# Classification_report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[633]:


#learning curve
train_size, train_score, test_score = learning_curve(modele, x_train, y_train, train_sizes=np.linspace(0.1,1,10), cv=5, random_state=0)
plt.plot(train_size, train_score.mean(axis=1), train_size, test_score.mean(axis=1))
plt.legend(('train_score', 'test_score'))
plt.xlabel('train_size')
plt.ylabel('score')
plt.grid(True)
plt.show()


# In[ ]:




