# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import missingno as msno
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from yellowbrick.classifier import ClassPredictionError

from sklearn.model_selection import GridSearchCV # Поиск по решетке
from sklearn.model_selection import cross_val_score # оценка кросс-валидации
from sklearn.model_selection import cross_validate # кросс-валидация

from sklearn import datasets
data = datasets.load_breast_cancer()
#print(data.DESCR)

df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
df.head(5)

# Информация о данных
df.info()

x = data.data
y = data.target
print(df['target'].value_counts())
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

print(X_train)

"""# Логистическая регрессия"""

from sklearn.linear_model import LogisticRegression
LogisticRegressionModel = LogisticRegression()
LogisticRegressionModel.fit(X_train, y_train)
y_pred=LogisticRegressionModel.predict(X_test)

print('precision_score:', metrics.precision_score(y_test, y_pred))
print('f1_score:', metrics.f1_score(y_test, y_pred))
print('Accuracy score:', metrics.accuracy_score(y_test, y_pred))

oz = ClassPredictionError(LogisticRegression())
oz.fit(X_train, y_train)
oz.score(X_test, y_test)
oz.show()

fig, ax = plt.subplots(figsize=(6,6))
fig,ay= plt.subplots(figsize=(6,6))

ax.scatter(X_test[:, 5], X_test[:, 8],
c=np.where(y_pred,'green','blue'))
ax.set_xlabel('attr1')
ax.set_ylabel('attr2')
ay.scatter(X_test[:, 5], X_test[:, 8],
c=np.where(y_test,'green','blue'))
ay.set_xlabel('attr1')
ay.set_ylabel('attr2')
plt.show()

LogisticRegressionModel.get_params().keys()

LinRegresModel=LogisticRegression()
parameters = { 'penalty':['l1', 'l2', 'elasticnet', None],
               'C':[1,10,100],
              'tol':[1e-5,1e-4,1e-3],
              'solver':['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky']}
scoring_param = {'accuracy','f1','precision'}
gscv = GridSearchCV(LogisticRegressionModel, parameters,scoring=scoring_param,refit='accuracy', return_train_score=True)
gscv.fit(x,y)

print('best parameters:', gscv.best_params_)
print('best score:', gscv.best_score_)

CVLRM=gscv.best_estimator_
CVLRM.fit(X_train,y_train)
y_pred=CVLRM.predict(X_test)

print('precision_score:', metrics.precision_score(y_test, y_pred))
print('f1_score:', metrics.f1_score(y_test, y_pred))
print('Accuracy score:', metrics.accuracy_score(y_test, y_pred))

param_num=1
x_reg = X_test[:,param_num]
y_reg = y_pred

fig, ax = plt.subplots(figsize=(12,7))
ax.scatter(x_reg,y_reg,color='red',
        label='Модель LogReg')
ax.scatter(X_test[:,param_num],y_test,
           color='blue', label='данные')
ax.legend()
plt.show()

oz = ClassPredictionError(CVLRM)
oz.fit(X_train, y_train)
oz.score(X_test, y_test)
oz.show()

"""# Гребневый классификатор"""

from sklearn.linear_model import RidgeClassifier
RidgeClassifierModel = RidgeClassifier(alpha=1.0)
RidgeClassifierModel.fit(X_train, y_train)
y_pred=RidgeClassifierModel.predict(X_test)

print('precision_score:', metrics.precision_score(y_test, y_pred))
print('f1_score:', metrics.f1_score(y_test, y_pred))
print('Accuracy score:', metrics.accuracy_score(y_test, y_pred))

oz = ClassPredictionError(RidgeClassifier())
oz.fit(X_train, y_train)
oz.score(X_test, y_test)
oz.show()

RidgeClassifierModel.get_params().keys()

RidgeClassifierModel=RidgeClassifier()
parameters = {'alpha':[1,5,10,100],
              'tol':[1e-5,1e-4,1e-3],
              'solver':['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs']}
scoring_param = {'accuracy','f1','precision'}
gscv = GridSearchCV(RidgeClassifierModel, parameters,scoring=scoring_param,refit='accuracy', return_train_score=True)
gscv.fit(x,y)

print('best parameters:', gscv.best_params_)
print('best score:', gscv.best_score_)

CVRM=gscv.best_estimator_
CVRM.fit(X_train,y_train)
y_pred=CVRM.predict(X_test)

print('precision_score:', metrics.precision_score(y_test, y_pred))
print('f1_score:', metrics.f1_score(y_test, y_pred))
print('Accuracy score:', metrics.accuracy_score(y_test, y_pred))

fig, ax = plt.subplots(figsize=(6,6))
fig,ay= plt.subplots(figsize=(6,6))

ax.scatter(X_test[:, 5], X_test[:, 8],
c=np.where(y_pred,'green','blue'))
ax.set_xlabel('attr1')
ax.set_ylabel('attr2')
ay.scatter(X_test[:, 5], X_test[:, 8],
c=np.where(y_test,'green','blue'))
ay.set_xlabel('attr1')
ay.set_ylabel('attr2')
plt.show()

param_num=1
x_reg = X_test[:,param_num]
y_reg = y_pred

fig, ax = plt.subplots(figsize=(12,7))
ax.scatter(x_reg,y_reg,color='red',
        label='Модель LogReg')
ax.scatter(X_test[:,param_num],y_test,
           color='blue', label='данные')
ax.legend()
plt.show()

"""# SGDClassifier Градиентный спуск"""

from sklearn.linear_model import SGDClassifier
SGDClassifierModel = SGDClassifier(alpha=1.0)
SGDClassifierModel.fit(X_train, y_train)
y_pred=SGDClassifierModel.predict(X_test)

print('precision_score:', metrics.precision_score(y_test, y_pred))
print('f1_score:', metrics.f1_score(y_test, y_pred))
print('Accuracy score:', metrics.accuracy_score(y_test, y_pred))

SGDClassifierModel.get_params()

oz = ClassPredictionError(SGDClassifier())
oz.fit(X_train, y_train)
oz.score(X_test, y_test)
oz.show()

SGDClassifierModel.get_params().keys()

SGDClassifierModel=SGDClassifier()
parameters = {'alpha':[0.0001,0.001,0.01,1,10],
              'tol':[1e-5,1e-4,1e-3],
              'epsilon':[0,0.1,1,10],
              'penalty':['l1', 'l2', 'elasticnet', None],
              'loss':['hinge', 'log_loss', 'modified_huber', 'squared_hinge', 'perceptron']}
scoring_param = {'accuracy','f1','precision'}
gscv = GridSearchCV(SGDClassifierModel, parameters,scoring=scoring_param,refit='accuracy', return_train_score=True)
gscv.fit(x,y)

print('best parameters:', gscv.best_params_)
print('best score:', gscv.best_score_)

CVSGDM=gscv.best_estimator_
CVSGDM.fit(X_train,y_train)
y_pred=CVSGDM.predict(X_test)

print('precision_score:', metrics.precision_score(y_test, y_pred))
print('f1_score:', metrics.f1_score(y_test, y_pred))
print('Accuracy score:', metrics.accuracy_score(y_test, y_pred))

gscv.best_estimator_.get_params()

fig, ax = plt.subplots(figsize=(6,6))
fig,ay= plt.subplots(figsize=(6,6))

ax.scatter(X_test[:, 5], X_test[:, 8],
c=np.where(y_pred,'green','blue'))
ax.set_xlabel('attr1')
ax.set_ylabel('attr2')
ay.scatter(X_test[:, 5], X_test[:, 8],
c=np.where(y_test,'green','blue'))
ay.set_xlabel('attr1')
ay.set_ylabel('attr2')
plt.show()

oz = ClassPredictionError(CVSGDM)
oz.fit(X_train, y_train)
oz.score(X_test, y_test)
oz.show()

"""# Деревья решений"""

from sklearn.tree import DecisionTreeClassifier
DTCModel = DecisionTreeClassifier()
DTCModel.fit(X_train, y_train)
y_pred=DTCModel.predict(X_test)

print('precision_score:', metrics.precision_score(y_test, y_pred))
print('f1_score:', metrics.f1_score(y_test, y_pred))
print('Accuracy score:', metrics.accuracy_score(y_test, y_pred))

oz = ClassPredictionError(DTCModel)
oz.fit(X_train, y_train)
oz.score(X_test, y_test)
oz.show()

DTCModel.get_params().keys()

DTCModel.get_params()

DTCModel=DecisionTreeClassifier()
parameters = {'splitter':['best', 'random'],
              'criterion':['gini', 'entropy', 'log_loss'],
              'min_samples_split':[2,3,5],
              'min_samples_leaf':[1,2,3]}
scoring_param = {'accuracy','f1','precision'}
gscv = GridSearchCV(DTCModel, parameters,scoring=scoring_param,refit='accuracy', return_train_score=True)
gscv.fit(x,y)

print('best parameters:', gscv.best_params_)
print('best score:', gscv.best_score_)

CVDTCM=gscv.best_estimator_
CVDTCM.fit(X_train,y_train)
y_pred=CVDTCM.predict(X_test)

print('precision_score:', metrics.precision_score(y_test, y_pred))
print('f1_score:', metrics.f1_score(y_test, y_pred))
print('Accuracy score:', metrics.accuracy_score(y_test, y_pred))

fig, ax = plt.subplots(figsize=(6,6))
fig,ay= plt.subplots(figsize=(6,6))

ax.scatter(X_test[:, 5], X_test[:, 8],
c=np.where(y_pred,'green','blue'))
ax.set_xlabel('attr1')
ax.set_ylabel('attr2')
ay.scatter(X_test[:, 5], X_test[:, 8],
c=np.where(y_test,'green','blue'))
ay.set_xlabel('attr1')
ay.set_ylabel('attr2')
plt.show()

oz = ClassPredictionError(CVDTCM)
oz.fit(X_train, y_train)
oz.score(X_test, y_test)
oz.show()

"""# Байесовский классификатор"""

from sklearn.naive_bayes import GaussianNB
GNBModel = GaussianNB()
GNBModel.fit(X_train, y_train)
y_pred=GNBModel.predict(X_test)

print('precision_score:', metrics.precision_score(y_test, y_pred))
print('f1_score:', metrics.f1_score(y_test, y_pred))
print('Accuracy score:', metrics.accuracy_score(y_test, y_pred))

GNBModel.get_params()

oz = ClassPredictionError(GNBModel)
oz.fit(X_train, y_train)
oz.score(X_test, y_test)
oz.show()

GNBModel.get_params().keys()

GNBModel=GaussianNB()
parameters = {'var_smoothing':[1e-10,1e-9,1e-8,1e-5,1e-3,1e-1,10]}
scoring_param = {'accuracy','f1','precision'}
gscv = GridSearchCV(GNBModel, parameters,scoring=scoring_param,refit='precision', return_train_score=True)
gscv.fit(x,y)

print('best parameters:', gscv.best_params_)
print('best score:', gscv.best_score_)

CVGNBM=gscv.best_estimator_
CVGNBM.fit(X_train,y_train)
y_pred=CVGNBM.predict(X_test)

print('precision_score:', metrics.precision_score(y_test, y_pred))
print('f1_score:', metrics.f1_score(y_test, y_pred))
print('Accuracy score:', metrics.accuracy_score(y_test, y_pred))

oz = ClassPredictionError(CVGNBM)
oz.fit(X_train, y_train)
oz.score(X_test, y_test)
oz.show()

"""# Ансамбль моделей"""

from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import StackingClassifier

LogisticRegressionModel = LogisticRegression()
LogisticRegressionModel.fit(X_train, y_train)
LRM_score=LogisticRegressionModel.score(X_test,y_test)
LRM_score

RidgeClassifierModel = RidgeClassifier()
RidgeClassifierModel.fit(X_train, y_train)
RCM_score=RidgeClassifierModel.score(X_test,y_test)
RCM_score

SGDClassifierModel = SGDClassifier()
SGDClassifierModel.fit(X_train, y_train)
SGDCM_score=SGDClassifierModel.score(X_test,y_test)
SGDCM_score

print('LogisticRegression: {}'.format(LRM_score))
print('RidgeClassifier: {}'.format(RCM_score))
print('SGDClassifier: {}'.format(SGDCM_score))

model_mean = np.mean([LRM_score, RCM_score, SGDCM_score])
print('model mean: {}'.format(model_mean))

"""## Bagging"""

LRM_bg=BaggingClassifier(base_estimator=LogisticRegression(), n_estimators=15, max_samples=0.75)
LRM_bg.fit(X_train, y_train)
LRM_bg_score=LRM_bg.score(X_test,y_test)
print("LM score =", LRM_score)
print("Bagging LM =", LRM_bg_score)

RCM_bg=BaggingClassifier(base_estimator=RidgeClassifier(), n_estimators=15, max_samples=0.75)
RCM_bg.fit(X_train, y_train)
RCM_bg_score=RCM_bg.score(X_test,y_test)
print("LaM score =", RCM_score)
print("Bagging LaM =", RCM_bg_score)

SGDCM_bg=BaggingClassifier(base_estimator=SGDClassifier(), n_estimators=15, max_samples=0.75)
SGDCM_bg.fit(X_train, y_train)
SGDCM_bg_score=SGDCM_bg.score(X_test,y_test)
print("RM score =", SGDCM_score)
print("Bagging RM =", SGDCM_bg_score)

print('Bagging LM: {}'.format(LRM_bg_score))
print('Bagging LaM: {}'.format(RCM_bg_score))
print('Bagging RM: {}'.format(SGDCM_bg_score))

model_bg_mean = np.mean([LRM_bg_score, RCM_bg_score, SGDCM_bg_score])
print('bagging model mean: {}'.format(model_bg_mean))

"""## Voting"""

# список моделей - пары (имя, ссылка)
estimators=[('Bagging LogisticRegression', LRM_bg), ('Bagging RidgeClassifier', RCM_bg), ('Bagging SGDClassifier', SGDCM_bg)]
# создадим ансамбль
ensemble = VotingClassifier(estimators, voting='hard')
# обучим ансамбль на обучающей выборке
ensemble.fit(X_train, y_train)
# оценка на тестовой выборке
print('model mean: {}'.format(model_mean))
print('voting : {}'.format(ensemble.score(X_test, y_test)))

oz = ClassPredictionError(ensemble)
oz.fit(X_train, y_train)
oz.score(X_test, y_test)
oz.show()

"""## Stacking"""

estimators=[('Bagging LogisticRegression', LRM_bg), ('Bagging RidgeClassifier', RCM_bg), ('Bagging SGDClassifier', SGDCM_bg)]
meta_clf_bg = StackingClassifier(estimators=estimators)
meta_clf_bg.fit(X_train, y_train)
print('bagging model mean: {}'.format(model_bg_mean))
print('voting: {}'.format(ensemble.score(X_test, y_test)))
print('stacking: {}'.format(meta_clf_bg.score(X_test, y_test)))

oz = ClassPredictionError(meta_clf_bg)
oz.fit(X_train, y_train)
oz.score(X_test, y_test)
oz.show()
