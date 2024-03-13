# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import missingno as msno #визуализация недостающих данных
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics

from sklearn.linear_model import LinearRegression # линейная регрессия
from sklearn.linear_model import Lasso # лассо- регресивный анализ
from sklearn.linear_model import LassoCV # Линейная модель Лассо с итеративной подгонкой по пути регуляризации.
from sklearn.linear_model import Ridge #один из методов понижения размерности, применяется для борьбы с избыточностью данных

from sklearn.preprocessing import MinMaxScaler # нормализация данных от 0 до 1
from sklearn.preprocessing import StandardScaler #предварительной обработки перед многими моделями машинного обучения, чтобы стандартизировать диапазон функциональных возможностей входного набора данных
from sklearn.model_selection import train_test_split # для разделения датасета перед обучением
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
#удаляем среднее значение и масштабирует данные до единичной дисперсии
scaler = StandardScaler()
#scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)
y = data.target
#print(x)
#Для валидации разделим данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(x, y)

"""# Линейная регрессия"""

regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred=regressor.predict(X_test)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('R2 Score:', metrics.r2_score(y_test, y_pred))

LinearModel=LinearRegression()
parameters = {'positive':[False,True]}
scoring_param = {'r2','neg_mean_absolute_error','neg_mean_squared_error'}
gscv = GridSearchCV(LinearModel, parameters,scoring=scoring_param,refit='r2', return_train_score=True)
gscv.fit(x,y)

print('best parameters:', gscv.best_params_)
print('best score:', gscv.best_score_)

CVLM=gscv.best_estimator_
CVLM.fit(X_train,y_train)
y_pred=CVLM.predict(X_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('R2 Score:', metrics.r2_score(y_test, y_pred))
lr_mae=metrics.mean_absolute_error(y_test, y_pred)
lr_mse=metrics.mean_squared_error(y_test, y_pred)
lr_r2=metrics.r2_score(y_test, y_pred)

param_num=1
x_reg = X_test[:,param_num]
y_reg = y_pred

fig, ax = plt.subplots(figsize=(12,7))
ax.scatter(x_reg,y_reg,color='red',
        label='Модель ')
ax.scatter(X_test[:,param_num],y_test,
           color='blue', label='данные')
ax.legend()
plt.show()

x_reg = np.linspace(0, 1, 143)
y_reg =np.around(y_pred)

atr_1=5
atr_2=9
fig, ax = plt.subplots(figsize=(6,6))
fig,ay= plt.subplots(figsize=(6,6))

ax.scatter(X_test[:, atr_1], X_test[:, atr_2],
c=np.where(y_reg,'green','blue'))
ax.set_xlabel('attr1')
ax.set_ylabel('attr2')
ay.scatter(X_test[:, atr_1], X_test[:, atr_2],
c=np.where(y_test,'green','blue'))
ay.set_xlabel('attr1')
ay.set_ylabel('attr2')
plt.show()

from yellowbrick.regressor import residuals_plot
viz = residuals_plot(regressor, X_train, y_train, X_test, y_test)

from yellowbrick.regressor.alphas import alphas
from yellowbrick.regressor import ManualAlphaSelection
alphas = np.logspace(0, 1, 50)

# Instantiate the visualizer
visualizer = ManualAlphaSelection(
    Ridge(),
    alphas=alphas,
    cv=12,
    scoring="r2"
)

visualizer.fit(x, y)
visualizer.show()

"""#   Лассо

"""

LassoModel = LassoCV(max_iter=4000)
LassoModel.fit(X_train, y_train)
y_pred=LassoModel.predict(X_test)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('R2 Score:', metrics.r2_score(y_test, y_pred))

LassoModel.get_params()

LassoModel=Lasso()
parameters = {'alpha': [0.01,0.05,0.1, 0.5, 1., 1.5,5,10,100],
              'positive':[False,True],
              'selection':['cyclic', 'random'],
              'tol':[1e-5,1e-4,1e-3]}
scoring_param = {'r2','neg_mean_absolute_error','neg_mean_squared_error'}
gscv = GridSearchCV(LassoModel, parameters,scoring=scoring_param,refit='r2', return_train_score=True)
gscv.fit(x,y)

LassoModel.get_params().keys()

metrics.SCORERS.keys()

print('best parameters:', gscv.best_params_)
print('best score:', gscv.best_score_)

CVLaM=gscv.best_estimator_
CVLaM.fit(X_train,y_train)
y_pred=CVLaM.predict(X_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('R2 Score:', metrics.r2_score(y_test, y_pred))
la_mae=metrics.mean_absolute_error(y_test, y_pred)
la_mse=metrics.mean_squared_error(y_test, y_pred)
la_r2=metrics.r2_score(y_test, y_pred)

param_num=9
x_reg = X_test[:,param_num]
y_reg = y_pred

fig, ax = plt.subplots(figsize=(12,7))
ax.scatter(x_reg,y_reg,color='red',
        label='Модель Lasso')
ax.scatter(X_test[:,param_num],y_test,
           color='blue', label='данные')
ax.legend()
plt.show()

x_reg = np.linspace(0, 1, 143)
y_reg =np.around(y_pred)

atr_1=3
atr_2=8
fig, ax = plt.subplots(figsize=(6,6))
fig,ay= plt.subplots(figsize=(6,6))

ax.scatter(X_test[:, atr_1], X_test[:, atr_2],
c=np.where(y_reg,'green','blue'))
ax.set_xlabel('attr1')
ax.set_ylabel('attr2')
ay.scatter(X_test[:, atr_1], X_test[:, atr_2],
c=np.where(y_test,'green','blue'))
ay.set_xlabel('attr1')
ay.set_ylabel('attr2')
plt.show()

"""# Гребневая регрессия"""

RidgeModel = Ridge(alpha=1.0)
RidgeModel.fit(X_train, y_train)
y_pred=RidgeModel.predict(X_test)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('R2 Score:', metrics.r2_score(y_test, y_pred))

RidgeModel=Ridge()
parameters = {'alpha': [0.01,0.05,0.1, 0.5, 1., 1.5,5,10,100],
              'positive':[False,True],
              'solver':['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs'],
              'tol':[1e-5,1e-4,1e-3]}
scoring_param = {'r2','neg_mean_absolute_error','neg_mean_squared_error'}
gscv = GridSearchCV(RidgeModel, parameters,scoring=scoring_param,refit='r2', return_train_score=True)
gscv.fit(x,y)

RidgeModel.get_params().keys()

print('best parameters:', gscv.best_params_)
print('best score:', gscv.best_score_)

CVRM=gscv.best_estimator_
CVRM.fit(X_train,y_train)
y_pred=CVRM.predict(X_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('R2 Score:', metrics.r2_score(y_test, y_pred))
rm_mae=metrics.mean_absolute_error(y_test, y_pred)
rm_mse=metrics.mean_squared_error(y_test, y_pred)
rm_r2=metrics.r2_score(y_test, y_pred)

x_reg = np.linspace(0, 1, 143)
y_reg =np.around(y_pred)

fig, ax = plt.subplots(figsize=(12,7))
ax.plot(x_reg,y_reg,color='red',
        label='Модель Lasso')
ax.scatter(np.linspace(0, 1, 143),y_test,
           color='blue', label='данные')
ax.legend()
plt.show()



atr_1=1
atr_2=4
fig, ax = plt.subplots(figsize=(6,6))
fig,ay= plt.subplots(figsize=(6,6))

ax.scatter(X_test[:, atr_1], X_test[:, atr_2],
c=np.where(y_reg,'green','blue'))
ax.set_xlabel('attr1')
ax.set_ylabel('attr2')
ay.scatter(X_test[:, atr_1], X_test[:, atr_2],
c=np.where(y_test,'green','blue'))
ay.set_xlabel('attr1')
ay.set_ylabel('attr2')
plt.show()

"""# Ансамбль методов"""

from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import StackingRegressor

LinearRegressionModel = LinearRegression()
LinearRegressionModel.fit(X_train, y_train)
LRM_score=LinearRegressionModel.score(X_test,y_test)
LRM_score

LassoModel = LassoCV()
LassoModel.fit(X_train, y_train)
LaM_score=LassoModel.score(X_test,y_test)
LaM_score

RidgeModel = Ridge(alpha=1.0)
RidgeModel.fit(X_train, y_train)
RM_score=RidgeModel.score(X_test,y_test)
RM_score

print('LinearRegression: {}'.format(LRM_score))
print('Lasso: {}'.format(LaM_score))
print('Ridge: {}'.format(RM_score))

model_mean = np.mean([LRM_score, LaM_score, RM_score])
print('model mean: {}'.format(model_mean))

"""## Bagging"""

LM_bg=BaggingRegressor(base_estimator=LinearRegression(), n_estimators=15, max_samples=0.75)
LM_bg.fit(X_train, y_train)
LM_bg_score=LM_bg.score(X_test,y_test)
print("LM score =", LRM_score)
print("Bagging LM =", LM_bg_score)

LaM_bg=BaggingRegressor(base_estimator=LassoCV(max_iter=5000), n_estimators=15, max_samples=0.75)
LaM_bg.fit(X_train, y_train)
LaM_bg_score=LM_bg.score(X_test,y_test)
print("LaM score =", LaM_score)
print("Bagging LaM =", LaM_bg_score)

RM_bg=BaggingRegressor(base_estimator=Ridge(), n_estimators=15, max_samples=0.75)
RM_bg.fit(X_train, y_train)
RM_bg_score=RM_bg.score(X_test,y_test)
print("RM score =", RM_score)
print("Bagging RM =", RM_bg_score)

print('Bagging LM: {}'.format(LM_bg_score))
print('Bagging LaM: {}'.format(LaM_bg_score))
print('Bagging RM: {}'.format(RM_bg_score))

model_bg_mean = np.mean([LM_bg_score, LM_bg_score, RM_bg_score])
print('bagging model mean: {}'.format(model_bg_mean))

"""## Voting"""

# список моделей - пары (имя, ссылка)
estimators=[('Bagging LM', LM_bg), ('Bagging LaM', LaM_bg), ('Bagging RM', RM_bg)]
# создадим ансамбль
ensemble = VotingRegressor(estimators)
# обучим ансамбль на обучающей выборке
ensemble.fit(X_train, y_train)
# оценка на тестовой выборке
print('model mean: {}'.format(model_mean))
print('voting : {}'.format(ensemble.score(X_test, y_test)))
vote_score=ensemble.score(X_test, y_test)

"""## Stacking"""

estimators = [('Bagging LM', LM_bg), ('Bagging LaM', LaM_bg), ('Bagging RM', RM_bg)]
meta_clf_bg = StackingRegressor(estimators=estimators)
meta_clf_bg.fit(X_train, y_train)
print('bagging model mean: {}'.format(model_bg_mean))
print('voting: {}'.format(ensemble.score(X_test, y_test)))
print('stacking: {}'.format(meta_clf_bg.score(X_test, y_test)))
stack_score=meta_clf_bg.score(X_test, y_test)

y_pred=meta_clf_bg.predict(X_test)

y_reg =np.around(y_pred)

atr_1=3
atr_2=8
fig, ax = plt.subplots(figsize=(6,6))
fig,ay= plt.subplots(figsize=(6,6))

ax.scatter(X_test[:, atr_1], X_test[:, atr_2],
c=np.where(y_reg,'green','blue'))
ax.set_xlabel('attr1')
ax.set_ylabel('attr2')
ay.scatter(X_test[:, atr_1], X_test[:, atr_2],
c=np.where(y_test,'green','blue'))
ay.set_xlabel('attr1')
ay.set_ylabel('attr2')
plt.show()

x_reg = np.linspace(0, 1, 143)
y_reg =np.around(y_pred)

fig, ax = plt.subplots(figsize=(12,7))
ax.plot(x_reg,y_reg,color='red',
        label='Stacking')
ax.scatter(np.linspace(0, 1, 143),y_test,
           color='blue', label='данные')
ax.legend()
plt.show()

"""# Нейронка"""

import tensorflow #открытая  библиотека для машинного обучения для тренировки нейронной сети
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='mse', metrics=['mae','mse'])

history = model.fit(X_train,
                    y_train,
                    epochs=100,
                    validation_split=0.1,
                    verbose=2)

pred = model.predict(X_test)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, pred))
print('R2 Score:', metrics.r2_score(y_test, pred))

plt.plot(history.history['mse'],
         label='Средняя квадратичная ошибка на обучающем наборе')
plt.plot(history.history['val_mse'],
         label='Средняя квадратичная ошибка на проверочном наборе')
plt.xlabel('Время обучения')
plt.ylabel('Средняя квадратичная ошибка')
plt.legend()
plt.show()

plt.plot(history.history['mae'],
         label='Средняя абсолютная ошибка на обучающем наборе')
plt.plot(history.history['val_mae'],
         label='Средняя абсолютная ошибка на проверочном наборе')
plt.xlabel('Время обучения')
plt.ylabel('Средняя абсолютная ошибка')
plt.legend()
plt.show()

x_reg = np.linspace(0, 1, 143)
y_reg =pred

fig, ax = plt.subplots(figsize=(12,7))
ax.plot(x_reg,y_reg,color='red',
        label='Pred')
ax.scatter(np.linspace(0, 1, 143),y_test,
           color='blue', label='данные')
ax.legend()
plt.show()

param_num=3
x_reg = X_test[:,param_num]
y_reg = pred

fig, ax = plt.subplots(figsize=(12,7))
ax.scatter(x_reg,y_reg,color='red',
        label='Pred')
ax.scatter(X_test[:,param_num],y_test,
           color='blue', label='данные')
ax.legend()
plt.show()

!git clone https://github.com/keras-team/keras-tuner.git

# Commented out IPython magic to ensure Python compatibility.
# %cd keras-tuner
!pip install .

from kerastuner import RandomSearch,Hyperband,BayesianOptimization
def build_model(hp):
    model=Sequential()
    hp_units1 = hp.Int('units1', min_value=5, max_value=1024, step=16)
    hp_units2 = hp.Int('units2', min_value=5, max_value=1024, step=16)
    activation_choice1=hp.Choice('activation1',values=['relu','sigmoid','tanh','elu','selu'])
    activation_choice2=hp.Choice('activation2',values=['relu','sigmoid','tanh','elu','selu'])
    model.add(Dense(units=hp_units1,input_dim=X_train.shape[1],activation=activation_choice1))
    model.add(Dense(units=hp_units2,activation=activation_choice2))
    model.add(Dense(1))
    model.compile(
        optimizer=hp.Choice('optimizer',values=['adam','rmsprop','SGD']),
        loss='MeanSquaredError',
        metrics=['MeanSquaredError'])
    return model

tuner=BayesianOptimization(
    build_model,
    objective='loss',

    max_trials=30,
    directory='test_directory',
    overwrite=True,
)

tuner.search_space_summary()

tuner.search(X_train,y_train,batch_size=40,epochs=50,validation_split=0.2,verbose=1)

models=tuner.get_best_models(num_models=3)
for model in models:
  model.summary()

model = Sequential()
model.add(Dense(229, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(677, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='mse', metrics=['mae','mse'])

history = model.fit(X_train,
                    y_train,
                    epochs=100,
                    validation_split=0.1,
                    verbose=2)

pred = model.predict(X_test)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, pred))
print('R2 Score:', metrics.r2_score(y_test, pred))

"""# Графики сравнения моделей"""

fig, ax = plt.subplots(figsize=(12,7))
ax.plot([1,2,3],[lr_mae,lr_mse,lr_r2],color='red',
        label='Linear')
ax.plot([1,2,3],[la_mae,la_mse,la_r2],color='green',
        label='Lasso')
ax.plot([1,2,3],[rm_mae,rm_mse,rm_r2],color='blue',
        label='Ridge')
ax.legend()
plt.show()

fig, ax = plt.subplots(figsize=(12,7))
ax.scatter([1,2,3],[model_bg_mean,stack_score,vote_score],color='green', label='bagging/stack/vote')

ax.legend()
plt.show()