# Breast_Cancer_classification
## The task of classifying breast cancer using various methods and evaluating their results. / Задача классификации рака груди различными методами и оценка их результатов
![image](https://github.com/ArtemAvgutin/Breast_Cancer_classification/assets/131138862/04491b6d-8657-484e-9af8-7f0ac93634aa)

### (Rus) Задача классификации рака груди является важной задачей в области медицинского анализа изображений. Поэтому задачей было использование различных классификаторов, для определения их точности. [Данные взяты из соревнований Kaggle.](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
### Результат: В результате работы были созданы классификаторы для решение данной задачи и вычислена их точнсоть работы.
#### Краткое описание применения моделей, которые были использованы:
* Логистическая регрессия - это метод бинарной классификации, который может использоваться для предсказания вероятности того, что опухоль является злокачественной или доброкачественной. Точность - 93%
* Гребневый классификатор (Ridge Classifier) - это метод регуляризации, который помогает уменьшить переобучение модели путем добавления штрафа за большие значения весов. Точность - 80%
* SGDClassifier (стохастический градиентный спуск) - это метод оптимизации, который можно применить для обучения моделей машинного обучения, включая классификацию рака груди. Точность - 93%
* Деревья решений - это метод обучения с учителем, который ищет оптимальные правила принятия решений на основе исходных данных для классификации опухолей. Точность - 92%
* Байесовский классификатор - это вероятностный метод классификации, который может использоваться для прогнозирования вероятности обнаружения рака груди на основе статистических свойств данных. Точность - 95%
* Метод лассо - заключается во введении дополнительного слагаемого регуляризации в функционал оптимизации модели. Точность - 76%
* Ансамбль моделей объединяет несколько моделей машинного обучения для повышения точности предсказаний. Например, случайный лес или градиентный бустинг могут быть использованы как ансамбли для классификации рака груди. Точность - 81% и 95%
* Также была использована нейронная сеть. Точность - 90%

### (Eng) The task of breast cancer classification is an important task in the field of medical image analysis. [Data taken from Kaggle competitions.](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
### Result: As a result of the work, classifiers were created to solve this problem and their accuracy was calculated.
#### Brief description of the application of the models that were used:
* Logistic regression is a binary classification method that can be used to predict the likelihood that a tumor is malignant or benign. Accuracy - 93%
* Ridge Classifier is a regularization method that helps reduce model overfitting by adding a penalty for large weight values. Accuracy - 80%
* SGDClassifier (Stochastic Gradient Descent) is an optimization technique that can be applied to train machine learning models, including breast cancer classification. Accuracy - 93%
* Decision trees are a supervised learning method that searches for optimal decision rules based on input data for tumor classification. Accuracy - 92%
* A Bayesian classifier is a probabilistic classification method that can be used to predict the likelihood of detecting breast cancer based on statistical properties of the data. Accuracy - 95%
* Lasso method - involves introducing an additional regularization term into the model optimization functional. Accuracy - 76%
* Model Ensemble combines multiple machine learning models to improve prediction accuracy. For example, random forest or gradient boosting can be used as ensembles for breast cancer classification. Accuracy - 81% & 95%
* A neural network was also used. Accuracy - 90%
