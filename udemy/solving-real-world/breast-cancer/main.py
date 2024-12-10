import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

df_cancer = pd.DataFrame(np.c_[cancer['data'],cancer['target']], columns=np.append(cancer['feature_names'], ['target']))

X = df_cancer.drop(['target'], axis=1)
Y = df_cancer['target']

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=5)

from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

svc_model = SVC()
svc_model.fit(X_train, Y_train)

Y_pred = svc_model.predict(X_test)
# ###############       Predicted Positive    Predicted Negative #
# Actual Positive        TP                    FN                #
# Actual Negative        FP                    TN                #
# #################################################################
confusion = confusion_matrix(Y_test, Y_pred)


# # normalize the data
min_train = X_train.min()
max_train = X_train.max()
range_train = (max_train - min_train)

X_train_scaled = (X_train - min_train)/range_train

sns.scatterplot(x=X_train_scaled['mean area'], y=X_train_scaled['mean smoothness'], hue=Y_train)
# plt.show()

min_test = X_test.min()
range_test = (X_test.max() - X_test.min())
X_test_scaled = (X_test - min_test)/range_test

svc_model.fit(X_train_scaled, Y_train)

Y_pred = svc_model.predict(X_test_scaled)
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']}
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(SVC(), param_grid,refit=True, verbose=4)

grid.fit(X_train_scaled, Y_train)

grid_predictions = grid.predict(X_test_scaled)

confusion = confusion_matrix(Y_test, grid_predictions)
sns.heatmap(confusion, annot=True)
plt.show()