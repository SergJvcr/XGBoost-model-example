# Import relevant libraries and modules
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from xgboost import plot_importance # plot feature importance
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
import matplotlib.pyplot as plt
# This displays all of the columns, preventing Juptyer from redacting them
pd.set_option('display.max_columns', None)
import pickle as pkl

airline_data = pd.read_csv('google_data_analitics\\Invistico_Airline.csv')

print(airline_data.head(10))
print(airline_data.dtypes)
print(airline_data.info())

# MODEL PREPARATION
airline_data_prepared = pd.get_dummies(airline_data, columns=['satisfaction', 'Customer Type', 'Type of Travel', 'Class'])
print(airline_data_prepared.tail(10))

# Define the y (target) variable
y = airline_data_prepared['satisfaction_satisfied']
# Define the X (predictor) variables
X = airline_data_prepared.drop(['satisfaction_satisfied', 'satisfaction_dissatisfied'], axis=1)
# Perform the split operation on a data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

# MODEL BUILDING
# Define a XGBClassifier
xgb_classifier = XGBClassifier(objective='binary:logistic', random_state=42)
# Define parameters for tuning
cv_params = {'max_depth':[2, 4, 6],
             'min_child_weight':[3, 5],
             'learning_rate':[0.1, 0.2, 0.3],
             'n_estimators':[5, 10, 15],
             'subsample':[0.7],
             'colsample_bytree':[0.7]
            }
# Define your criteria as `scoring`
scoring = ['accuracy', 'precision', 'recall', 'f1']
# Construct GridSearch cross-validation
xgb_cross_validation = GridSearchCV(estimator=xgb_classifier, param_grid=cv_params, scoring=scoring, cv=5, refit='f1')
# fit the GridSearch model to training data
xgb_cross_validation.fit(X_train, y_train)
print('--------The best parameters for XGBosst model:--------')
print(xgb_cross_validation.best_params_)

# Use `pickle` to save the trained model
path = 'D:\\projects\\simple_examples\\google_data_analitics\\'

# Pickle the model
with open(path + 'xgb_cross_validation_model.pickle', 'wb') as to_write:
    pkl.dump(xgb_cross_validation, to_write) 

# Open pickled model
with open(path+'xgb_cross_validation_model.pickle', 'rb') as to_read:
    xgb_cross_validation = pkl.load(to_read)

# RESULTS AND EVALUTION
# Apply the model to predict on the test data
y_predict = xgb_cross_validation.predict(X_test)
# Print scores
ac_score = accuracy_score(y_test, y_predict)
print(f'Accuracy score final XGB model: {round(ac_score, 3)}')
pc_score = precision_score(y_test, y_predict)
print(f'Precision score final XGB model: {round(pc_score, 3)}')
rc_score = recall_score(y_test, y_predict)
print(f'Recall score final XGB model: {round(rc_score, 3)}')
f_1_score = f1_score(y_test, y_predict)
print(f'F1 score final XGB model: {round(f_1_score, 3)}')

# Construct and display a confusion matrix
conf_matrix = confusion_matrix(y_test, y_predict, labels=xgb_cross_validation.classes_)
disp_conf_matrix = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=xgb_cross_validation.classes_)
disp_conf_matrix.plot(values_format='')
plt.show()

# Plot the relative feature importance of the predictor variables in your model
fig, ax = plt.subplots(figsize=(10, 9))
plot_importance(xgb_cross_validation.best_estimator_, color='orange', ax=ax)
plt.show()

# Create a table of results to compare model performance.
table = pd.DataFrame({'Model': "Tuned XGBoost",
                      'F1':  [f_1_score],
                      'Recall': [rc_score],
                      'Precision': [pc_score],
                      'Accuracy': [ac_score]
                      })
print(table)