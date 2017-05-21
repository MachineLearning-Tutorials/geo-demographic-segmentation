# artifical neural network

# data preprocessing

# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# encode categorical independent variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

# dummy variables
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# split dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# ANN

# import Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# initialize ANN
classifier = Sequential()

# add input layer and first hidden layer with dropout
classifier.add(Dense(units = 6, activation = 'relu', kernel_initializer = 'uniform', input_shape = (11,)))
classifier.add(Dropout(rate = 0.1))

# add second hidden layer with dropout
classifier.add(Dense(units = 6, activation = 'relu', kernel_initializer = 'uniform'))
classifier.add(Dropout(rate = 0.1))

# add output layer
classifier.add(Dense(units = 1, activation = 'sigmoid', kernel_initializer = 'uniform'))

# compile ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# fit ANN to training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# make predictions and evaluate model

# predict test set results
y_pred = classifier.predict(X_test)

# convert probabilities to true/false
y_pred = (y_pred > 0.5)

# making a single prediction
"""Predict if the customer with the following information will leave the bank:
Geography: France
Credit Score: 600
Gender: Male
Age: 40
Tenure: 3
Balance: 60000
Number of Products: 2
Has Credit Card: Yes
Is Active Member: Yes
Estimated Salary: 50000"""
new_prediction = classifier.predict(sc_X.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = (new_prediction > 0.5)

# create confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# evaluation, improvement, and parameter tuning

# evaluation
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier() :
    classifier = Sequential()
    classifier.add(Dense(units = 6, activation = 'relu', kernel_initializer = 'uniform', input_shape = (11,)))
    classifier.add(Dense(units = 6, activation = 'relu', kernel_initializer = 'uniform'))
    classifier.add(Dense(units = 1, activation = 'sigmoid', kernel_initializer = 'uniform'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

# k-fold cross validation
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
mean = accuracies.mean()
variance = accuracies.std()

# improve neural network
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

def build_classifier(optimizer) :
    classifier = Sequential()
    classifier.add(Dense(units = 6, activation = 'relu', kernel_initializer = 'uniform', input_shape = (11,)))
    classifier.add(Dense(units = 6, activation = 'relu', kernel_initializer = 'uniform'))
    classifier.add(Dense(units = 1, activation = 'sigmoid', kernel_initializer = 'uniform'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier)

# parameter tuning

# dictionary to store hyperparameters
parameters = {'batch_size': [25, 32],
              'epochs': [100, 500],
              'optimizer': ['adam', 'rmsprop']}

# grid search
grid_search = GridSearchCV(estimator = classifier, 
                           param_grid = parameters, 
                           scoring = 'accuracy', 
                           cv = 10)

grid_search = grid_search.fit(X_train, y_train)

# find out best parameters and accuracies
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_









