# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import dataset
dataset = pd.read_csv('input/churn.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# data preprocessing
# encode categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# encode countries
label_encoder_x_country = LabelEncoder()
X[:, 1] = label_encoder_x_country.fit_transform(X[:, 1])

# encode gender
label_encoder_x_gender = LabelEncoder()
X[:, 2] = label_encoder_x_gender.fit_transform(X[:, 2])

# dummy variables (binary one-hot encoding)
onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()
# avoid dummy variable trap
X = X[:, 1:]

# split dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# scale features (standardize range of independent variables)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# build artificial neural network
import keras
from keras.models import Sequential # used to initialize neural network
from keras.layers import Dense # used to create layers
from keras.layers import Dropout # used for dropout regularization

# initialize neural network
classifier = Sequential()

# add layers
# units = average of input nodes and output nodes = (11 + 1)/2 = 6

# input layer and first hidden layer
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_shape=(11,)))
classifier.add(Dropout(rate=0.1))

# second hidden layer
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
classifier.add(Dropout(rate=0.1))

# output layer
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

# compile neural network
# stochastic gradient optimizer : adam
# loss function : binary crossentropy
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# fit neural network to training set
classifier.fit(X_train, y_train, batch_size=10, epochs=100)
print ("Training complete!")

# predict test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
print ("Testing complete!")

# create confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
accuracy = (cm[0, 0] + cm[1, 1])/(cm[0, 0] + cm[1, 1] + cm[0, 1] + cm[1, 0])

# print accuracy
print ("Model accuracy: {acc}".format(acc=accuracy))

# predict a single new observation
'''
Geography: France
Credit Score: 600
Gender: Male
Age: 40 years old
Tenure: 3 years
Balance: $60000
Number of Products: 2
Does this customer have a credit card ? Yes
Is this customer an Active Member: Yes
Estimated Salary: $50000
'''

new_observation_desc = "Geography: France\nCredit Score: 600\nGender: Male\nAge: 40 years old\nTenure: 3 years\nBalance: $60000\nNumber of Products: 2\nDoes this customer have a credit card ? Yes\nIs this customer an Active Member: Yes\nEstimated Salary: $50000"

print ("\n----------------")
print ("NEW OBSERVATION:")
print ("----------------")
print (new_observation_desc)

new_observation = np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])

# preprocessing new observation
# scale new observation
new_observation = sc_X.transform(new_observation)

new_prediction_prob = classifier.predict(new_observation)
new_prediction = (new_prediction_prob > 0.5)

print ("-----------------------------------------------------------------------")
print ("Probability that new customer will leave the bank: {prob}".format(prob=new_prediction_prob[0][0]))
print ("-----------------------------------------------------------------------")


# evaluation, improvement, and tuning
# evaluation
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

# build classifier
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_shape=(11,)))
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

# global classifier
classifier = KerasClassifier(build_fn=build_classifier, batch_size=10, epochs=100)
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10, n_jobs=-1)

# mean accuracy
mean = accuracies.mean()
variance = accuracies.std()

print("Mean accuracy: {mean_acc}".format(mean_acc=mean))
print("Accuracy variance: {var_acc}".format(var_acc=variance))

# hyperparameter tuning
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

# build classifier
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_shape=(11,)))
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn=build_classifier)

# parameters
parameters = {'batch_size': [25, 32],
              'epochs': [100, 500],
              'optimizer': ['adam', 'rmsprop']}

# grid search on parameters
grid_search = GridSearchCV(estimator=classifier, param_grid=parameters, scoring='accuracy', cv=10)

grid_search = grid_search.fit(X_train, y_train)

# best parameters
best_parameters = grid_search.best_params_

# best accuracy
best_accuracy = grid_search.best_score_

print("Best parameters: {best_params}".format(best_params=best_parameters))
print("Best accuracy: {best_acc}".format(best_acc=best_accuracy))
