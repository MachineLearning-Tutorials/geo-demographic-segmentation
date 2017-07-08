# Geo-demographic segmentation model

This is the code for a double-layer feedforward neural network that predicts which customers are at the highest risk of leaving a (simulated) bank company. 

## Overview

I built the model using the [Keras](https://keras.io/) library, which is built on top of [Tensorflow](https://www.tensorflow.org/) and [Theano](http://deeplearning.net/software/theano/). The inputs are a mixture of numeric, binary, as well as categorical values, and the output is binary (indicating whether the customer leaves the bank). I used [adam](https://arxiv.org/pdf/1412.6980.pdf) for stochastic optimization, and [binary crossentropy](http://heliosphan.org/cross-entropy.html) as the loss function.

## Dependencies

- tensorflow
- keras
- numpy
- pandas
- scikit-learn

Install dependencies using [pip](https://pip.pypa.io/en/stable/).

## Dataset
I used a [simulated dataset](https://www.superdatascience.com/deep-learning/) (input/churn.csv) with 10,000 observations (customers) and 13 attributes.

| Variable  | Definition |
| ------------- | ------------- |
| CustomerId  | Customer's account ID  |
| Surname  | Customer's surname  |
| CreditScore  | Customer's credit score  |
| Geography  | Country (France/Germany/Spain)  |
| Gender  | Customer's gender (Male/Female)  |
| Age  | Customer's age  |
| Tenure  | Number of years customer has been with the bank  |
| Balance  | Customer's account balance  |
| NumOfProducts  | Number of bank products used by customer  |
| HasCrCard  | Does customer have a credit card?  |
| IsActiveMember  | Is the customer an active member?  |
| EstimatedSalary  | Customer's estimated salary  |
| Exited  | Did the customer leave the bank? |

## Usage
Run `python script.py` in terminal to see the network in training.

### Test Run
```
...
Epoch 98/100
8000/8000 [==============================] - 2s - loss: 0.3924 - acc: 0.8361     
Epoch 99/100
8000/8000 [==============================] - 2s - loss: 0.3931 - acc: 0.8374     
Epoch 100/100
8000/8000 [==============================] - 2s - loss: 0.3929 - acc: 0.8367     
Training complete!
Testing complete!
Model accuracy: 0.8455

----------------
NEW OBSERVATION:
----------------
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
-----------------------------------------------------------------------
Probability that new customer will leave the bank: 0.058069657534360886
-----------------------------------------------------------------------
```
