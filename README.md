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
I used a [simulated dataset](https://www.superdatascience.com/deep-learning/) (inputs/churn.csv) with 10,000 observations (customers) and 13 attributes.

| Variable  | Definition |
| ------------- | ------------- |
| CustomerId  | Customer's account ID  |
| Surname  | Customer's surname  |
| CreditScore  | Customer's credit score  |
| Geography  | Country (France/Spain/Germany)  |
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
