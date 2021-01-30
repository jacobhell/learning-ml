# First Deep Learning Project

## Defining the Model

For my first deep learning project, I followed the steps here: https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/.

This project uses Keras with a Tensorflow backend. I've often been confused about the difference between Keras, Theano, and Tensorflow.

Here are the differences:

* **Keras**: Sits on top of Tensorflow.
* **TensorFlow**: Framework created to replace Theano. More focused on deep learning.
* **Theano**: Development has ceased as of 2017.

First we use the numpy library to import a .csv file:

```python
from numpy import loadtxt
...
dataset = loadtxt('pima-indians-diabetes.data.csv', delimiter=',')
```

The data has 8 columns. The first 7 columns are inputs with the eighth being an output variable.

We then get the input variables using numpy:

```python
x_variables = dataset[:, 0:8]
y_variables = dataset[:, 8]
```

Next, we load up Keras to create a model. We are using a sequential model.

When adding to a sequential model, the first thing to get right is the number of input variables.

The number of input variables is set using the `input_dim` argument in a Dense class:

```python
model = Sequential()
# first hidden layer
# input_dim = variables for data
model.add(Dense(12, input_dim=8, activation='relu'))
```

In this project we are going to use 3 layers.

The first argument of the Dense class is the amount of neurons/nodes. We then define an activation function.

An activation function defines the output based on the input.

The most performant activation is the rectified linear unit activation function or relu.

## Compiling the model

Next we are going to compile the model.

To compile a model, we must first choose a loss function to evaluate the weights on the network.

We choose cross entropy as the loss argument for this problem. We also choose stochastic gradient descent as the optimizer function.

Lastly, we choose accuracy as the metric to track.

```python
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

## Fitting the model

Training occurs over epochs and each epoch is split into batches

* Epoch: One pass through all rows
* Batch: One or more samples considered by the model within an epoch

These configurations can be chosen experimentally by trial and error.

We want to train the model good enough so it learns good enough mapping for classification.

There will always be error, but the error levels out after some point. This is called model convergence.

```python
model.fit(x_variables, y_variables, epochs=150, batch_size=10)
```

## Evaluate the model

Normally, you would separate test and training data before evaluation of the model

```python
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))
```

So that's it for my first deep learning project.

What I want to do in next project:

* Separate training and evaluating data
* Save a model to a file
* Create a visualization of model
* Use JuPyter

What I want to do in future projects:

* Plot learning curves
* Learn new dataset
