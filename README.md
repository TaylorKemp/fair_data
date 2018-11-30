# Fair Data

## About


## Getting Started

The dataset used for this project can be found [here](https://www.kaggle.com/danofer/compass). This project makes use of [Tensorflow](https://www.tensorflow.org/), [Numpy](http://www.numpy.org/), [Pandas](https://pandas.pydata.org/), [Matplotlib](https://matplotlib.org/). This code was writen in [Python 3.6](https://www.python.org/downloads/release/python-367/). 

## Running 

In order to run the script in terminal type the following. 

'''console
foo@bar:~$ python run.py
'''

The code should present a plot of the accuracies of the model using different importance placed on the regularization parameter for fair data. If you then type 

'''console
foo@bar:~$ tensorboard --logdir="graph"
'''
and go to the correct local host in a window, you should be able to see more training information as well as the graph of the model used for training. 

