# -*- coding: utf-8 -*-
"""
Created on Sat Dec 05 23:38:33 2015
assert self.n_layers > 0

#xrange(self.n_layers):
# the input to this layer is either the activation of the hidden
            # layer below or the input of the SdA if you are on the first
            # layer

 # end-snippet-2
            
            
        # We now need to add a logistic layer on top of the MLP 
        ###last layer
            
            self.params.extend(self.logLayer.params)
        # construct a function that implements one step of finetunining

        # compute the cost for second phase of training,
        # defined as the negative log likelihood
        self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)
        # compute the gradients with respect to the model parameters
        # symbolic variable that points to the number of errors made on the
        # minibatch given by self.x and self.y
        self.errors = self.logLayer.errors(self.y)
        

print'hidden_layers_units', hidden_layers_units
        # The SdA is an MLP, for which all weights of intermediate layers
        # are shared with a different denoising hoencoders
        # We will first construct the SdA as a deep multilayer perceptron,
        # and when constructing each sigmoidal layer we also construct a
        # denoising autoencoder that shares weights with that layer
        # During pretraining we will train these autoencoders (which will
        # lead to chainging the weights of the MLP as well)
        # During finetunining we will finish training the SdA by doing
        # stochastich gradient descent on the MLP

# construct the sigmoidal layer

            # the size of the input is either the number of hidden units of
            # the layer below or the input size if we are on the first layer


# add the layer to our list of layers
            self.sgd_layers_list.append(sgd_layer)
            # its arguably a philosophical question...
            # but we are going to only declare that the parameters of the
            # sgd_layers_list are parameters of the StackedDAA
            # the visible biases in the dA are parameters of those
            # dA, but not the SdA
            self.params.extend(sgd_layer.params)

            # Construct a denoising autoencoder that shared weights with this
            # layer
"""
 
"""
   main 1: 
   
Demonstrates how to train and test a stochastic denoising autoencoder.

This is demonstrated on MNIST.

:type learning_rate: float
:param learning_rate: learning rate used in the finetune stage
(factor for the stochastic gradient)

:type pretraining_epochs: int
:param pretraining_epochs: number of epoch to do pretraining

:type pretrain_lr: float
:param pretrain_lr: learning rate to be used during pre-training

:type n_iter: int
:param n_iter: maximal number of iterations to run the optimizer

:type dataset: string
:param dataset: path the the pickled dataset

"""

"""
 This tutorial introduces stacked denoising auto-encoders (SdA) using Theano.

 Denoising autoencoders are the building blocks for SdA.
 They are based on auto-encoders as the ones used in Bengio et al. 2007.
 An autoencoder takes an input x and first maps it to a hidden representation
 y = f_{\theta}(x) = s(Wx+b), parameterized by \theta={W,b}. The resulting
 latent representation y is then mapped back to a "reconstructed" vector
 z \in [0,1]^d in input space z = g_{\theta'}(y) = s(W'y + b').  The weight
 matrix W' can optionally be constrained such that W' = W^T, in which case
 the autoencoder is said to have tied weights. The network is trained such
 that to minimize the reconstruction error (the error between x and z).

 For the denosing autoencoder, during training, first x is corrupted into
 \tilde{x}, where \tilde{x} is a partially destroyed version of x by means
 of a stochastic mapping. Afterwards y is computed as before (using
 \tilde{x}), y = s(W\tilde{x} + b) and z as s(W'y + b'). The reconstruction
 error is now measured between z and the uncorrupted input x, which is
 computed as the cross-entropy :
      - \sum_{k=1}^d[ x_k \log z_k + (1-x_k) \log( 1-z_k)]


 References :
   - P. Vincent, H. Larochelle, Y. Bengio, P.A. Manzagol: Extracting and
   Composing Robust Features with Denoising Autoencoders, ICML'08, 1096-1103,
   2008
   - Y. Bengio, P. Lamblin, D. Popovici, H. Larochelle: Greedy Layer-Wise
   Training of Deep Networks, Advances in Neural Information Processing
   Systems 19, 2007

"""


"""Stacked denoising auto-encoder class (SdA)

A stacked denoising autoencoder model is obtained by stacking several
dAs. The hidden layer of the dA at layer `i` becomes the input of
the dA at layer `i+1`. The first layer dA gets as input the input of
the SdA, and the hidden layer of the last dA represents the output.
Note that after pretraining, the SdA is dealt with as a normal MLP,
the dAs are only used to initialize the weights.
"""
""" This class is made to support a variable number of layers.

    :type numpy_rng: numpy.random.RandomState
    :param numpy_rng: numpy random number generator used to draw initial
                weights

    :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
    :param theano_rng: Theano random generator; if None is given one is
                       generated based on a seed drawn from `rng`

    :type n_input_units: int
    :param n_input_units: dimension of the input to the sdA

    :type n_layers_sizes: list of ints
    :param n_layers_sizes: intermediate layers size, must contain
                           at least one value

    :type n_output_units: int
    :param n_output_units: dimension of the output of the network

    :type corruption_fraction: list of float
    :param corruption_fraction: amount of corruption to use for each
                              layer
"""
'''
def build_finetune_functions(self, datasets, batch_size, learning_rate):

Generates a function `train` that implements one step of
finetuning, a function `validate` that computes the error on
a batch from the validation set, and a function `test` that
computes the error on a batch from the testing set

:type datasets: list of pairs of theano.tensor.TensorType
:param datasets: It is a list that contain all the datasets;
                 the has to contain three pairs, `train`,
                 `valid`, `test` in this order, where each pair
                 is formed of two Theano variables, one for the
                 datapoints, the other for the labels

:type batch_size: int
:param batch_size: size of a minibatch

:type learning_rate: float
:param learning_rate: learning rate used during finetune stage
'''
'''
 Generates a list of functions, each of them implementing one
step in trainnig the dA corresponding to the layer with same index.
The function will require as input the minibatch index, and to train
a dA you just need to iterate, calling the corresponding function on
all minibatch indexes.

:type train_x: theano.tensor.TensorType
:param train_x: Shared variable that contains all datapoints used
                    for training the dA

:type batch_size: int
:param batch_size: size of a [mini]batch

:type learning_rate: float
:param learning_rate: learning rate used during training for any of
                      the dA layers
'''
 
 
@author:
"""
import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer
from dA import dA

def test_SdA(finetune_lr=0.1, pretraining_epochs=15,
             pretrain_lr=0.001, training_epochs=1000,
             dataset='mnist.pkl.gz', batch_size=1):
    """
    Demonstrates how to train and test a stochastic denoising autoencoder.

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used in the finetune stage
    (factor for the stochastic gradient)

    :type pretraining_epochs: int
    :param pretraining_epochs: number of epoch to do pretraining

    :type pretrain_lr: float
    :param pretrain_lr: learning rate to be used during pre-training

    :type n_iter: int
    :param n_iter: maximal number of iterations ot run the optimizer

    :type dataset: string
    :param dataset: path the the pickled dataset

    """

    datasets = load_data(dataset)
    
    n_factor = 5
    train_set_x, train_set_y = datasets[0] ## array([50000,   784], dtype=int64), array([50000], dtype=int64)
    train_set_x = train_set_x[0:(50000/n_factor),:]
    train_set_y = train_set_y[0:(50000/n_factor)]
    
    valid_set_x, valid_set_y = datasets[1]
    valid_set_x = valid_set_x[0:(10000/n_factor),:]
    valid_set_y = valid_set_y[0:(10000/n_factor)]
    
    test_set_x, test_set_y = datasets[2]
    test_set_x = test_set_x[0:(10000/n_factor),:]
    test_set_y = test_set_y[0:(10000/n_factor)]
    
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size

    

if __name__ == '__main__':
    test_SdA()
