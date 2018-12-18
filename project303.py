# -*- coding: utf-8 -*-
"""
Created on Sat Dec 05 19:17:09 2015

@author: 

"""  
## numpy        
import numpy

## theano and modules
import theano
from theano.tensor.shared_randomstreams import RandomStreams
import theano.tensor as T

## from deeplearning.net
from loaddata import load_data, hidden_layer
from logistic_reg_sgd import logistic_regression 
from dA import dA

class SdA(object):
  
    def __init__(
        self,
        numpy_rng,
        theano_rng=None,
        n_input_units=784,
        hidden_layers_units=[500, 500],
        n_output_units=10,
        corruption_fraction=[0.2, 0.2]):
 
        self.params = []
        print'param', self.params
        print'hidden_layers_units', hidden_layers_units
        self.n_layers = len(hidden_layers_units)
       
        self.sgd_layers_list = []
        self.dA_layers_list = []
        
###############################################################################
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 20))
        
        self.x = T.matrix('x')  
        self.y = T.ivector('y')  
        print'hidden_layers_units', hidden_layers_units

        for i in xrange(self.n_layers) : #range(0, len(hidden_layers_units))  : 
            
            print '---------------------------- layer', i
            
            if i == 0:
                input_size = n_input_units
            else:
                input_size = hidden_layers_units[i - 1]
            print 'input layer size', input_size
            
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sgd_layers_list[-1].output

            sgd_layer = hidden_layer(rng=numpy_rng,
                                        n_in=input_size,
                                        input=layer_input,
                                        n_out=hidden_layers_units[i],
                                        activation=T.nnet.sigmoid)
                                        
            
            self.sgd_layers_list.append(sgd_layer)
            self.params.extend(sgd_layer.params)
            
###############################################################################
            dA_layer = dA(numpy_rng=numpy_rng,
                          theano_rng=theano_rng,
                          W=sgd_layer.W,
                          bhid=sgd_layer.b,
                          input=layer_input,
                          n_visible=input_size,
                          n_hidden=hidden_layers_units[i])
            self.dA_layers_list.append(dA_layer)
            
       
        self.logLayer = logistic_regression(
            input=self.sgd_layers_list[-1].output,
            n_in=hidden_layers_units[-1],
            n_out=n_output_units)

        self.params.extend(self.logLayer.params)
        self.finetuning_cost = self.logLayer.negative_log_likelihood(self.y)
        self.errors = self.logLayer.errors(self.y)


    def pretraining_functions(self, train_x, batch_size):
    
        index_minibatch = T.lscalar('index')  #  mini batch index
        corruption_level = T.scalar('corruption')  # corruption
        learning_rate = T.scalar('lr')  # LR
        
        batch_begin = index_minibatch * batch_size
        batch_end = batch_begin + batch_size

        pretrain_funct = []
        for dA in self.dA_layers_list:
###############################################################################            
            cost, updates = dA.get_cost_updates(corruption_level,
                                                learning_rate)
            # theano functions compilation
            fn = theano.function(
                inputs=[
                    index_minibatch,
                    theano.Param(corruption_level, default=0.2),
                    theano.Param(learning_rate, default=0.1)
                ],
                outputs=cost,
                updates=updates,
                givens={self.x: train_x[batch_begin: batch_end]})
            
            pretrain_funct.append(fn)

        return pretrain_funct

    def build_finetune_functions(self, datasets, batch_size, learning_rate):
        
        ## factor that decides what fraction of entire dataset is to be used        
        n_factor = 5000
        print 'data reduction factor' , n_factor 

        (train_x, train_y) = datasets[0]
        print 'print train_x.shape.eval()', train_x.shape.eval()
        (valid_x, valid_y) = datasets[1]
        print 'print valid_x.shape.eval()', valid_x.shape.eval()
        (test_x, test_y) = datasets[2]
        print 'print test_x.shape.eval()', test_x.shape.eval()

        # minibatches for training, validation, and testing
        n_valid_batches = valid_x.get_value(borrow=True).shape[0]
        print n_valid_batches
        n_valid_batches = n_valid_batches/n_factor
        print n_valid_batches
        n_valid_batches /= batch_size
        print n_valid_batches
        
        n_test_batches = test_x.get_value(borrow=True).shape[0]
        print n_test_batches
        n_test_batches = n_test_batches/n_factor
        print n_test_batches
        n_test_batches /= batch_size
        print 'testing batch', n_test_batches
###############################################################################

    #    train set  # array([50000,   784], dtype=int64), array([50000], dtype=int64)
        train_x = train_x[0:(50000/n_factor),:]
        print 'print train_x.shape.eval()', train_x.shape.eval()
        train_y = train_y[0:(50000/n_factor)]
        
    #   validation set 
        valid_x = valid_x[0:(10000/n_factor),:]
        print 'print valid_x.shape.eval()', valid_x.shape.eval()
        valid_y = valid_y[0:(10000/n_factor)]
    
    #    test set
        test_x = test_x[0:(10000/n_factor),:]
        print 'print test_x.shape.eval()', test_x.shape.eval()
        test_y = test_y[0:(10000/n_factor)]

###############################################################################
        
        index_minibatch = T.lscalar('index')  

        gparameters = T.grad(self.finetuning_cost, self.params)

        updates = [(param, param - gparameter * learning_rate)
            for param, gparameter in zip(self.params, gparameters)]

        train_function = theano.function(
            inputs=[index_minibatch],
            outputs=self.finetuning_cost,
            updates=updates,
            givens={
                self.x: train_x[index_minibatch * batch_size: (index_minibatch + 1) * batch_size],
                self.y: train_y[index_minibatch * batch_size: (index_minibatch + 1) * batch_size]
            },
            name='train')

        test_error_score_i = theano.function(
            [index_minibatch],
            self.errors,
            givens={
                self.x: test_x[index_minibatch * batch_size: (index_minibatch + 1) * batch_size],
                self.y: test_y[index_minibatch * batch_size: (index_minibatch + 1) * batch_size]
            },
            name='test')

        valid_error_score_i = theano.function(
            [index_minibatch],
            self.errors,
            givens={
                self.x: valid_x[index_minibatch * batch_size: (index_minibatch + 1) * batch_size],
                self.y: valid_y[index_minibatch * batch_size: (index_minibatch + 1) * batch_size]
            },
            name='valid')
        
        def valid_error_score():
            return [valid_error_score_i(i) for i in xrange(n_valid_batches)]

        def test_error_score():
            return [test_error_score_i(i) for i in xrange(n_test_batches)]

        return train_function, valid_error_score, test_error_score

def test_SdA(finetune_learning_rate=0.1, train_epochs=10,
             pretrain_epochs=10,
             pretrain_learning_rate=0.001,
             dataset='mnist.pkl.gz', batch_size=1):
   
    ##load the dataset and divide it into train, validation, and test
    datasets = load_data(dataset)

    train_x, train_y = datasets[0]
    valid_x, valid_y = datasets[1]
    test_x, test_y = datasets[2]
    
    ###########################################################################
#    datasets = load_data(dataset)
    
# compute number of minibatches for training, validation and testing
    n_train_batches = train_x.get_value(borrow=True).shape[0]  
    print n_train_batches  
    n_factor = 5000
    n_train_batches = n_train_batches/n_factor
    print n_train_batches 
    n_train_batches /= batch_size 
    print 'final batch size',n_train_batches 
    
#    train set ## array([50000,   784], dtype=int64), array([50000], dtype=int64)
    train_x = train_x[0:(50000/n_factor),:]
    print 'print train_x.shape.eval()', train_x.shape.eval()
    train_y = train_y[0:(50000/n_factor)]
    
#    validation set 
    valid_x = valid_x[0:(10000/n_factor),:]
    print 'print valid_x.shape.eval()', valid_x.shape.eval()
    valid_y = valid_y[0:(10000/n_factor)]
    
#    test set
    test_x = test_x[0:(10000/n_factor),:]
    print 'print test_x.shape.eval()', test_x.shape.eval()
    test_y = test_y[0:(10000/n_factor)]
    ###########################################################################
    
    # numpy RNG   
    numpy_rng = numpy.random.RandomState(77777)
    print 'building sdA model'
    
    print 'call sdA class'
    sda = SdA(
        numpy_rng=numpy_rng,
        hidden_layers_units=[1000, 1000, 1000],
        n_input_units=28 * 28,
        n_output_units=10)
  
    #####pretraining###########################################################
     
    print 'pretraining'
    pretraining_fns = sda.pretraining_functions(train_x=train_x, batch_size=batch_size)

    corruption_fraction = [.1, .2, .3]
    for i in xrange(sda.n_layers):
        
        for epoch in xrange(pretrain_epochs):
            
            cost_ = []
            for batch_index in xrange(n_train_batches):
                cost_.append(pretraining_fns[i](index=batch_index,
                         corruption=corruption_fraction[i],
                         lr=pretrain_learning_rate))
            print 'Pretraining layer %i, epoch %d, cost ' % (i, epoch),
            print numpy.mean(cost_)

    ##finetuning ##############################################################

    # get the training, validation and testing function for the model
    print 'getting the finetuning functions'
    train_function, validate_model, test_model = sda.build_finetune_functions(
        datasets=datasets,
        batch_size=batch_size,
        learning_rate=finetune_learning_rate)

    print 'finetunning'
    
###############################################################################    
    patience = 10 * n_train_batches  
    patience_increase = 2.  
    improvement_threshold = 0.994
    validation_frequency = min(n_train_batches, patience / 2)
                                  
    best_validation_loss = numpy.inf
    test_error_score = 0.

    done_looping = False
    epoch = 0

    while (epoch < train_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            
            iteration = (epoch - 1) * n_train_batches + minibatch_index

            if (iteration + 1) % validation_frequency == 0:
                validation_losses = validate_model()
                current_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, validation error %f %%' %
                      (epoch, current_validation_loss * 100.))

                if current_validation_loss < best_validation_loss:

                    if (current_validation_loss < best_validation_loss *
                        improvement_threshold):
                        patience = max(patience, iteration * patience_increase)
                        
                    best_validation_loss = current_validation_loss
                    best_iteration = iteration
                    test_losses = test_model()
                    test_error_score = numpy.mean(test_losses)
                    print(('-----> epoch %i, test error of ''best model %f %%') %
                          (epoch, test_error_score * 100.))

            if patience <= iteration:
                done_looping = True
                break
###############################################################################
    print(('best validation error %f %%, ' 'iterations =  %i, ''test error %f %%')
        % (best_validation_loss * 100., best_iteration + 1, test_error_score * 100.))

if __name__ == '__main__':
    test_SdA()
  
