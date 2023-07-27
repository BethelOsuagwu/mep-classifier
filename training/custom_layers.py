# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 11:43:29 2023

@author: Bethel
"""
import tensorflow as tf;
from tensorflow import keras;
from keras import layers;
from keras import backend as K;
# Classes
class Implode(layers.Layer):
    def __init__(self):
        """
        Remove the second axis by summing it out.

        Parameters
        ----------
        activation : function Keras activation, optional
            The default is None.

        Returns
        -------
        TF tensor with same shape as input
        
        Call parameter
        inputs : Tensor
            Tensor of shape (batch_size,1,features)

        """
        super().__init__(dynamic=True,name='peak2peak_layer');

        #self.activation=activation;
    def build(self,input_shape):
        num_time_steps=input_shape[1];
        num_features=input_shape[-1];
        self.W=self.add_weight(shape=(num_time_steps,num_features),
                               initializer="random_normal",
                               name='kernel',
                               trainable=True);
        self.b=self.add_weight(shape=(1,),
                               initializer="zeros",
                               name='bias',
                               trainable=True);
    def call(self,inputs):
        batch_size = K.shape(inputs)[0];
        a=tf.repeat( [self.W],repeats=batch_size,axis=0);
        inputs=tf.multiply(inputs,a)+self.b;
        return K.sum(inputs,axis=1,keepdims=True);