#!/usr/bin/env python
# coding: utf-8

# In[6]:


from keras.engine.topology import Layer
from keras import activations
from keras.initializers import RandomUniform
from keras import backend as K
import numpy as np
import tensorflow as tf

class DSNlayer(Layer):
    
    def __init__(self, spheres, activation = None, **kwargs):
        
        self.spheres = spheres
        self.activation = activations.get(activation)
        
        super(DSNlayer, self).__init__(**kwargs)

        
    def build(self, input_shape):
        
        self.centroid = self.add_weight(name='Centroids', 
                                      shape=(self.spheres,1,input_shape[1]), 
                                      initializer = RandomUniform(minval=-1, maxval=1, seed=None),
                                      trainable = True)
        #print("Centroid:",self.centroid)
        
        self.radius = self.add_weight(name = 'Radius', 
                                      shape = (self.spheres,1),
                                      initializer = RandomUniform(minval=-1, maxval=1, seed=None),
                                      trainable = True)
        #print("Radius:",self.radius)
        
        super(DSNlayer, self).build(input_shape) 

    
    def call(self, x):
        #print("1.-:",x)
        x = K.expand_dims(x, axis = 0)
        #print("2.-:",x)
        x = K.repeat_elements(x,int(np.shape(self.centroid)[0]),0)
        #print("3.-:",x)
        #print("Centroids:",self.centroid)
        dif = x-self.centroid
        #print("dif:",dif)
        distance = tf.norm(dif,axis=2)
        #distance = tf.reduce_sum(dif,axis=2)
        #print("distance:",distance)
        #print("Radius:",self.radius)
        
        
        output=self.radius-(distance)
        output = K.permute_dimensions(output, (1, 0))
        #print("output:",output)
        if self.activation is not None:
          output = self.activation(output)
        return output
    
    def compute_output_shape(self, input_shape):
        
        return (input_shape[0], int(np.shape(self.radius)[0]))
      
    def get_config(self):
        
        config = super().get_config().copy()
        config.update({
            'spheres' : self.spheres,
            'activation' : self.activation,
        })
        return config


# In[ ]:




