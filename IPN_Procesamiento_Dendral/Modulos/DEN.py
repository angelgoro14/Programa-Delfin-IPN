#!/usr/bin/env python
# coding: utf-8

# In[4]:


from keras.engine.topology import Layer
from keras import backend as K
from keras.initializers import RandomUniform
from keras import activations
import numpy as np
import tensorflow as tf

class DENlayer(Layer):
    
    def __init__(self,ellipses, activation = None, *args , **kwargs):
               
        
        self.ellipses = ellipses
        self.activation = activations.get(activation)
        super(DENlayer, self).__init__(**kwargs)
       

            
    def build(self, input_shape):
        #print("build")  
        #print(input_shape)
        self.Centroids = self.add_weight(name='Centroids', 
                                     shape=(self.ellipses, 1, input_shape[1]), 
                                     initializer = RandomUniform(minval=-0.5, maxval=0.5, seed=None),
                                     trainable = True)
        
        self.Sigmas = self.add_weight(name = 'Sigmas', 
                                      shape = (self.ellipses, input_shape[1], input_shape[1]),
                                      initializer = RandomUniform(minval=-0.5, maxval=0.5, seed=None),
                                      trainable = True)
        
        
        super(DENlayer, self).build(input_shape) 

    
    def call(self, x):
        
        x = K.expand_dims(x, axis = 0)
        
        x = K.repeat_elements(x, int(np.shape(self.Centroids)[0]), 0)
        
        
        
        dif = x - self.Centroids
        
        difT = K.permute_dimensions(dif, (0, 2, 1))
                
        mah = K.batch_dot(dif, tf.linalg.inv(self.Sigmas))
        mah = K.batch_dot(mah, difT)

        #diag = tf.matrix_diag_part(mah)
        diag = tf.linalg.diag_part(mah)
        output = K.permute_dimensions(diag, (1, 0))

       
        if self.activation is not None:
            
           output = self.activation(output)

        return output

    
    def compute_output_shape(self, input_shape):
        
        return (input_shape[0], int(np.shape(self.Centroids)[0]))
      
    def get_config(self):
        
        config = super().get_config().copy()
        config.update({
            'ellipses' : self.ellipses,
            'activation' : self.activation,
        })
        return config

  
    


# In[ ]:




