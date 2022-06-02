#!/usr/bin/env python
# coding: utf-8

# In[2]:


from keras.engine.topology import Layer
from keras import backend as K
from keras.initializers import RandomUniform
from keras import activations

class MDlayer(Layer):
    def __init__(self, units, activation=None, **kwargs):
        self.units = units   #Number of dendrites
        self.activation = activations.get(activation)
        #self.boxes = boxes
    
        super(MDlayer, self).__init__(**kwargs)

    def build(self, input_shape):
        
        self.Wmin = self.add_weight(name='Wmin', 
                                  shape=(self.units, input_shape[1]),
                                  initializer=RandomUniform(minval=-2, maxval=-0.1, seed=None),
                                  trainable=True)
        self.Wmax = self.add_weight(name='Wmax', 
                                  shape=(self.units, input_shape[1]),
                                  initializer=RandomUniform(minval=0.1, maxval=2, seed=None),
                                  trainable=True)

        #if(len(self.boxes)>0):
         #   print("Pesos!!")
        
            
            
        super(MDlayer, self).build(input_shape) 

    def call(self, x):
        Q = K.int_shape(x)[0]
        if Q is None: Q = 1
        X = K.repeat(x,self.units)
        Wmin = K.permute_dimensions(K.repeat(self.Wmin, Q), (1,0,2))
        L1 = K.min(X - Wmin, axis=2)
        Wmax = K.permute_dimensions(K.repeat(self.Wmax, Q), (1,0,2))
        L2 = K.min(Wmax - X, axis=2)
        output = K.minimum(L1,L2)
        if self.activation is not None:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0],self.units) 

    def get_config(self):
        config = super(MDlayer, self).get_config()
        config.update({"units": self.units,
                       "activation":self.activation})
        return config
#    def get_config(self):

#        config = super().get_config().copy()
#        config.update({
#            'Nd' : self.Nd,
#            'activation' : self.activation,
#        })
#        return config

