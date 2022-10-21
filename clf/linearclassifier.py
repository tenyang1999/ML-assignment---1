import numpy as np
import pandas as pd
import math

class linearclassifier:
    
    def __init__(self, n_iters=1000):
        self.n_iters= n_iters
        self.w = None
        
    def _init_weights_bias(self,x):
        input_shape = x.shape[1]
        self.w =  np.array([-1 for i in range (input_shape//2)]+[1 for i in range (round(input_shape/2))])
        
    def fit(self,x,y):
        self._init_weights_bias(x)
        for _ in range(self.n_iters):
            for i in range(x.shape[0]):
                    label = y[i]
                    pred = np.sign(np.dot(x[i], self.w))
                    if pred*label < 0 :
                         self.w =  self.w+ x[i]*label
        #print( self.w)
        
    def predict(self,x,y):
        count = 0
        for i in range(x.shape[0]):
            label = y[i]
            pred = np.sign(np.dot(x[i], self.w))
            if pred == label:
                count += 1
        acc = count/x.shape[0]
        return  acc
    