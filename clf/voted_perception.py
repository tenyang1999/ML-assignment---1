import numpy as np
import pandas as pd
import math


class voted_perception:
    
    def __init__(self,n_iters=1000):
        self.n_iters= n_iters
        self.w = None
        self.m = None
        self.c = None
        self.w_c = []  #store w and c
        
    def _init_weights_bias(self,x):
        input_shape = x.shape[1]
        self.w =  np.array([-1 for i in range (input_shape//2)]+[1 for i in range (round(input_shape/2))])
        self.m = 0
        self.c = 0
        
    def fit(self,x,y):
        self._init_weights_bias(x)
        for _ in range(self.n_iters):
            for i in range(x.shape[0]):
                    label = y[i]
                    pred = np.sign(np.dot(x[i], self.w))
                    if pred*label < 0 :
                        self.w_c =  self.w_c + [[self.w, self.c]]
                        self.w =  self.w + x[i]*label
                        self.m =  self.m + 1
                        self.c =  1
                    else:
                        self.c = self.c + 1
                        
        #print(self.w,'\n',self.b)
        
    def predict(self,x,y):
        count = 0
        for i in range(x.shape[0]):
            label = y[i]
            sum_pred = 0
            for m in range (self.m):
                c = self.w_c[m][1]
                w = self.w_c[m][0]
                sum_pred = sum_pred + c*np.sign(np.dot(x[i], w))
            pred = np.sign(sum_pred)    
            if pred == label:
                count += 1
        acc = count/x.shape[0]
        return  acc
    