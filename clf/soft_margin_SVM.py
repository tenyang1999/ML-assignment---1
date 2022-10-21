import numpy as np
import pandas as pd
import math

class soft_margin_SVM:
    
    def __init__(self, learning_rate=0.1,lambda_=0.01, n_iters=1000,slack = 0.01,C =1):
        self.lr = learning_rate
        self.lambda_ = lambda_
        self.n_iters= n_iters
        self.w = None
        self.b = None
        self.s = slack
        self.C = C
        
    def _init_weights_bias(self,x):
        input_shape = x.shape[1]
        self.w =  np.array([-1 for i in range (input_shape//2)]+[1 for i in range (round(input_shape/2))])
        self.b =  0
        

    def fit(self,x,y):
        self._init_weights_bias(x)
        
        for _ in range(self.n_iters):
            
            for i in range(x.shape[0]):
                
                label = y[i]   
                pred = np.dot(x[i], self.w)+ self.b
                if pred*label+ self.C*self.s < 0 :
                    dw = (self.lambda_*self.w) - (np.dot(label,x[i]))
                    db = -label
                else:
                    dw = self.lambda_*self.w 
                    db = 0
                
                self.w = self.w - self.lr*dw
                self.b = self.b - self.lr*db
                 
    def predict(self,x,y):
        count = 0
        for i in range(x.shape[0]):
            label = y[i]
            pred = np.sign(np.dot(x[i], self.w) + self.b)
            
            if pred == label:
                count += 1

        return count/x.shape[0]
                    
