from sklearn.neural_network._stochastic_optimizers import BaseOptimizer
import numpy as np
import copy

class RProp(BaseOptimizer):
    def __init__(
        self, params, learning_rate_init=0.01, etas=(0.5, 1.2), step_sizes=(1e-06, 0.1)
    ):
        super().__init__(learning_rate_init)
        self.eta_p = etas[0]
        self.eta_n = etas[1]
        self.step_sizes = step_sizes
        self.eta = [np.ones(p.shape) * self.learning_rate for p in params]
        self.g_prev = [np.zeros_like(p) for p in params]

    def _get_updates(self, grads):
        grads = copy.deepcopy(grads)

        for i in range(len(grads)):
            #prev_eta = copy.deepcopy(self.eta[i])
            sgn = self.g_prev[i] * grads[i]

            sgn[sgn > 0] = self.eta_p 
            sgn[sgn < 0] = self.eta_n  
            sgn[sgn == 0] = 1.

            self.eta[i] = np.clip(self.eta[i] * sgn,self.step_sizes[0],self.step_sizes[1])

            grads[i][grads[i] == self.eta_n] = 0.
            #print(i,"diff = ", np.asarray(self.eta[i] != prev_eta, dtype=int).sum())
        
        self.g_prev = copy.deepcopy(grads)

        return  [- e * np.sign(g) for e,g in zip(self.eta, grads)]                    
    

