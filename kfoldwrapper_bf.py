"""
Implementation of the estimator wrapper to support customized base estimators.
"""


__all__ = ["KFoldWrapper"]

import copy
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble._forest import _generate_unsampled_indices, _get_n_samples_bootstrap, _generate_sample_indices


from gated_perceptron import ConvexGatedReLU 
from gated_perceptron import data_mvp, data_mvp_soft, data_mvp_expit, gradient_hb, gradient_hm, gradient_l2

from sklearn.linear_model import ridge_regression, Ridge

import nlopt

from sklearn._loss.loss import HalfBinomialLoss, HalfMultinomialLoss, HalfSquaredError

def to_shape(a, shape):
    y_, x_ = shape
    y, x = a.shape
    y_pad = (y_-y)
    x_pad = (x_-x)
    return np.pad(a,((y_pad//2, y_pad//2 + y_pad%2), 
                     (x_pad//2, x_pad//2 + x_pad%2)),
                  mode = 'constant')

def getIndicatorsLt(estimator, X):
    W = X[:,estimator._indexes]    
    A = (W > estimator._trhxs[None,:]).astype(np.double)
    sel = np.asarray([[i, A.shape[1] + i] for i in range(A.shape[1])]).flatten()
    return np.hstack([A, 1 - A])[:,sel]     

def getIndicators(estimator, X, sampled = True, do_sample = True):
    Is = []
    n_samples = X.shape[0]
    n_samples_bootstrap = _get_n_samples_bootstrap(
        n_samples,
        estimator.max_samples,
    )  
    idx = estimator.apply(X)
    for i,clf in enumerate(estimator.estimators_):
        if do_sample:
            if sampled:
                indices = _generate_sample_indices(
                    clf.random_state,
                    n_samples,
                    n_samples_bootstrap,
                )        
            else:    
                indices = _generate_unsampled_indices(
                    clf.random_state,
                    n_samples,
                    n_samples_bootstrap,
                )
        else:
            indices = list(range(X.shape[0]))                    
    
        depth = clf.get_depth()

        if depth < 1:
            I = np.zeros((X.shape[0], 3))
        else:  
            I = np.zeros((X.shape[0], 2**(depth + 1) - 1))

        for j in indices:
            I[j,idx[j,i]] = 1.0    
        Is.append(I)
    return np.hstack(Is)

def get_loss(best_res, best_v,I,r,D, gradient, data_mvp, raw_loss, bias):
    def myfunc(v, grad):
        if grad.size > 0:
            grad[:] = gradient(v,I,r,D, bias)

        res = raw_loss(r.flatten(),data_mvp(v, I, D, bias)) + 0.00001 * (v @ v).sum() * 0.5
        #print(res) 
        if res < best_res:
            best_res[:] = res
            best_v[:] = copy.deepcopy(v) 
        return res
    return myfunc

class KFoldWrapper(object):
    """
    A general wrapper for base estimators without the characteristic of
    out-of-bag (OOB) estimation.
    """

    def __init__(
        self,
        estimator_forest,
        n_splits,
        C=1.0,
        factor = 0.5,
        hidden_size = 1,
        random_state=None,
        loss = HalfSquaredError,
        verbose=1,
    ):
     
        # Parameters were already validated by upstream methods
        self.dummy_estimator_ = estimator_forest
        #self.dummy_lin = estimator_linear
        self.n_splits = n_splits
        self.random_state = random_state
        self.verbose = verbose
        # Internal container
        self.estimators_ = []
        self.C = C
        self.factor = factor 
        self.hidden_size = hidden_size
        self.nn_estimator_w = []
        self.nn_estimator_g = []
        self.loss = loss

        if isinstance(loss,HalfSquaredError):
            self.grad = gradient_l2
            self.mvp = data_mvp
        elif isinstance(loss,HalfBinomialLoss):   
            self.grad = gradient_hb
            self.mvp = data_mvp_expit
        elif isinstance(loss,HalfMultinomialLoss):  
            self.grad = gradient_hm
            self.mvp = data_mvp_soft
        else:
            raise TypeError("Unsupported loss")

    @property
    def estimator_(self):
        return self.estimators_

    def fit(self, X, y, r, bias, y_,sample_weight=None,second_reflection=None, train_batch_ids=None):
        splitter = KFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state,
        )

        trains = []
        tests = []

        if len(y.shape) > 1:
            n_classes = y.shape[1]
        else:
            n_classes = 1
        
        val_ = np.asarray(list(set(range(X.shape[0])) - set(train_batch_ids)))

        #print(i)
        I = []
        estimator = copy.deepcopy(self.dummy_estimator_)


        for k in range(n_classes):
            if len(y.shape) == 2 and y.shape[1] == 1:
                estimator.fit(X[train_batch_ids], y[train_batch_ids][:, k].flatten())
            else:                     
                estimator.fit(X[train_batch_ids], y[train_batch_ids][:, k])
            if k < n_classes - 1:    
                estimator.set_params(n_estimators=estimator.n_estimators + self.dummy_estimator_.n_estimators)
                
        indexes = []
        trhxs = []    

        #uniques = np.where(np.asarray([len(np.unique(X[train_batch_ids][:,i])) for i in range(X.shape[1])]) > 1)
        avgs =  X[train_batch_ids].mean(axis=0)
        for est in estimator:
            sel = est.tree_.feature >= 0
            if sel.sum() == 0:
                #temporary solution, works only if tree depth = 1
                feat = np.random.choice(np.arange(X.shape[1]),1)
                trx = avgs[feat]
                indexes.append(feat)
                trhxs.append(trx)
            else:    
                indexes.append(est.tree_.feature[sel])
                trhxs.append(est.tree_.threshold[sel])

        estimator._indexes = np.hstack(indexes)
        estimator._trhxs = np.hstack(trhxs)                
    
        self.estimators_.append(estimator) 

        if estimator.max_depth == 1:
            I.append(getIndicatorsLt(estimator, X[train_batch_ids]))
        else:    
            I.append(getIndicators(estimator, X[train_batch_ids], do_sample = False))

        I = np.hstack(I)
        
        trains.append(train_batch_ids)
        tests.append(val_)

        G = np.random.default_rng().standard_normal((I.shape[1], self.hidden_size))
        #G = np.ones(G.shape)

        model = ConvexGatedReLU(G,c=1) #=n_classes)

        D = model.compute_activations(I)

        if n_classes > 1:
            M = []
            r_ = []
            sw_ = []

            for k in range(n_classes):
                #
                D_ = np.multiply(D, second_reflection[:,k])
                M_ = np.transpose(np.einsum("ij, il->lji", I, D_).reshape(-1,I.shape[0]))
                sw_idx_ = sample_weight[train_batch_ids,k].flatten() > 0.001
                M.append(M_[sw_idx_])
                r_.append(r[train_batch_ids][:,k][sw_idx_])
                sw_.append(sample_weight[train_batch_ids,k].flatten()[sw_idx_])

            M_ = np.vstack(M)      
            r_ = np.hstack(r_)  
            sw = np.hstack(sw_)
        else:
            M_ = np.transpose(np.einsum("ij, il->lji", I , D).reshape(-1,I.shape[0]))

        if n_classes > 1:
            U = ridge_regression(M_,r_, sample_weight = sw,alpha = 0.000001,solver='sparse_cg',verbose=1).reshape(D.shape[1], I.shape[1])
        else:
            if sample_weight is None:
                U  = ridge_regression(M_,r[train_batch_ids], alpha = 0.00001,solver='sparse_cg').reshape(D.shape[1], I.shape[1])                
            else:    
                U  = ridge_regression(M_,r[train_batch_ids], alpha = 0.00001,solver='sparse_cg',sample_weight = sample_weight[train_batch_ids].flatten()).reshape(D.shape[1], I.shape[1])                

        
        # mkv = (M_ @ np.swapaxes(U,1,2)).T.reshape(-1,n_classes)
        # lt0 = self.loss(y_[train_idx].flatten(),mkv) 
        # lt = self.loss(y_[train_idx].flatten(),data_mvp(U, I, D, bias[train_idx]))  
        # #lt2 = self.loss(y_[train_idx].flatten(), ((I @ U.reshape(-1,I.shape[1]).T) * D).sum(axis=1) + bias[train_idx].flatten())  

        # if estimator.max_depth == 1:
        #     I = getIndicatorsLt(estimator, X[val_idx])
        # else:    
        #     I = getIndicators(estimator, X[val_idx], do_sample = False)    
        # D = model.compute_activations(I)                        
        # ltest = self.loss(y_[val_idx].flatten(),data_mvp(U, I, D, bias[val_idx]))  

        # #U = best_v.reshape(D.shape[1] * n_classes, I.shape[1]) 
        # print("KV: ",lt0,lt,ltest)            
        p1 = np.swapaxes(U.reshape(-1, I.shape[1]),0,1)
        p2 = G
        self.nn_estimator_w.append(p1)
        self.nn_estimator_g.append(p2)             
        return trains, tests    

         
