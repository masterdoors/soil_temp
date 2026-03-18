"""
Implementation of the estimator wrapper to support customized base estimators.
"""


__all__ = ["KFoldWrapper"]

import copy
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble._forest import _generate_unsampled_indices, _get_n_samples_bootstrap, _generate_sample_indices
from types import SimpleNamespace

from gated_perceptron import ConvexGatedReLU 
from gated_perceptron import data_mvp, data_mvp_soft, data_mvp_expit, gradient_hb, gradient_hm, gradient_l2

from sklearn.linear_model import ridge_regression, Ridge

import nlopt
import time

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
    A1 = A[:,estimator.chain_mask[0]]

    sel = np.asarray([[i, A1.shape[1] + i] for i in range(A1.shape[1])]).flatten()
    res = np.hstack([A1, 1 - A1])[:,sel]     
    if len(estimator.chain_mask) > 1:
        A2 = A[:,estimator.chain_mask[1]]

        #sel = np.asarray([[i, A2.shape[1] + i] for i in range(A2.shape[1])]).flatten()
        sel = np.asarray([[i, A2.shape[1] + i] for i in range(A2.shape[1])]).flatten()
        res = np.concatenate([np.asarray(np.tile(res,(1,2))).reshape(-1,1,res.shape[1] * 2),np.asarray(np.hstack([A2, 1 - A2])[:,sel]).reshape(-1,1,res.shape[1] * 2)],axis=1).prod(axis=1)  
        #res = np.concatenate([np.asarray(res).reshape(-1,1,res.shape[1]),np.asarray(A2).reshape(-1,1,res.shape[1])],axis=1).prod(axis=1)          
    return res

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

def process_node(est,idx,d,avgs):
    if est.t.feature[idx] < 0:
        feat = np.random.choice(np.arange(avgs.shape[0]),1)
        trx = avgs[feat]
        est.t.feature[idx] = feat
        est.t.threshold[idx] = trx
        est.t.children_left[idx] = len(est.t.children_left)
        est.t.children_right[idx] = len(est.t.children_right) + 1
        est.t.children_left = np.append(est.t.children_left,[-1,-1])
        est.t.children_right = np.append(est.t.children_right,[-1,-1])
        est.t.feature = np.append(est.t.feature,[-1,-1])
        est.t.threshold = np.append(est.t.threshold,[-1,-1])                        
    if d < est.max_depth - 1:
        process_node(est,est.t.children_left[idx],d+1,avgs)    
        process_node(est,est.t.children_right[idx],d+1,avgs) 

     
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

        start_time = time.time()
        for k in range(n_classes):
            if len(y.shape) == 2 and y.shape[1] == 1:
                estimator.fit(X[train_batch_ids], y[train_batch_ids][:, k].flatten())
            else:                     
                estimator.fit(X[train_batch_ids], y[train_batch_ids][:, k])
            if k < n_classes - 1:    
                estimator.set_params(n_estimators=estimator.n_estimators + self.dummy_estimator_.n_estimators)
                
        #est_orig = copy.deepcopy(estimator)
        indexes = []
        trhxs = []    
        chain_masks = []

        print("RF time: ", time.time() - start_time)

        #uniques = np.where(np.asarray([len(np.unique(X[train_batch_ids][:,i])) for i in range(X.shape[1])]) > 1)
        avgs =  np.asarray(X[train_batch_ids].mean(axis=0)).flatten()
        node_offset = 0
        for est in estimator:
            sel = est.tree_.feature >= 0
            est.t = SimpleNamespace()
            est.t.feature = est.tree_.feature
            est.t.threshold = est.tree_.threshold
            est.t.children_left = est.tree_.children_left
            est.t.children_right = est.tree_.children_right

            if int(sel.sum()) < 2**est.max_depth - 1:
                root_idx = list(set(range(len(est.tree_.feature))) - set(est.tree_.children_left) - set(est.tree_.children_right))[0]
                process_node(est,root_idx,0,avgs)  
                sel = est.t.feature >= 0

            indexes.append(est.t.feature[sel])
            trhxs.append(est.t.threshold[sel])

            seconds = est.t.children_left
            second_mask = est.t.feature[seconds] >= 0 

            for k in range(len(seconds)):
                seconds[k] = seconds[k] - (second_mask[:seconds[k]] == False).sum()

            lefts = seconds[second_mask]

            seconds = est.t.children_right
            second_mask = est.t.feature[seconds] >= 0 

            for k in range(len(seconds)):
                seconds[k] = seconds[k] - (second_mask[:seconds[k]] == False).sum() + len(lefts)

            rights = seconds[second_mask]
            tops = np.asarray(list(set(list(range(len(indexes[len(indexes) - 1])))) - set(lefts) - set(rights)))

            tops += node_offset
            lefts += node_offset
            rights += node_offset
            if  est.max_depth > 1:
                chain_masks.append([tops,lefts, rights])
            else:
                chain_masks.append(tops) 

            node_offset = max(tops.max(),lefts.max(),rights.max()) + 1 #fix!

        est = SimpleNamespace()
        est._indexes = np.hstack(indexes)    
        est._trhxs = np.hstack(trhxs)  
        top_row = np.hstack([c[0] for c in chain_masks])
        est.chain_mask = [top_row,np.hstack([c[1] for c in chain_masks] + [c[2] for c in chain_masks])]                
    
        self.estimators_.append(est) 

        #if estimator.max_depth == 1:
        start_time = time.time()
        i1 = getIndicatorsLt(est, X[train_batch_ids])
        #i2 = getIndicators(est_orig, X[train_batch_ids], do_sample = False)
        print("Ind time: ", time.time() - start_time)
        I.append(i1)
        #else:    
        #    I.append(getIndicators(estimator, X[train_batch_ids], do_sample = False))

        I = np.hstack(I)
        
        # pos_idxs = np.where(I)
 
        # rng = np.random.default_rng()
        # to_empty = rng.choice(len(pos_idxs[0]), size=int(len(pos_idxs[0]) * 0.1), replace=False)
        # I[pos_idxs[0][to_empty],pos_idxs[1][to_empty]] = 0
        
        trains.append(train_batch_ids)
        tests.append(val_)

        G = np.random.default_rng().standard_normal((I.shape[1], self.hidden_size))
        #G = np.ones(G.shape)

        model = ConvexGatedReLU(G,c=1) #=n_classes)

        D = model.compute_activations(I)

        start_time = time.time()

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

        print("DS time: ", time.time() - start_time)    

        start_time = time.time()                
        if n_classes > 1:
            U = ridge_regression(M_,r_, sample_weight = sw,alpha = 0.000001,solver='sparse_cg',verbose=1).reshape(D.shape[1], I.shape[1])
        else:
            if sample_weight is None:
                U  = ridge_regression(M_,r[train_batch_ids], alpha = 0.00001,solver='sparse_cg').reshape(D.shape[1], I.shape[1])                
            else:    
                U  = ridge_regression(M_,r[train_batch_ids], alpha = 0.00001,solver='sparse_cg',sample_weight = sample_weight[train_batch_ids].flatten()).reshape(D.shape[1], I.shape[1])                

        print("Reg time: ", time.time() - start_time)
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

         
