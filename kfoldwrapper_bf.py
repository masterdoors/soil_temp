"""
Implementation of the estimator wrapper to support customized base estimators.
"""


__all__ = ["KFoldWrapper"]

import copy
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble._forest import _generate_unsampled_indices, _get_n_samples_bootstrap, _generate_sample_indices


from gated_perceptron import ConvexGatedReLU 
from gated_perceptron import data_mvp, gradient_hb, gradient_hm, gradient_l2


import cProfile
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


def profile(func):
    """Decorator for run function profile"""
    def wrapper(*args, **kwargs):
        profile_filename = func.__name__ + '.prof'
        profiler = cProfile.Profile()
        result = profiler.runcall(func, *args, **kwargs)
        profiler.dump_stats(profile_filename)
        return result
    return wrapper

def get_loss(best_res, best_v,I,r,D, gradient, data_mvp, raw_loss):
    def myfunc(v, grad):
        if grad.size > 0:
            grad[:] = gradient(v,I,r,D)

        res = raw_loss(r.flatten(),data_mvp(v, I, D).flatten())

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
        elif isinstance(loss,HalfBinomialLoss):   
            self.grad = gradient_hb
        elif isinstance(loss,HalfMultinomialLoss):  
            self.grad = gradient_hm
        else:
            raise TypeError("Unsupported loss")

    @property
    def estimator_(self):
        return self.estimators_

    #@profile
    def fit(self, X, y, r, sample_weight=None):
        splitter = KFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state,
        )

        trains = []
        tests = []
        for i, (train_idx, val_idx) in enumerate(splitter.split(X, y)):
            #print(i)
            estimator = copy.deepcopy(self.dummy_estimator_)

            # Fit on training samples
            if sample_weight is None:
                # Notice that a bunch of base estimators do not take
                # `sample_weight` as a valid input.
                estimator.fit(X[train_idx], y[train_idx].flatten())
            else:
                estimator.fit(
                    X[train_idx], y[train_idx].flatten(), sample_weight[train_idx]
                )
                
            indexes = []
            trhxs = []            
            for est in estimator:
                sel = est.tree_.feature >= 0
                indexes.append(est.tree_.feature[sel])
                trhxs.append(est.tree_.threshold[sel])

            estimator._indexes = np.hstack(indexes)
            estimator._trhxs = np.hstack(trhxs)                
            
            self.estimators_.append(estimator) 
            trains.append(train_idx)
            tests.append(val_idx)

            if estimator.max_depth == 1:
                I = getIndicatorsLt(estimator, X[train_idx])
            else:    
                I = getIndicators(estimator, X[train_idx], do_sample = False)

            G = np.random.default_rng().standard_normal((I.shape[1], self.hidden_size))
            model = ConvexGatedReLU(G)

            D = model.compute_activations(I)

            best_res = np.asarray([1e+31])
            x0 = np.random.rand(model.parameters[0].flatten().shape[0])
            best_v = np.zeros(x0.shape)

            myfunc = get_loss(best_res, best_v,I,r[train_idx],D, self.grad, data_mvp, self.loss)

            opt = nlopt.opt(nlopt.LD_TNEWTON, x0.shape[0])

            opt.set_min_objective(myfunc)

            opt.set_maxeval(500)
            opt.set_xtol_rel(0.01)
            try:
                opt.optimize(x0)
            except Exception as e:
                pass
            U = best_v.reshape(D.shape[1], I.shape[1]) 

            p1 = np.swapaxes(U,0,1)
            p2 = G
            self.nn_estimator_w.append(p1)
            self.nn_estimator_g.append(p2)             
        return trains, tests    

         
