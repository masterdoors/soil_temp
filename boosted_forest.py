'''
Created on Sep 16, 2023

@author: keen
'''

from sklearn.metrics import mean_squared_error
from sklearn.ensemble._gb import BaseGradientBoosting
from sklearn.dummy import DummyClassifier, DummyRegressor
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, issparse
from scipy.sparse import hstack
from sklearn.ensemble._gradient_boosting import _random_sample_mask
from sklearn.tree._tree import DOUBLE, DTYPE
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils._param_validation import HasMethods, Interval, StrOptions
from sklearn.utils.validation import _check_sample_weight
from sklearn.utils import check_array, check_random_state, column_or_1d
from sklearn.exceptions import NotFittedError
from sklearn.base import ClassifierMixin, RegressorMixin, is_classifier, _fit_context
from sklearn.ensemble._forest import _generate_unsampled_indices, _get_n_samples_bootstrap, _generate_sample_indices

from kfoldwrapper_bf import KFoldWrapper
from sklearn.ensemble import RandomForestRegressor
from numbers import Integral, Real
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import  accuracy_score
from time import time

from sklearn.ensemble import ExtraTreesRegressor
from _binner import Binner
from sklearn.model_selection import train_test_split

from torch.nn import BCEWithLogitsLoss
from torch.nn import MSELoss
from torch_mlp import MLPRB

import copy
import torch
from joblib import Parallel, delayed

from sklearn._loss.loss import (
    _LOSSES,
    AbsoluteError,
    ExponentialLoss,
    HalfBinomialLoss,
    HalfMultinomialLoss,
    HalfSquaredError,
    HuberLoss,
    PinballLoss,
)

def fitter(eid,estimator,restimator,n_splits,C,n_estimators,random_state,verbose,X_aug, residual,r,hidden_size,sample_weight):
    if eid %2 == 0:
        kfold_estimator = KFoldWrapper(
            estimator,
            n_splits,
            C,
            1. / n_estimators,
            hidden_size,
            random_state,
            verbose
        )
    else:
        kfold_estimator = KFoldWrapper(
            restimator,
            n_splits,
            C,
            1. / n_estimators,
            hidden_size,
            random_state,
            verbose
        )                       
        
    trains_, tests_ = kfold_estimator.fit(X_aug, residual,r,sample_weight)    
    return kfold_estimator, trains_, tests_ 

def _init_raw_predictions(X, estimator, loss, use_predict_proba):
    # TODO: Use loss.fit_intercept_only where appropriate instead of
    # DummyRegressor which is the default given by the `init` parameter,
    # see also _init_state.
    if len(X.shape) > 2:
        X_ = X.reshape(-1,X.shape[2])
    else:
        X_ = X
            
    if use_predict_proba:
        # Our parameter validation, set via _fit_context and _parameter_constraints
        # already guarantees that estimator has a predict_proba method.
        predictions = estimator.predict_proba(X_)
        if not loss.is_multiclass:
            predictions = predictions[:, 1]  # probability of positive class
        eps = np.finfo(np.float32).eps  # FIXME: This is quite large!
        predictions = np.clip(predictions, eps, 1 - eps, dtype=np.float64)
    else:
        predictions = estimator.predict(X_).astype(np.float64)

    if predictions.ndim == 1:
        return loss.link.link(predictions).reshape(-1, 1)
    else:
        return loss.link.link(predictions)

class VerboseReporter:
    """Reports verbose output to stdout.

    Parameters
    ----------
    verbose : int
        Verbosity level. If ``verbose==1`` output is printed once in a while
        (when iteration mod verbose_mod is zero).; if larger than 1 then output
        is printed for each update.
    """

    def __init__(self, verbose):
        self.verbose = verbose

    def init(self, est, begin_at_stage=0):
        """Initialize reporter

        Parameters
        ----------
        est : Estimator
            The estimator

        begin_at_stage : int, default=0
            stage at which to begin reporting
        """
        # header fields and line format str
        header_fields = ["Iter", "Train Loss"]
        verbose_fmt = ["{iter:>10d}", "{train_score:>16.4f}"]
        # do oob?
        if est.subsample < 1:
            header_fields.append("OOB Improve")
            verbose_fmt.append("{oob_impr:>16.4f}")
        header_fields.append("Time")
        verbose_fmt.append("{remaining_time:>16s}")

        # print the header line
        print(("%10s " + "%16s " * (len(header_fields) - 1)) % tuple(header_fields))

        self.verbose_fmt = " ".join(verbose_fmt)
        # plot verbose info each time i % verbose_mod == 0
        self.verbose_mod = 1
        self.start_time = time()
        self.begin_at_stage = begin_at_stage

    def update(self, j, est):
        """Update reporter with new iteration.

        Parameters
        ----------
        j : int
            The new iteration.
        est : Estimator
            The estimator.
        """
        do_oob = est.subsample < 1
        # we need to take into account if we fit additional estimators.
        i = j - self.begin_at_stage  # iteration relative to the start iter
        if (i + 1) % self.verbose_mod == 0:
            oob_impr = est.oob_improvement_[j] if do_oob else 0
            remaining_time = time() - self.start_time
            
            if remaining_time > 60:
                remaining_time = "{0:.2f}m".format(remaining_time / 60.0)
            else:
                remaining_time = "{0:.2f}s".format(remaining_time)
            print(
                self.verbose_fmt.format(
                    iter=j + 1,
                    train_score=est.train_score_[j],
                    oob_impr=oob_impr,
                    remaining_time=remaining_time,
                )
            )
            if self.verbose == 1 and ((i + 1) // (self.verbose_mod * 10) > 0):
                # adjust verbose frequency (powers of 10)
                self.verbose_mod *= 10        

class BaseBoostedCascade(BaseGradientBoosting):
    @_fit_context(
        # GradientBoosting*.init is not validated yet
        prefer_skip_nested_validation=False
    )
    def fit(self, X, y, sample_weight=None, monitor=None):
        """Fit the gradient boosting model.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        y : array-like of shape (n_samples,)
            Target values (strings or integers in classification, real numbers
            in regression)
            For classification, labels must correspond to classes.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. In the case of
            classification, splits are also ignored if they would result in any
            single class carrying a negative weight in either child node.

        monitor : callable, default=None
            The monitor is called after each iteration with the current
            iteration, a reference to the estimator and the local variables of
            ``_fit_stages`` as keyword arguments ``callable(i, self,
            locals())``. If the callable returns ``True`` the fitting procedure
            is stopped. The monitor can be used for various things such as
            computing held-out estimates, early stopping, model introspect, and
            snapshotting.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        if not self.warm_start:
            self._clear_state()

        # Check input
        # Since check_array converts both X and y to the same dtype, but the
        # trees use different types for X and y, checking them separately.

        X, y = self._validate_data(
            X, y, accept_sparse=["csr", "csc", "coo"], dtype=DTYPE, multi_output=True
        )
        sample_weight_is_none = sample_weight is None
        sample_weight = _check_sample_weight(sample_weight, X)
        if sample_weight_is_none:
            y = self._encode_y(y=y, sample_weight=None)
        else:
            y = self._encode_y(y=y, sample_weight=sample_weight)
        y = column_or_1d(y, warn=True)  # TODO: Is this still required?

        self._set_max_features()

        # self.loss is guaranteed to be a string
        self._loss = self._get_loss(sample_weight=sample_weight)

        # if self.n_iter_no_change is not None:
        #     stratify = y if is_classifier(self) else None
        #     (
        #         X_train,
        #         X_val,
        #         y_train,
        #         y_val,
        #         sample_weight_train,
        #         sample_weight_val,
        #     ) = train_test_split(
        #         X,
        #         y,
        #         sample_weight,
        #         random_state=self.random_state,
        #         test_size=self.validation_fraction,
        #         stratify=stratify,
        #     )
        #     if is_classifier(self):
        #         if self.n_classes_ != np.unique(y_train).shape[0]:
        #             # We choose to error here. The problem is that the init
        #             # estimator would be trained on y, which has some missing
        #             # classes now, so its predictions would not have the
        #             # correct shape.
        #             raise ValueError(
        #                 "The training data after the early stopping split "
        #                 "is missing some classes. Try using another random "
        #                 "seed."
        #             )
        # else:
        X_train, y_train, sample_weight_train = X, y, sample_weight
        X_val = y_val = sample_weight_val = None

        n_samples = X_train.shape[0]

        # First time calling fit.
        if not self._is_fitted():
            # init state
            self._init_state()

            # fit initial model and initialize raw predictions
            if self.init_ == "zero":
                raw_predictions = np.zeros(
                    shape=(n_samples, self.n_trees_per_iteration_),
                    dtype=np.float64,
                )
            else:
                # XXX clean this once we have a support_sample_weight tag
                if sample_weight_is_none:
                    self.init_.fit(X_train, y_train)
                else:
                    msg = (
                        "The initial estimator {} does not support sample "
                        "weights.".format(self.init_.__class__.__name__)
                    )
                    try:
                        self.init_.fit(
                            X_train, y_train, sample_weight=sample_weight_train
                        )
                    except TypeError as e:
                        if "unexpected keyword argument 'sample_weight'" in str(e):
                            # regular estimator without SW support
                            raise ValueError(msg) from e
                        else:  # regular estimator whose input checking failed
                            raise
                    except ValueError as e:
                        if (
                            "pass parameters to specific steps of "
                            "your pipeline using the "
                            "stepname__parameter" in str(e)
                        ):  # pipeline
                            raise ValueError(msg) from e
                        else:  # regular estimator whose input checking failed
                            raise

                raw_predictions = _init_raw_predictions(
                    X_train, self.init_, self._loss, is_classifier(self)
                )

            begin_at_stage = 0

            # The rng state must be preserved if warm_start is True
            self._rng = check_random_state(self.random_state)

        # warm start: this is not the first time fit was called
        else:
            # add more estimators to fitted model
            # invariant: warm_start = True
            if self.n_estimators < self.estimators_.shape[0]:
                raise ValueError(
                    "n_estimators=%d must be larger or equal to "
                    "estimators_.shape[0]=%d when "
                    "warm_start==True" % (self.n_estimators, self.estimators_.shape[0])
                )
            begin_at_stage = self.estimators_.shape[0]
            # The requirements of _raw_predict
            # are more constrained than fit. It accepts only CSR
            # matrices. Finite values have already been checked in _validate_data.
            X_train = check_array(
                X_train,
                dtype=DTYPE,
                order="C",
                accept_sparse="csr",
                force_all_finite=False,
            )
            raw_predictions = self._raw_predict(X_train)
            self._resize_state()

        # fit the boosting stages
        n_stages = self._fit_stages(
            X_train,
            y_train,
            raw_predictions,
            sample_weight_train,
            self._rng,
            X_val,
            y_val,
            sample_weight_val,
            begin_at_stage,
            monitor,
        )

        # change shape of arrays after fit (early-stopping or additional ests)
        if n_stages != len(self.estimators_):
            self.estimators_ = self.estimators_[:n_stages]
            self.train_score_ = self.train_score_[:n_stages]
            if hasattr(self, "oob_improvement_"):
                # OOB scores were computed
                self.oob_improvement_ = self.oob_improvement_[:n_stages]
                self.oob_scores_ = self.oob_scores_[:n_stages]
                self.oob_score_ = self.oob_scores_[-1]
        self.n_estimators_ = n_stages
        return self


    def _bin_data(self, binner, X, is_training_data=True):
        """
        Bin data X. If X is training data, the bin mapper is fitted first."""
        
        return X
        description = "training" if is_training_data else "testing"

        tic = time()
        if len(X.shape) > 2:
            X_ = X.reshape(-1,X.shape[2])
        else:
            X_ = X    
        
        if is_training_data:
            X_binned = binner.fit_transform(X_)
        else:
            X_binned = binner.transform(X_)
            X_binned = np.ascontiguousarray(X_binned)
        toc = time()
        binning_time = toc - tic

        if self.verbose > 1:
            msg = (
                "{} Binning {} data: {:.3f} MB => {:.3f} MB |"
                " Elapsed = {:.3f} s"
            )
            print(
                msg.format(
                    str(time()),
                    description,
                    X.nbytes / (1024 * 1024),
                    X_binned.nbytes / (1024 * 1024),
                    binning_time,
                )
            )
        if len(X.shape) > 2:
            X_binned = X_binned.reshape(X.shape) 
        return X_binned    
    
    def __init__(self,
        *,
        loss="log_loss",
        learning_rate=0.1,
        n_estimators=2,
        n_layers=3,
        subsample=1.0,
        criterion="friedman_mse",
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_depth=3,
        min_impurity_decrease=0.0,
        init=None,
        random_state=None,
        max_features=None,
        verbose=0,
        max_leaf_nodes=None,
        warm_start=False,
        validation_fraction=0.1,
        n_iter_no_change=None,
        tol=1e-4,
        ccp_alpha=0.0,
        C=1.0,
        n_splits=5,
        n_bins=255,
        bin_subsample=200000,
        bin_type="percentile",
        n_trees=100):
        super().__init__(loss = loss,
                         learning_rate = learning_rate,
                         n_estimators = n_layers,
                         subsample = subsample,
                         criterion = criterion,
                         min_samples_split = min_samples_split,
                         min_samples_leaf = min_samples_leaf,
                         min_weight_fraction_leaf = min_weight_fraction_leaf,
                         max_depth = max_depth,
                         min_impurity_decrease = min_impurity_decrease,
                         init = init,
                         random_state = random_state,
                         max_features = max_features,
                         verbose = verbose,
                         max_leaf_nodes = max_leaf_nodes,
                         warm_start = warm_start,
                         validation_fraction = validation_fraction,
                         n_iter_no_change = n_iter_no_change,
                         tol = tol,
                         ccp_alpha = ccp_alpha
                         )
        self.n_layers = n_layers
        self.n_estimators = n_estimators
        self.C = C
        self.n_splits = n_splits
        self.n_bins = n_bins
        self.bin_subsample = bin_subsample
        self.bin_type = bin_type
        self.binners = []
        self.device = "cpu"#torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.init = "zero"
        self.n_trees = n_trees
        
    def _init_state(self):
        """Initialize model state and allocate model state data structures."""

        self.init_ = self.init
        if self.init_ is None:
            if is_classifier(self):
                self.init_ = DummyClassifier(strategy="prior")
            elif isinstance(self._loss, (AbsoluteError, HuberLoss)):
                self.init_ = DummyRegressor(strategy="quantile", quantile=0.5)
            elif isinstance(self._loss, PinballLoss):
                self.init_ = DummyRegressor(strategy="quantile", quantile=self.alpha)
            else:
                self.init_ = DummyRegressor(strategy="mean")

        self.estimators_ = []
        self.train_score_ = np.zeros((self.n_layers,), dtype=np.float64)
        # do oob?
        if self.subsample < 1.0:
            self.oob_improvement_ = np.zeros((self.n_layers), dtype=np.float64)
            self.oob_scores_ = np.zeros((self.n_layers), dtype=np.float64)
            self.oob_score_ = np.nan           
    
    def _fit_stages(
        self,
        X,
        y,
        raw_predictions,
        sample_weight,
        random_state,
        X_val,
        y_val,
        sample_weight_val,
        begin_at_stage=0,
        monitor=None,
    ):
        """Iteratively fits the stages.

        For each stage it computes the progress (OOB, train score)
        and delegates to ``_fit_stage``.
        Returns the number of stages fit; might differ from ``n_estimators``
        due to early stopping.
        """
        self.lr = []
        binner_ = Binner(
            n_bins=self.n_bins,
            bin_subsample=self.bin_subsample,
            bin_type=self.bin_type,
            random_state=self.random_state,
        )  
        
        self.binners.append(binner_)      
        X_ = self._bin_data(binner_, X, is_training_data=True)
        
        n_samples = X.shape[0]
        do_oob = self.subsample < 1.0
        sample_mask = np.ones((n_samples,), dtype=bool)
        n_inbag = max(1, int(self.subsample * n_samples))
        loss_ = self._loss

        if self.verbose:
            verbose_reporter = VerboseReporter(verbose=self.verbose)
            verbose_reporter.init(self, begin_at_stage)

        X_csc = csc_matrix(X_) if issparse(X) else None
        X_csr = csr_matrix(X_) if issparse(X) else None

        if self.n_iter_no_change is not None:
            loss_history = np.full(self.n_iter_no_change, np.inf)
            # We create a generator to get the predictions for X_val after
            # the addition of each successive stage
            y_val_pred_iter = self._staged_raw_predict(X_val, check_input=False)

        history_sum = np.zeros((X_.shape[0],self.hidden_size))
        # perform boosting iterations
        i = begin_at_stage

        vobling = 0

        for i in range(begin_at_stage, self.n_layers):
            # subsampling
            if do_oob:
                sample_mask = _random_sample_mask(n_samples, n_inbag, random_state)
                if i == 0:  # store the initial loss to compute the OOB score
                    initial_loss = loss_(
                        y[~sample_mask],
                        raw_predictions[~sample_mask],
                        sample_weight[~sample_mask],
                    )

            history_sum_old = history_sum
            # fit next stage of trees
            raw_predictions, history_sum = self._fit_stage(
                i,
                X_,
                y,
                raw_predictions,
                history_sum,
                sample_weight,
                sample_mask,
                random_state,
                X_csc,
                X_csr
            )
            
            nrm = np.linalg.norm(history_sum_old - history_sum,axis=1)
            #print("Hidden diff: ", nrm.max(),nrm.min(),nrm.mean())
            # track loss
            if do_oob:
                self.train_score_[i] = loss_(
                    y[sample_mask],
                    raw_predictions[sample_mask],
                    sample_weight[sample_mask],
                )
                self.oob_scores_[i] = loss_(
                    y[~sample_mask],
                    raw_predictions[~sample_mask],
                    sample_weight[~sample_mask],
                )
                previous_loss = initial_loss if i == 0 else self.oob_scores_[i - 1]
                self.oob_improvement_[i] = previous_loss - self.oob_scores_[i]
                self.oob_score_ = self.oob_scores_[-1]
            else:
                # no need to fancy index w/ no subsampling
                if self._loss.n_classes == 2:
                    K = 1
                else:
                    K = self._loss.n_classes    
                self.train_score_[i] = loss_(y.flatten(), raw_predictions.flatten(), sample_weight)

            if self.verbose > 0:
                verbose_reporter.update(i, self)
                if self.loss in {"squared_error", "absolute_error", "huber", "quantile"}:
                    #print("Regressor loss: ", self.train_score_[i])
                    #pred_ = self._raw_predict(X)
                    #print("Regressor pred loss: ", loss_(y.flatten(), pred_.flatten(), sample_weight))
                    #print(y.flatten()[:5],pred_.flatten()[:5])
                    pass
                else:    
                    if self._loss.n_classes == 2:
                        encoded_classes = np.asarray(raw_predictions.reshape(y.shape) >= 0, dtype=int)
                    else:  
                        K = self._loss.n_classes    
                        encoded_classes = np.argmax(raw_predictions.reshape(y.shape + (K,)), axis=len(y.shape))
                    print("Acc: ",accuracy_score(encoded_classes.flatten(),y.flatten())) 

            if monitor is not None:
                if monitor(i, self, locals()):
                    break

            # We also provide an early stopping based on the score from
            # validation set (X_val, y_val), if n_iter_no_change is set
            if self.n_iter_no_change is not None and i > 0:
                if self.train_score_[i - 1] < self.train_score_[i]:
                    if vobling == self.n_iter_no_change:
                        break
                    else:
                        vobling += 1
                else:
                    vobling = 0        
          
        if vobling > 0:
            cut = max(1, i - vobling + 1)
            self.estimators_ = self.estimators_[:cut]
            self.lr = self.lr[:cut]
            self.n_layers = cut - 1
            return cut               
        else:
            self.n_layers = i
            return i + 1   
    
    def _fit_stage(
        self,
        i,
        X,
        y,
        raw_predictions,
        history_sum,
        sample_weight,
        sample_mask,
        random_state,
        X_csc=None,
        X_csr=None,
    ):
        """Fit another stage of ``_n_classes`` trees to the boosting model."""

        assert sample_mask.dtype == bool
        loss = self._loss
        original_y = y

        # Need to pass a copy of raw_predictions to negative_gradient()
        # because raw_predictions is partially updated at the end of the loop
        # in update_terminal_regions(), and gradients need to be evaluated at
        # iteration i - 1.
        raw_predictions_copy = raw_predictions.copy()
        
        estimator = RandomForestRegressor(
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            min_impurity_decrease=self.min_impurity_decrease,
            max_features=self.max_features,
            max_leaf_nodes=self.max_leaf_nodes,
            ccp_alpha=self.ccp_alpha,
            oob_score = False,
            bootstrap=True,
            n_estimators=self.n_trees,
            n_jobs = -1
        )  
        
        restimator = ExtraTreesRegressor(
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            min_impurity_decrease=self.min_impurity_decrease,
            max_features=self.max_features,
            max_leaf_nodes=self.max_leaf_nodes,
            ccp_alpha=self.ccp_alpha,
            oob_score = False,
            bootstrap=True,
            n_estimators=self.n_trees,
            n_jobs = -1
        )        

        residual = - loss.gradient(
            y.flatten(), raw_predictions_copy.flatten() 
        )
        
        if len(residual.shape) == 1:
            residual = residual.reshape(-1,1)
            
        binner_ = Binner(
            n_bins=self.n_bins,
            bin_subsample=self.bin_subsample,
            bin_type=self.bin_type,
            random_state=self.random_state,
        )  
        
        self.binners.append(binner_)      
        
        rp_old_bin = self._bin_data(binner_, history_sum, is_training_data=True)           
         
        #TODO convert y for classifier case
        #if loss.n_classes > 2:
        #    y = np.array(original_y == k, dtype=np.float64)

        # induce regression forest on residuals
        if self.subsample < 1.0:
            # no inplace multiplication!
            sample_weight = sample_weight * sample_mask.astype(np.float64)

        self.estimators_.append([])
        X = X_csr if X_csr is not None else X
        
        if isinstance(X,np.ndarray):
            X_aug = np.hstack([X,rp_old_bin])
        else:
            X_aug = hstack([X, csr_matrix(rp_old_bin)])               
        trains = []
        tests = []
        start_time = time()

        if i == 0:
            r = y
        else:    
            r = y.flatten() - raw_predictions.flatten()

        all_ze_staff = Parallel(n_jobs=1,backend="loky")(delayed(fitter)(eid,estimator,restimator,self.n_splits,self.C,self.n_estimators,self.random_state,self.verbose,X_aug, residual,r, history_sum.shape[1], sample_weight) for eid in range(self.n_estimators))
        # all_ze_staff = []
        # for eid in range(self.n_estimators):
        #     all_ze_staff.append(fitter(eid,estimator,restimator,self.n_splits,self.C,self.n_estimators,self.random_state,self.verbose,X_aug, residual,r, history_sum.shape[1], sample_weight))
        for kfold_estimator, trains_, tests_ in all_ze_staff: 
            trains += trains_
            tests += tests_
            self.estimators_[i].append(kfold_estimator)            

        end_time = time()
        execution_time = end_time - start_time      
        print("RF training time: ", execution_time) 
        init_values = None
        #if i > 0:
        #    init_values = copy.deepcopy(self.lr[i - 1].model.fc2.weight.detach().cpu()), copy.deepcopy(self.lr[i - 1].model.fc2.bias.detach().cpu())
        weight = 0.1 #pow(0.1, i)    
        raw_predictions, history_sum = self.update_terminal_regions(self.estimators_[i],trains, tests,X_aug, y,history_sum,sample_weight,init_values,weight)    
    
        return raw_predictions, history_sum  
    
    def getIndicatorsLt(self,estimator, X):
        W = X[:,estimator._indexes]    
        A = (W > estimator._trhxs[None,:]).astype(np.double)
        sel = np.asarray([[i, A.shape[1] + i] for i in range(A.shape[1])]).flatten()
        return np.hstack([A, 1 - A])[:,sel]     
    
    def getIndicators(self, estimator, X, sampled = True, do_sample = True):
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

    def update_terminal_regions(self,estimators, trains,tests,X, y, history_sum, sample_weight,init_values,weight):
        start_time = time()
    
        I = []
        for ek in estimators:        
            for e in ek.estimators_:
                if self.max_depth == 1:
                    I.append(self.getIndicatorsLt(e, X))
                else:    
                    I.append(self.getIndicators(e, X, do_sample = False))

        end_time = time()
        execution_time = end_time - start_time        
        print("Indicator building time: ", execution_time)        
        start_time = time()

        init_values = [[],[]]
        for ek in estimators:        
            init_values[0].append(ek.nn_estimator_w)            
            init_values[1].append(ek.nn_estimator_g)   
            #print([e.shape for e in ek.nn_estimator_w])   
            #print([e.shape for e in ek.nn_estimator_g])   

        init_values[0] = np.swapaxes(np.vstack(init_values[0]),0,1)
        init_values[1] = np.swapaxes(np.vstack(init_values[1]),0,1)

        cur_lr = self.lin_estimator(weight)
        cur_lr.mimic_fit(I,y,init_values)
        #cur_lr.fit(I, y, trains, tests, bias = history_sum, sample_weight = sample_weight,init_values = init_values)
        raw_predictions, hidden = cur_lr.decision_function(I,tests,history_sum)
        # oob_loss = self._loss(y.flatten(), raw_predictions.flatten(), sample_weight)
        # rp, _ = cur_lr.decision_function(I,None,history_sum)  
        # lrp = self._loss(y.flatten(), rp.flatten(), sample_weight)
        end_time = time()
        execution_time = end_time - start_time        
        # print("NN time: ", execution_time) 

        # for val_idx in tests:
        #     obb_loss_t = self._loss(y[val_idx].flatten(), raw_predictions[val_idx].flatten())
        #     print("OOB KV",obb_loss_t)

        # print("OOB res:",oob_loss)
        # print("Train res: ", lrp,mean_squared_error(y.flatten(),rp.flatten()))
        self.lr.append(cur_lr)
        return raw_predictions, hidden
    
    def _raw_predict_init(self, X):
        return np.zeros((X.shape[0],self.hidden_size))    
    
    def _raw_predict(self, X):
        """Return the sum of the trees raw predictions (+ init estimator)."""
        hidden = self._raw_predict_init(X)
        out = self.predict_stages(X,hidden)
        return out

    def _staged_raw_predict(self, X, check_input=True):
        if check_input:
            X = self._validate_data(
                X, dtype=DTYPE, order="C", accept_sparse="csr", reset=False
            )
        X = self._bin_data(self.binners[0], X, False)
        hidden = self._raw_predict_init(X)
        for i in range(self.n_layers):
            out, hidden = self.predict_stage(i, X, hidden)
            yield out.copy() 
            
    def predict_stage(self, i, X, hidden):
        rp = self._bin_data(self.binners[i + 1], hidden, False) #copy.deepcopy(raw_predictions) #
        if isinstance(X,np.ndarray):
            X_aug = np.hstack([X,rp])         
        else:
            X_aug = hstack([X,csr_matrix(rp)])          

        if self.estimators_[i] is not None:
            I = []
            for estimator in self.estimators_[i]:
                for e in estimator.estimators_:
                    if self.max_depth == 1:
                        I.append(self.getIndicatorsLt(e, X_aug)) 
                    else:
                        I.append(self.getIndicators(e, X_aug, do_sample = False)) 

            out, hidden = self.lr[i].decision_function(I,None,hidden) 

        return out, hidden    
    
    def predict_stages(self, X, hidden):
        X = self._bin_data(self.binners[0], X, False)        
        for i in range(len(self.estimators_)):
            out, hidden = self.predict_stage(i, X, hidden)
        return out    

# +
def getMLPRB(alpha,max_iter,tol,device,batch_size,learning_rate_init,
                                   hidden_size,
                                   n_splits,
                                   n_estimators,
                                   criterion,
                                   verbose):
    def get(weight):
        return MLPRB(alpha = alpha, 
                                   criterion = criterion,   
                                   max_iter=max_iter,
                                   tol = tol,
                                   device=device,
                                   batch_size=batch_size,
                                   learning_rate_init=learning_rate_init,
                                   hidden_size = hidden_size,
                                   n_splits=n_splits,
                                   n_estimators=n_estimators,
                                   weight = weight, 
                                   verbose = verbose)
    return get


class CascadeBoostingClassifier(ClassifierMixin, BaseBoostedCascade):
    _parameter_constraints: dict = {
        **BaseBoostedCascade._parameter_constraints,
        "loss": [StrOptions({"log_loss", "exponential"})],
        "init": [StrOptions({"zero"}), None, HasMethods(["fit", "predict_proba"])],
    }

    def __init__(
        self,
        *,
        loss="log_loss",
        learning_rate=0.1,
        n_estimators=2,
        n_layers=3,
        subsample=1.0,
        criterion="friedman_mse",
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_depth=3,
        min_impurity_decrease=0.0,
        C=1.0,
        init=None,
        random_state=None,
        max_features=None,
        verbose=0,
        max_leaf_nodes=None,
        warm_start=False,
        validation_fraction=0.1,
        n_iter_no_change=2,
        tol=1e-4,
        ccp_alpha=0.0,
        n_trees=100
    ):
        super().__init__(
            loss=loss,
            learning_rate=learning_rate,
            n_layers=n_layers,
            n_estimators=n_estimators,
            criterion=criterion,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_depth=max_depth,
            init=init,
            subsample=subsample,
            max_features=max_features,
            random_state=random_state,
            verbose=verbose,
            C=C,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            warm_start=warm_start,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            tol=tol,
            ccp_alpha=ccp_alpha,
            n_trees=n_trees
        )
        self.lin_estimator = getMLPRB(alpha = 1. / C, 
                                   max_iter=50,
                                   tol = 0.0000001,
                                   device=self.device,
                                   batch_size=64,
                                   learning_rate_init=0.001,
                                   hidden_size = self.hidden_size,
                                   n_splits=self.n_splits,
                                   n_estimators=self.n_estimators,
                                   criterion=BCEWithLogitsLoss,
                                   verbose = False)

    def _encode_y(self, y, sample_weight):
        # encode classes into 0 ... n_classes - 1 and sets attributes classes_
        # and n_trees_per_iteration_
        check_classification_targets(y)

        label_encoder = LabelEncoder()
        encoded_y_int = label_encoder.fit_transform(y)
        self.classes_ = label_encoder.classes_
        n_classes = self.classes_.shape[0]
        # only 1 tree for binary classification. For multiclass classification,
        # we build 1 tree per class.
        self.n_trees_per_iteration_ = 1 if n_classes <= 2 else n_classes
        encoded_y = encoded_y_int.astype(float, copy=False)

        # From here on, it is additional to the HGBT case.
        # expose n_classes_ attribute
        self.n_classes_ = n_classes
        if sample_weight is None:
            n_trim_classes = n_classes
        else:
            n_trim_classes = np.count_nonzero(np.bincount(encoded_y_int, sample_weight))

        if n_trim_classes < 2:
            raise ValueError(
                "y contains %d class after sample_weight "
                "trimmed classes with zero weights, while a "
                "minimum of 2 classes are required." % n_trim_classes
            )
        return encoded_y
    
    def _get_loss(self, sample_weight):
        if self.loss == "log_loss":
            if self.n_classes_ == 2:
                return HalfBinomialLoss(sample_weight=sample_weight)
            else:
                return HalfMultinomialLoss(
                    sample_weight=sample_weight, n_classes=self.n_classes_
                )
        elif self.loss == "exponential":
            if self.n_classes_ > 2:
                raise ValueError(
                    f"loss='{self.loss}' is only suitable for a binary classification "
                    f"problem, you have n_classes={self.n_classes_}. "
                    "Please use loss='log_loss' instead."
                )
            else:
                return ExponentialLoss(sample_weight=sample_weight) 
            
    def _validate_y(self, y, sample_weight):
        check_classification_targets(y)
        self.classes_, y = np.unique(y, return_inverse=True)
        n_trim_classes = np.count_nonzero(np.bincount(y, sample_weight))
        if n_trim_classes < 2:
            raise ValueError(
                "y contains %d class after sample_weight "
                "trimmed classes with zero weights, while a "
                "minimum of 2 classes are required." % n_trim_classes
            )
        self._n_classes = len(self.classes_)
        # expose n_classes_ attribute
        self.n_classes_ = self._n_classes
        return y
    
    def decision_function(self, X):
        X = self._validate_data(
            X, dtype=DTYPE, order="C", accept_sparse="csr", reset=False
        )
        raw_predictions = self._raw_predict(X)
        if raw_predictions.shape[1] == 1:
            return raw_predictions.ravel()
        return raw_predictions

    def staged_decision_function(self, X):
        yield from self._staged_raw_predict(X)

    def predict(self, X):
        raw_predictions = self.decision_function(X)
        if raw_predictions.ndim == 1:  # decision_function already squeezed it
            encoded_classes = (raw_predictions >= 0).astype(int)
        else:
            encoded_classes = np.argmax(raw_predictions, axis=1)
            
        return self.classes_[encoded_classes]


    def staged_predict(self, X):
        for raw_predictions in self._staged_raw_predict(X):
            encoded_labels = self._loss._raw_prediction_to_decision(raw_predictions)
            yield self.classes_.take(encoded_labels, axis=0)

    def predict_proba(self, X):
        raw_predictions = self.decision_function(X)
        try:
            return self._loss._raw_prediction_to_proba(raw_predictions)
        except NotFittedError:
            raise
        except AttributeError as e:
            raise AttributeError(
                "loss=%r does not support predict_proba" % self.loss
            ) from e

    def predict_log_proba(self, X):
        proba = self.predict_proba(X)
        return np.log(proba)

    def staged_predict_proba(self, X):
        try:
            for raw_predictions in self._staged_raw_predict(X):
                yield self._loss._raw_prediction_to_proba(raw_predictions)
        except NotFittedError:
            raise
        except AttributeError as e:
            raise AttributeError(
                "loss=%r does not support predict_proba" % self.loss
            ) from e


# -

class CascadeBoostingRegressor(RegressorMixin, BaseBoostedCascade):
    _parameter_constraints: dict = {
        **BaseBoostedCascade._parameter_constraints,
        "loss": [StrOptions({"squared_error", "absolute_error", "huber", "quantile"})],
        "init": [StrOptions({"zero"}), None, HasMethods(["fit", "predict"])],
        "alpha": [Interval(Real, 0.0, 1.0, closed="neither")],
    }

    def __init__(
        self,
        *,
        loss="squared_error",
        learning_rate=0.1,
        n_estimators=2,
        n_layers=3,
        subsample=1.0,        
        criterion="friedman_mse",
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_depth=3,
        min_impurity_decrease=0.0,
        init=None,
        C = 1.0,
        random_state=None,
        max_features=None,
        alpha=0.9,
        verbose=0,
        max_leaf_nodes=None,
        warm_start=False,
        validation_fraction=0.1,
        n_iter_no_change=2,
        tol=1e-4,
        ccp_alpha=0.0,
        hidden_size = 10,
        n_trees = 100,
        batch_size = 64,
    ):
        super().__init__(
            loss=loss,
            learning_rate=learning_rate,
            n_layers=n_layers,            
            n_estimators=n_estimators,
            criterion=criterion,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_depth=max_depth,
            init=init,
            subsample=subsample,
            max_features=max_features,
            min_impurity_decrease=min_impurity_decrease,
            random_state=random_state,
            C = C,
            #alpha=alpha,
            verbose=verbose,
            max_leaf_nodes=max_leaf_nodes,
            warm_start=warm_start,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            tol=tol,
            ccp_alpha=ccp_alpha,
            n_trees=n_trees
        )
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.lin_estimator = getMLPRB(alpha = 1. / C, 
                                   criterion = MSELoss(),   
                                   max_iter=200,
                                   tol = 0.0000001,
                                   device=self.device,
                                   batch_size=batch_size,
                                   learning_rate_init=0.01,
                                   hidden_size = self.hidden_size,
                                   n_splits=self.n_splits,
                                   n_estimators=self.n_estimators,
                                   verbose = False)

    def _encode_y(self, y=None, sample_weight=None):
        # Just convert y to the expected dtype
        self.n_trees_per_iteration_ = 1
        y = y.astype(DOUBLE, copy=False)
        return y

    def _get_loss(self, sample_weight):
        if self.loss in ("quantile", "huber"):
            l =  _LOSSES[self.loss](sample_weight=sample_weight, quantile=self.alpha)
        else:
            l =  _LOSSES[self.loss](sample_weight=sample_weight)
        l.n_classes = 1
        return l    

    def _validate_y(self, y, sample_weight=None):
        if y.dtype.kind == "O":
            y = y.astype(DOUBLE)
        return y
    
    def predict(self, X):
        X = self._validate_data(
            X, dtype=DTYPE, order="C", accept_sparse="csr", reset=False
        )
        # In regression we can directly return the raw value from the trees.
        return self._raw_predict(X).ravel()

    def staged_predict(self, X):
        for raw_predictions in self._staged_raw_predict(X):
            yield raw_predictions.ravel()

    def apply(self, X):
        leaves = super().apply(X)
        leaves = leaves.reshape(X.shape[0], self.estimators_.shape[0])
        return leaves   
