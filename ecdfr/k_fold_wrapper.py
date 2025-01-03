import imp
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,GradientBoostingRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
import numpy as np
from sklearn.model_selection import KFold,RepeatedKFold
from sklearn.metrics import mean_squared_error,r2_score
from ecdfr.function import adjust_sample
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from sklearn.ensemble import RandomForestRegressor
from joblib import Parallel, delayed


def kfold_worker(wrapper,cv,x,y,k,y_valid,error_threshold,resampling_rate):
	est=wrapper._init_estimator()
	train_id, val_id=cv[k]
	x_train,y_train=x[train_id],y[train_id]
	if y_valid is not None:
		x_train,y_train=adjust_sample(x_train,y_train,y_valid[train_id],error_threshold,resampling_rate)
	est.fit(x_train,y_train)
	y_pred=est.predict(x[val_id])
	return est, y_pred, val_id
	


class KFoldWapper(object):
    def __init__(self,layer_id,index,config,random_state):
        self.config=config
        self.name="layer_{}, estimstor_{}, {}".format(layer_id,index,self.config["type"])
        if random_state is not None:
            self.random_state=(random_state+hash(self.name))%1000000007
        else:
            self.random_state=None
        self.n_fold=self.config["n_fold"]
        self.estimators=[]
        self.config.pop("n_fold")
        self.estimator_class=RandomForestRegressor#globals()[self.config["type"]]
        self.config.pop("type")
    
    def _init_estimator(self):
        estimator_args=self.config
        est_args=estimator_args.copy()
        est_args["random_state"]=self.random_state
        return self.estimator_class(**est_args)
    
    def fit(self,x,y,y_valid,error_threshold,resampling_rate):
        kf=RepeatedKFold(n_splits=self.n_fold,n_repeats=1,random_state=self.random_state)
        cv=[(t,v) for (t,v) in kf.split(x)]
        y_train_pred=np.zeros((x.shape[0],))
        
        all_ze_staff = Parallel(n_jobs=self.n_fold,backend="loky")(delayed(kfold_worker)(self,cv,x,y,k,y_valid,error_threshold,resampling_rate) for k in range(self.n_fold))
        for est, y_pred, val_id in all_ze_staff:        
            self.estimators.append(est)
            y_train_pred[val_id] = y_pred
        return y_train_pred

    def predict(self,x):
        pre_value=0
        for est in self.estimators:
            pre_value+=est.predict(x)
        pre_value/=len(self.estimators)
        return pre_value