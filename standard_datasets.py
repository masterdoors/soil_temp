#!/usr/bin/env python
# coding: utf-8

from os.path import join
def printf(*args, fname="log.txt"):
    with open(join("test_outputs",fname),"a+") as f:
        for a in args:
            f.write(str(a) + " ")
        f.write("\n") 
    print(args) 



import numpy as np

from sklearn import datasets

all_data = []
# # Diabetes

diabetes = datasets.load_diabetes()
X, y = diabetes.data, diabetes.target


all_data.append({0:[X,y]})


# # California housing

from sklearn.datasets import fetch_california_housing

X, y = fetch_california_housing(return_X_y=True, as_frame=True)
all_data.append({0:[X.to_numpy(),y.to_numpy()]})


# # Liver disorders


from sklearn.datasets import fetch_openml
ld = fetch_openml(name='liver-disorders')



X, y = ld['data'].to_numpy(),ld['target'].to_numpy()
all_data.append({0:[X,y]})



from sklearn.model_selection import train_test_split
dict_data = {}
dict_data["Diabetes"] = {}
dict_data["California housing"] = {}
dict_data["Liver disorders"] = {}
#dict_data["KDD98"] = {}

#for k in all_data[0]:
#    x01,x02,y01,y02 = all_data[0][k][0], all_data[0][k][2], all_data[0][k][1], all_data[0][k][3]
#    dict_data["KDD98"][k] = {"train":{"X":x01,"y":y01},"test":{"X":x02,"y":y02}}

for k in all_data[0]:
    x01,x02,y01,y02 = train_test_split(all_data[0][k][0], all_data[0][k][1], test_size=0.3,random_state=42)
    dict_data["Diabetes"][k] = {"train":{"X":x01,"y":y01},"test":{"X":x02,"y":y02}}
    
for k in all_data[1]:
    x11,x12,y11,y12 = train_test_split(all_data[1][k][0], all_data[1][k][1],test_size=0.3,random_state=42)
    dict_data["California housing"][k] = {"train":{"X":x11,"y":y11},"test":{"X":x12,"y":y12}} 

for k in all_data[2]:
    x11,x12,y11,y12 = train_test_split(all_data[2][k][0], all_data[2][k][1],test_size=0.3,random_state=42)
    dict_data["Liver disorders"][k] = {"train":{"X":x11,"y":y11},"test":{"X":x12,"y":y12}} 

all_data = dict_data


# # Boosting



import xgboost as xgb
from boosted_forest import CascadeBoostingRegressor
from deepforest import CascadeForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import optuna

np.bool = np.bool_

from sklearn.model_selection import KFold
#xgb.set_config(verbosity=2)

def make_modelXGB(max_depth,layers,C):
    return xgb.XGBRegressor(max_depth = max_depth, n_estimators = layers)

def make_modelCascade(max_depth,layers,C):
    return CascadeForestRegressor(max_depth = max_depth, max_layers = layers, n_estimators=4,backend="sklearn",criterion='squared_error')

def make_modelBoosted(max_depth,layers,C,hs):
    return CascadeBoostingRegressor(C=C, n_layers=layers, n_estimators = 4, max_depth=max_depth, n_iter_no_change = 1, validation_fraction = 0.1, learning_rate = 1.0,hidden_size = hs,verbose=1)


models = {"Boosted Forest": make_modelBoosted}

bo_data = []    

for model_name in models:
    make_model = models[model_name]
    for ds_name in all_data:
        for depth in all_data[ds_name]:
            dat = all_data[ds_name][depth]
            x_train = dat["train"]["X"]
            x_test = dat["test"]["X"]
            Y_train = dat["train"]["y"].flatten()
            Y_test = dat["test"]["y"].flatten()            

            layers = 5
            max_depth = 1

            C = 100
            hs = 10

            model = make_model(max_depth,layers,C,hs)
                
            model.fit(
                 x_train,
                 Y_train,
            )        
            
            y_pred = model.predict(x_test) #, batch_size=batch_size)
            mse_score = mean_squared_error(Y_test.flatten(),y_pred.flatten())
            mae_score = mean_absolute_error(Y_test.flatten(),y_pred.flatten())
            print(model_name,ds_name,depth,mse_score, mae_score, Y_test.min(),Y_test.max())     
    

