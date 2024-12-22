#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
from os.path import join

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


def printf(*args, fname="log.txt"):
    with open(join("test_outputs",fname),"a+") as f:
        for a in args:
            f.write(str(a) + " ")
        f.write("\n") 
    print(args) 


# # Load Reynolds

# ## load daily soil temperature


import os
fdir = "data/Reynolds/soiltemperature"

def loadReynolds(fname):
    df = pd.read_csv(fname, delim_whitespace=True,comment="#", encoding="ISO-8859-1",on_bad_lines="warn",header=None)
    df = df[df[3]==17]
    df_time = pd.to_datetime(df[0].astype(str) +  ' ' + df[1].astype(str) + ' ' + df[2].astype(str),format="%m %d %Y")
    
    df2 = df[[5,6,7,8,9,10,11,12]].replace('.', np.nan).astype(float).ffill(axis=0)

    df2[1] = fname.replace(fdir,"").replace("/hourly","").replace("soiltemperature.txt","")
    
    return pd.concat([df_time,df2], axis=1) 

soil_temp = []
for file in os.listdir(fdir):
    if file.endswith(".txt") and file.find("hourly") > -1:
        print(file)
        path = os.path.join(fdir, file)
        soil_temp += [loadReynolds(path)]

reynolds_soil_temp = pd.concat(soil_temp, axis=0)


# In[ ]:


fdir2 = "data/Reynolds"

def loadReynoldsCL(fname):
    df = pd.read_csv(fname, delim_whitespace=True,comment="#", encoding="ISO-8859-1",on_bad_lines="warn",header=None)
    df = df[(df[2] > 1984)|(df[2] == 1984) & (df[0] == 12) &(df[1] > 4)]
    df_time = pd.to_datetime(df[0].astype(str) +  ' ' + df[1].astype(str) + ' ' + df[2].astype(str),format="%m %d %Y")
    
    df2 = df[[3,4]].replace('.', np.nan).astype(float).ffill(axis=0)

    df2[1] = fname.replace(fdir2,"").replace("/daily","").replace("climate.txt","")
    
    return pd.concat([df_time,df2], axis=1) 

climate = []
for file in os.listdir(fdir2):
    if file.endswith(".txt") and file.find("daily") > -1:
        print(file)
        path = os.path.join(fdir2, file)
        climate += [loadReynoldsCL(path)]

reynolds_climate = pd.concat(climate, axis=0)


# In[ ]:


reynolds_soil_temp = reynolds_soil_temp.rename(columns={0:'DATE',1:"LOC",5:"y_1",6:"y_2",7:"y_3",8:"y_4",9:"y_5",10:"y_6",11:"y_7",12:"y_8",3: 20, 4: 21}) 
reynolds_climate = reynolds_climate.rename(columns={0:'DATE',1:"LOC"})
climate = reynolds_climate.set_index(['DATE','LOC']).rename(columns={3: "C1", 4: "C2"}) 
soil = reynolds_soil_temp.set_index(['DATE','LOC'])


# In[ ]:


all_reynolds_data = pd.merge(soil,climate,left_index=True, right_index=True)


# In[ ]:


all_reynolds_data.to_csv("all_reynolds_data.csv",sep=";")




# # UK

# In[ ]:


uk_data = "data/UK/catalogue.ceh.ac.uk/datastore/eidchub/399ed9b1-bf59-4d85-9832-ee4d29f49bfb/"
climate_soil = []

def loadUK(fname):
    df = pd.read_csv(fname, sep=",",comment="#", on_bad_lines="warn")

    df1 = df.iloc[:, :2]
    df2 = df.iloc[:, 2:]    
    df2 = df2.astype(float).replace(-9999.0, np.nan).ffill(axis=0)    
    
    return pd.concat([df1,df2], axis=1)
    

for file in os.listdir(uk_data):
    if file.endswith(".csv") and file.find("daily") > -1 and file.find("flags") == -1 and file.find("metadata") == -1:
        print(file)
        path = os.path.join(uk_data, file)
        climate_soil += [loadUK(path)]

climate_soil =  pd.concat(climate_soil, axis=0)
climate_soil = climate_soil.rename(columns={"DATE_TIME":"DATE","SITE_ID":"LOC","TDT1_TSOIL":"y_1","TDT2_TSOIL":"y_2","TDT3_TSOIL":"y_3","TDT4_TSOIL":"y_4","TDT5_TSOIL":"y_5","TDT6_TSOIL":"y_6","TDT7_TSOIL":"y_7","TDT8_TSOIL":"y_8","TDT9_TSOIL":"y_9","TDT10_TSOIL":"y_10"}) 
climate_soil['DATE']= pd.to_datetime(climate_soil['DATE'])
climate_soil.set_index(["DATE","LOC"])


# In[ ]:


climate_soil.rename(columns={"DATE_TIME":"DATE","SITE_ID":"LOC","TDT1_TSOIL":"y_1","TDT2_TSOIL":"y_2","TDT3_TSOIL":"y_3","TDT4_TSOIL":"y_4","TDT5_TSOIL":"y_5","TDT6_TSOIL":"y_6","TDT7_TSOIL":"y_7","TDT8_TSOIL":"y_8","TDT9_TSOIL":"y_9","TDT10_TSOIL":"y_10"}) 
climate_soil = climate_soil.set_index(["DATE","LOC"])


# In[ ]:


climate_soil.to_csv("uk_soil.csv",sep=";")


climate_soil = climate_soil.drop(["SNOW_DEPTH", "TDT1_VWC","TDT2_VWC","TDT3_VWC","TDT4_VWC","TDT5_VWC","TDT6_VWC","TDT7_VWC","TDT8_VWC","TDT9_VWC","TDT10_VWC","PRECIP_TIPPING","PRECIP_RAINE"],axis=1)


# # Preprocessing of the Datasets

# ### Split onto 28 day long fragments. Remove fragments, which do not containt more than 30% of target values. 

# ### We do not consider temperatures for other soil layers as fatures deliberatly. 

# In[ ]:


from datetime import date, timedelta
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='mean')

PERIOD = 28
TRHX = 0.3

def daterange(start_date, end_date, step):
    for n in range(0,int((end_date - start_date).days), step):
        yield start_date + timedelta(n)

all_data = []
for df in [all_reynolds_data, climate_soil]:
    locations = df.index.get_level_values(1)
    
    rdatasX = {}
    rdatasY = {}
    for l in set(locations):
        ld = df.query("LOC == '" + l + "'").reset_index().set_index("DATE").drop("LOC",axis=1)
        all_start = ld.index.min()
        all_end = ld.index.max()
    
        for start in daterange(all_start, all_end, PERIOD):
            end = start + timedelta(PERIOD)
            period_data = ld.loc[start:end]
            
            y_columns = set([c for c in df.columns if c.find("y_") > -1])
            not_y_columns = [c for c in df.columns if c.find("y_") == -1]
            for y_counter in y_columns:
                if y_counter not in rdatasX: 
                    rdatasX[y_counter] = []
                    rdatasY[y_counter] = []
                    
                y_ = period_data[y_counter].to_numpy().astype(float)
                y = y_[1:]
    
                nans = np.count_nonzero(np.isnan(y))
    
                if float(nans) / y.shape[0] < TRHX and y.shape[0] == PERIOD:
                    X = period_data[not_y_columns].to_numpy()[:-1].astype(float)
                    X = np.hstack([X,y_[:-1].reshape(-1,1)]) # add current y as a feature
                    old_dim = X.shape[1]
                    X = imp.fit_transform(X)
                    if X.shape[1] == old_dim:
                        nans = np.count_nonzero(np.isnan(X))
                        if nans == 0:
                            y = imp.fit_transform(y.reshape(-1,1)).reshape(y.shape)
                            rdatasX[y_counter].append(X)
                            rdatasY[y_counter].append(y)
                    
    all_data.append({k:[np.asarray(rdatasX[k]), np.asarray(rdatasY[k])] for k in rdatasX})



from sklearn.preprocessing import normalize

for k in all_data[0]:
    all_data[0][k][0] = normalize(all_data[0][k][0].reshape(-1,all_data[0][k][0].shape[2]),axis=0).reshape(all_data[0][k][0].shape)
    
for k in all_data[1]:
    all_data[1][k][0] = normalize(all_data[1][k][0].reshape(-1,all_data[1][k][0].shape[2]),axis=0).reshape(all_data[1][k][0].shape)



from sklearn.model_selection import train_test_split
dict_data = {}
dict_data["Reynolds"] = {}
dict_data["UK"] = {}

for k in all_data[0]:
    x01,x02,y01,y02 = train_test_split(all_data[0][k][0], all_data[0][k][1], test_size=0.3,random_state=42)
    dict_data["Reynolds"][k] = {"train":{"X":x01,"y":y01},"test":{"X":x02,"y":y02}}
    
for k in all_data[1]:
    x11,x12,y11,y12 = train_test_split(all_data[1][k][0], all_data[1][k][1],test_size=0.3,random_state=42)
    dict_data["UK"][k] = {"train":{"X":x11,"y":y11},"test":{"X":x12,"y":y12}} 

all_data = dict_data


# # Experiments
# # Adaptive weighing (AWDF and ECDFR)

#ECDFR


# import xgboost as xgb
# from boosted_forest import CascadeBoostingRegressor
# from deepforest import CascadeForestRegressor
# from sklearn.metrics import mean_squared_error
# from sklearn.metrics import mean_absolute_error
# from sklearn.metrics import r2_score
# import optuna

# from ecdfr.gcForest import gcForest

# from sklearn.model_selection import KFold
# #xgb.set_config(verbosity=2)

# def make_modelECDFR(max_depth,layers,resampling_rate, et):
#     config = {"estimator_configs":[{"n_fold": 5,"type":None,"max_depth":max_depth},{"n_fold": 5,"type":None,"max_depth":max_depth},{"n_fold": 5,"type":None,"max_depth":max_depth},{"n_fold": 5,"type":None,"max_depth":max_depth}],
#               "error_threshold": et,
#               "resampling_rate": resampling_rate,
#               "random_state":None,
#               "max_layers":layers,
#               "early_stop_rounds":1,
#               "train_evaluation":r2_score}
    
#     return gcForest(config,2)

# models = {"ecdfr":make_modelECDFR}

# bo_data = []    
# work_pair = []
# best_pair = []

# max_score = 100


# for model_name in models:
#     make_model = models[model_name]
#     for ds_name in all_data:
#         for depth in all_data[ds_name]:
#             dat = all_data[ds_name][depth]
#             x_train = dat["train"]["X"].reshape(-1,dat["train"]["X"].shape[2])
#             x_test = dat["test"]["X"].reshape(-1,dat["test"]["X"].shape[2])
#             Y_train = dat["train"]["y"].flatten()
#             Y_test = dat["test"]["y"].flatten()            

#             def objective(trial):
#                 global max_score
#                 layers = trial.suggest_int('layers', 3, 15)
#                 max_depth = trial.suggest_int('max_depth', 1, 2)

#                 C = trial.suggest_float('resampling_rate', 0.1, 4)
#                 min_et = 0.5 * C - 1
#                 if min_et <= 0:
#                     min_et = 0.05
#                 else:
#                     if min_et > 0.99:
#                         min_et = 0.99

#                 max_et = C -1.

#                 if max_et > 0.99:
#                     max_et = 0.99
#                 else:
#                     if max_et <=0:
#                         max_et = 0.1

#                 if min_et >= max_et:
#                     max_et = min_et + 0.001
                    
#                 et = trial.suggest_float('et', 0.05, 0.95)
                
#                 kf = KFold(n_splits=3)
#                 scores = []
#                 try:
#                     for _, (train_index, test_index) in enumerate(kf.split(x_train)):
#                         model = make_modelECDFR(max_depth,layers,C,et)
                    
#                         model.fit(
#                              x_train[train_index],
#                              Y_train[train_index],
#                         )
#                         y_pred = model.predict(x_train[test_index]) #, batch_size=batch_size)
#                         scores.append(mean_squared_error(Y_train[test_index].flatten(),y_pred.flatten()))
#                         sc = mean_squared_error(Y_train[test_index].flatten(),y_pred.flatten())
#                         if max_score == 100:
#                             max_score = sc
#                         else:
#                             if sc > max_score:
#                                 max_score = sc
                                
#                         work_pair.append([C,et])
#                 except:
#                     scores = [max_score]
#                 return np.asarray(scores).mean() 
            
#             study = optuna.create_study(direction='minimize')
#             study.optimize(objective, n_trials=1000)    
            
#             layers = study.best_trial.params["layers"]  
#             max_depth = study.best_trial.params["max_depth"]  


#             C = study.best_trial.params["resampling_rate"]  
#             et = study.best_trial.params["et"]  

#             best_pair.append([C,et])
#             model = make_model(max_depth,layers,C,et)
#             model.fit(
#                  x_train,
#                  Y_train,
#             )        
            
#             y_pred = model.predict(x_test) #, batch_size=batch_size)
#             mse_score = mean_squared_error(Y_test.flatten(),y_pred.flatten())
#             mae_score = mean_absolute_error(Y_test.flatten(),y_pred.flatten())
#             printf(model_name,ds_name,depth,mse_score, mae_score, Y_test.min(),Y_test.max(),fname="ecdfr_output.txt")     
#             bo_data.append([model_name,ds_name,depth,mse_score, mae_score])
    


# # In[ ]:


# from matplotlib import pyplot as plt

# plt.scatter(*zip(*work_pair))

# plt.show()


# In[ ]:


# AWDF

from boosted_forest import CascadeBoostingRegressor
from deepforest import CascadeForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import optuna

from sklearn.model_selection import KFold
#xgb.set_config(verbosity=2)

def make_modelCascade(max_depth,layers,C,wt):
    wf = {0:"linear", 1:"1-w^1/2", 2:"1-w2"}
    return CascadeForestRegressor(max_depth = max_depth, max_layers = layers, n_estimators=4,adaptive=True,weighting_function = wf[wt],verbose=0,trx=1.0)


models = {"AWDF":make_modelCascade}

bo_data = []    

for model_name in models:
    make_model = models[model_name]
    for ds_name in all_data:
        for depth in all_data[ds_name]:
            dat = all_data[ds_name][depth]
            x_train = dat["train"]["X"].reshape(-1,dat["train"]["X"].shape[2])
            x_test = dat["test"]["X"].reshape(-1,dat["test"]["X"].shape[2])
            Y_train = dat["train"]["y"].flatten()
            Y_test = dat["test"]["y"].flatten()            

            def objective(trial):
                layers = trial.suggest_int('layers', 5, 15)
                max_depth = trial.suggest_int('max_depth', 1, 2)
                wt = trial.suggest_int('weight_function', 0, 2)   
                if model_name == "Boosted Forest":
                    C = trial.suggest_int('C', 1, 2000)
                else:
                    C = 0

                kf = KFold(n_splits=3)
                scores = []
                for _, (train_index, test_index) in enumerate(kf.split(x_train)):
                    model = make_model(max_depth,layers,C,wt)
                    
                    model.fit(
                         x_train[train_index],
                         Y_train[train_index],
                    )
                    y_pred = model.predict_sampled(x_train[test_index]) #, batch_size=batch_size)
                    scores.append(mean_squared_error(Y_train[test_index].flatten(),y_pred.flatten()))
                return np.asarray(scores).mean() 
            
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=500)    
            
            layers = study.best_trial.params["layers"]  
            max_depth = study.best_trial.params["max_depth"]  
            wt = study.best_trial.params['weight_function']
            if model_name == "Boosted Forest":
                C = study.best_trial.params["C"]  
            else:
                C = 0
            model = make_model(max_depth,layers,C,wt)
            model.fit(
                 x_train,
                 Y_train,
            )        
            
            y_pred = model.predict_sampled(x_test) #, batch_size=batch_size)
            mse_score = mean_squared_error(Y_test.flatten(),y_pred.flatten())
            mae_score = mean_absolute_error(Y_test.flatten(),y_pred.flatten())
            printf(model_name,ds_name,depth,mse_score, mae_score, Y_test.min(),Y_test.max(),fname="awdf_output.txt")     
            bo_data.append([model_name,ds_name,depth,mse_score, mae_score])
    