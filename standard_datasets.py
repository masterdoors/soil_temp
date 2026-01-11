import warnings
warnings.filterwarnings("ignore")
import argparse
from os.path import join
import gc

def printf(*args, fname="log.txt"):
    with open(join("test_outputs",fname),"a+") as f:
        for a in args:
            f.write(str(a) + " ")
        f.write("\n") 
    print(args) 



import numpy as np

from sklearn import datasets

all_data = []



import pandas as pd
import numpy as np

df_train = pd.read_csv('cup98LRN.txt')
num_train = df_train.shape[0]
df_eval = pd.read_csv('cup98VAL.txt')
df_eval_target = pd.read_csv('lifetime-value/kdd_cup_98/valtargt.txt')
df_eval = df_eval.merge(df_eval_target, on='CONTROLN')

# Original: https://github.com/google/lifetime_value/blob/master/notebooks/kdd_cup_98/regression.ipynb
df = pd.concat([df_train, df_eval], axis=0, sort=True)
y = df['TARGET_D'][:num_train]

VOCAB_FEATURES = [
    'ODATEDW',  # date of donor's first gift (YYMM)
    'OSOURCE',  # donor acquisition mailing list
    'TCODE',    # donor title code
    'STATE',
    'ZIP',
    'DOMAIN',   # urbanicity level and socio-economic status of the neighborhood
    'CLUSTER',  # socio-economic status
    'GENDER',
    'MAXADATE', # date of the most recent promotion received
    'MINRDATE',
    'LASTDATE',
    'FISTDATE',
    'RFA_2A',
]

df['ODATEDW'] = df['ODATEDW'].astype('str')
df['TCODE'] = df['TCODE'].apply(
    lambda x: '{:03d}'.format(x // 1000 if x > 1000 else x))
df['ZIP'] = df['ZIP'].str.slice(0, 5)
df['MAXADATE'] = df['MAXADATE'].astype('str')
df['MINRDATE'] = df['MINRDATE'].astype('str')
df['LASTDATE'] = df['LASTDATE'].astype('str')
df['FISTDATE'] = df['FISTDATE'].astype('str')

def label_encoding(y, frequency_threshold=100):
  value_counts = pd.value_counts(y)
  categories = value_counts[
      value_counts >= frequency_threshold].index.to_numpy()
  # 0 indicates the unknown category.
  return pd.Categorical(y, categories=categories).codes + 1

for key in VOCAB_FEATURES:
  df[key] = label_encoding(df[key])

MAIL_ORDER_RESPONSES = [
    'MBCRAFT',
    'MBGARDEN',
    'MBBOOKS',
    'MBCOLECT',
    'MAGFAML',
    'MAGFEM',
    'MAGMALE',
    'PUBGARDN',
    'PUBCULIN',
    'PUBHLTH',
    'PUBDOITY',
    'PUBNEWFN',
    'PUBPHOTO',
    'PUBOPP',
    'RFA_2F',
]

INDICATOR_FEATURES = [
    'AGE',  # age decile, 0 indicates unknown
    'NUMCHLD',
    'INCOME',
    'WEALTH1',
    'HIT',
] + MAIL_ORDER_RESPONSES

df['AGE'] = pd.qcut(df['AGE'].values, 10).codes + 1
df['NUMCHLD'] = df['NUMCHLD'].apply(lambda x: 0 if np.isnan(x) else int(x))
df['INCOME'] = df['INCOME'].apply(lambda x: 0 if np.isnan(x) else int(x))
df['WEALTH1'] = df['WEALTH1'].apply(lambda x: 0 if np.isnan(x) else int(x) + 1)
df['HIT'] = pd.qcut(df['HIT'].values, q=50, duplicates='drop').codes

for col in MAIL_ORDER_RESPONSES:
  df[col] = pd.qcut(df[col].values, q=20, duplicates='drop').codes + 1

NUMERIC_FEATURES = [
    # binary
    'MAILCODE',  # bad address
    'NOEXCH',    # do not exchange
    'RECINHSE',  # donor has given to PVA's in house program
    'RECP3',     # donor has given to PVA's P3 program
    'RECPGVG',   # planned giving record
    'RECSWEEP',  # sweepstakes record
    'HOMEOWNR',  # home owner
    'CHILD03',
    'CHILD07',
    'CHILD12',
    'CHILD18',

    # continuous
    'CARDPROM',
    'NUMPROM',
    'CARDPM12',
    'NUMPRM12',
    'RAMNTALL',
    'NGIFTALL',
    'MINRAMNT',
    'MAXRAMNT',
    'LASTGIFT',
    'AVGGIFT',
]

df['MAILCODE'] = (df['MAILCODE'] == 'B').astype('float32')
df['PVASTATE'] = df['PVASTATE'].isin(['P', 'E']).astype('float32')
df['NOEXCH'] = df['NOEXCH'].isin(['X', '1']).astype('float32')
df['RECINHSE'] = (df['RECINHSE'] == 'X').astype('float32')
df['RECP3'] = (df['RECP3'] == 'X').astype('float32')
df['RECPGVG'] = (df['RECPGVG'] == 'X').astype('float32')
df['RECSWEEP'] = (df['RECSWEEP'] == 'X').astype('float32')
df['HOMEOWNR'] = (df['HOMEOWNR'] == 'H').astype('float32')
df['CHILD03'] = df['CHILD03'].isin(['M', 'F', 'B']).astype('float32')
df['CHILD07'] = df['CHILD07'].isin(['M', 'F', 'B']).astype('float32')
df['CHILD12'] = df['CHILD12'].isin(['M', 'F', 'B']).astype('float32')
df['CHILD18'] = df['CHILD18'].isin(['M', 'F', 'B']).astype('float32')

df['CARDPROM'] = df['CARDPROM'] / 100
df['NUMPROM'] = df['NUMPROM'] / 100
df['CARDPM12'] = df['CARDPM12'] / 100
df['NUMPRM12'] = df['NUMPRM12'] / 100
df['RAMNTALL'] = np.log1p(df['RAMNTALL'])
df['NGIFTALL'] = np.log1p(df['NGIFTALL'])
df['MINRAMNT'] = np.log1p(df['MINRAMNT'])
df['MAXRAMNT'] = np.log1p(df['MAXRAMNT'])
df['LASTGIFT'] = np.log1p(df['LASTGIFT'])
df['AVGGIFT'] = np.log1p(df['AVGGIFT'])

CATEGORICAL_FEATURES = VOCAB_FEATURES + INDICATOR_FEATURES
ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

def dnn_split(df):
  df_train = df.iloc[:num_train]
  df_eval = df.iloc[num_train:]

  def feature_dict(df):
    features = {k: v.values.reshape(-1,1) for k, v in dict(df[CATEGORICAL_FEATURES]).items()}
    features['numeric'] = df[NUMERIC_FEATURES].astype('float32').values
    return features

  x_train, y_train = feature_dict(df_train), df_train['TARGET_D'].astype(
      'float32').values
  x_eval, y_eval = feature_dict(df_eval), df_eval['TARGET_D'].astype(
      'float32').values

  return x_train, x_eval, y_train, y_eval

x_train, x_eval, y_train, y_eval = dnn_split(df)

all_data = [{0:[np.hstack(list(x_train.values())),y_train,np.hstack(list(x_eval.values())),y_eval]}]

diabetes = datasets.load_diabetes()
X, y = diabetes.data, diabetes.target

all_data.append({0:[X,y]})

from sklearn.datasets import fetch_california_housing

X, y = fetch_california_housing(return_X_y=True, as_frame=True)
all_data.append({0:[X.to_numpy(),y.to_numpy()]})

from sklearn.datasets import fetch_openml
ld = fetch_openml(name='liver-disorders')



X, y = ld['data'].to_numpy(),ld['target'].to_numpy()
all_data.append({0:[X,y]})



from sklearn.model_selection import train_test_split
dict_data = {}
dict_data["Diabetes"] = {}
dict_data["California housing"] = {}
dict_data["Liver disorders"] = {}
dict_data["KDD98"] = {}

for k in all_data[0]:
    x01,x02,y01,y02 = all_data[0][k][0], all_data[0][k][2], all_data[0][k][1], all_data[0][k][3]
    dict_data["KDD98"][k] = {"train":{"X":x01,"y":y01},"test":{"X":x02,"y":y02}}

for k in all_data[1]:
     x01,x02,y01,y02 = train_test_split(all_data[1][k][0], all_data[1][k][1], test_size=0.3,random_state=42)
     dict_data["Diabetes"][k] = {"train":{"X":x01,"y":y01},"test":{"X":x02,"y":y02}}

for k in all_data[2]:
    x11,x12,y11,y12 = train_test_split(all_data[2][k][0], all_data[2][k][1],test_size=0.3,random_state=42)
    dict_data["California housing"][k] = {"train":{"X":x11,"y":y11},"test":{"X":x12,"y":y12}} 

for k in all_data[3]:
    x11,x12,y11,y12 = train_test_split(all_data[3][k][0], all_data[3][k][1],test_size=0.3,random_state=42)
    dict_data["Liver disorders"][k] = {"train":{"X":x11,"y":y11},"test":{"X":x12,"y":y12}} 

all_data = dict_data

import xgboost as xgb
from boosted_forest import CascadeBoostingRegressor
from deepforest import CascadeForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import optuna

np.bool = np.bool_

from sklearn.model_selection import KFold
#xgb.set_config(verbosity=2)

def make_modelXGB(max_depth,layers,C,n_trees,n_estimators):
    return xgb.XGBRegressor(max_depth = max_depth, n_estimators = layers)

def make_modelCascade(max_depth,layers,C,n_trees,n_estimators):
    return CascadeForestRegressor(max_depth = max_depth, max_layers = layers, n_estimators=n_estimators,backend="sklearn",criterion='squared_error',n_trees=n_trees,n_tolerant_rounds = 100)

def make_modelBoosted(max_depth,layers,C,hs,n_trees,n_estimators):
    return CascadeBoostingRegressor(C=C, n_layers=layers, n_estimators = n_estimators, max_depth=max_depth, n_iter_no_change = None, validation_fraction = 0.1, learning_rate = 1.0,hidden_size = hs,verbose=1, n_trees=n_trees,batch_size = 256)

models = {"Boosted Forest": make_modelBoosted,"Cascade Forest": make_modelCascade,"XGB":make_modelXGB}

bo_data = []    

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help="Model (XGB, BOOSTED, CASCADE)")
parser.add_argument("--dataset", type=str, help="Model (Diabetes,California,Liver,KDD98)")

parser.add_argument("--layers", type=int)
parser.add_argument("--max_depth", type=int)
parser.add_argument("--C", type=float)
parser.add_argument("--hs", type=int)
parser.add_argument("--n_trees", type=int)

args = parser.parse_args()
model_ = args.model
dataset_ = args.dataset

if model_ == "XGB":
    model_name = "XGB"
elif model_ == "BOOSTED":
    model_name = "Boosted Forest"    
else:
    model_name = "Cascade Forest"    

if dataset_ == "California":
    ds_name = "California housing"
elif dataset_ == "Liver":
    ds_name = "Liver disorders"        
else:
    ds_name = dataset_

layers = int(args.layers)
max_depth = int(args.max_depth)
C = float(args.C)
hs = int(args.hs)
n_trees = int(args.n_trees)

n_est = 30

make_model = models[model_name]
for depth in all_data[ds_name]:
    dat = all_data[ds_name][depth]
    x_train = dat["train"]["X"]
    x_test = dat["test"]["X"]
    Y_train = dat["train"]["y"].flatten()
    Y_test = dat["test"]["y"].flatten()            

    for _ in range(3):
        if hs > 0:    
            model = make_model(max_depth,layers,C,hs,n_trees,n_est)
        else:
            model = make_model(max_depth,layers,C,n_trees,n_est)
            
        model.fit(
            x_train,
            Y_train,
        )        
        
        y_pred = model.predict(x_test) #, batch_size=batch_size)
        y_pred2 = model.predict(x_train)
        mse_score = mean_squared_error(Y_test.flatten(),y_pred.flatten())
        mae_score = mean_absolute_error(Y_test.flatten(),y_pred.flatten())
        print("Outer train error: ", mean_squared_error(Y_train.flatten(),y_pred2.flatten()))
        printf(model_name,ds_name,depth,max_depth,layers,C,hs,n_trees,n_est,mse_score, mae_score, Y_test.min(),Y_test.max(),fname="classic_datasets/boosting_output.txt")     

    
