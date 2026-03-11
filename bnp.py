import pandas
import numpy
from sklearn.ensemble import GradientBoostingClassifier
from boosted_forest import CascadeBoostingClassifier

from sklearn import metrics
from scipy import sparse

tbl=pandas.read_csv("BNP/train.csv.tar.gz",sep=',',compression = 'infer')

mtx = tbl.to_numpy()

x_mtx = mtx[:,2:]
y_mtx = mtx[:,1]

y = numpy.asarray(y_mtx,dtype=int) + 1

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import math
from scipy import sparse

arrs_to_conc = []

def my_func():
    for i in range(x_mtx.shape[1]):

        try:
            arr = numpy.unique(x_mtx[:,i].astype(float))
        except:
            arr = numpy.unique(x_mtx[:,i].astype(str))    

        if len(arr) < 40000:
            digitized_arr = LabelEncoder().fit_transform(x_mtx[:,i])
            if isinstance(arr[0],float) and math.isnan(arr[0]):
                nan_idx = digitized_arr == 0
                digitized_arr[nan_idx] = len(arr) * 2            
            coded_arr =  sparse.lil_matrix(OneHotEncoder(sparse_output=True,handle_unknown='ignore').fit_transform(digitized_arr.reshape(-1, 1)))

            arrs_to_conc.append(coded_arr)
            #print (i,coded_arr.shape)

        else:
            arrs_to_conc.append(sparse.lil_matrix(x_mtx[:,i].reshape(-1, 1),dtype=float))

    return sparse.hstack(arrs_to_conc)    

from sklearn.preprocessing import normalize
from sklearn.impute import SimpleImputer   

res_arr = my_func()

res_arr = SimpleImputer(strategy='most_frequent',copy=False).fit_transform(res_arr)
x = normalize(res_arr,axis=0)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y)

print(numpy.unique(y))

model = CascadeBoostingClassifier(loss = "binomial", n_layers=100, n_estimators = 10, max_depth=1, n_iter_no_change = None, validation_fraction = 0.1, learning_rate = 0.1,hidden_size = 10,verbose=1, n_trees=8,batch_size = 500,max_features=0.5)

model.fit(x_train,y_train)

pred = model.predict(x_test)
pred_train = model.predict(x_train)

print(metrics.f1_score(pred_train, y_train, average='macro'))
print(metrics.f1_score(pred, y_test, average='macro'))        