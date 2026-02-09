from sklearn.metrics import accuracy_score
import numpy as np
from sklearn import preprocessing
from sklearn import datasets
from sklearn.model_selection import train_test_split
from boosted_forest import CascadeBoostingClassifier


digits = datasets.load_digits()

n_samples = len(digits.images)

data = digits.images.reshape((n_samples, -1))

Y =  np.asarray(digits.target).astype('int64')


x = preprocessing.normalize(data, copy=False, axis = 0)

x_train, x_validate, Y_train, Y_validate = train_test_split(
    x, Y, test_size=0.5, shuffle=True
)

print (np.unique(Y_train,return_counts=True))
print (np.unique(Y_validate,return_counts=True))

model = CascadeBoostingClassifier(loss = "multinomial", n_layers=10, n_estimators = 10, max_depth=1, n_iter_no_change = None, validation_fraction = 0.1, learning_rate = 0.1,hidden_size = 4,verbose=1, n_trees=4,batch_size = 1000)

model.fit(
    x_train,
    Y_train,
)        

y_pred = model.predict(x_validate) 
y_pred2 = model.predict(x_train)
mse_score = accuracy_score(Y_validate.flatten(),y_pred.flatten())
mae_score = accuracy_score(Y_validate.flatten(),y_pred.flatten())
print("Outer train error: ", accuracy_score(Y_train.flatten(),y_pred2.flatten()))


