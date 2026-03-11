import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import pickle
import os
from boosted_forest import CascadeBoostingClassifier

def load_cifar_batch(file):
    with open(file, 'rb') as f:
        data_dict = pickle.load(f, encoding='latin1')
        return data_dict['data'], np.array(data_dict['fine_labels'])

X_train, y_train = load_cifar_batch(os.path.join('cifar-100-python', 'train'))

X_test,y_test  = load_cifar_batch(os.path.join('cifar-100-python', 'test'))

X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

model = CascadeBoostingClassifier(loss = "multinomial", n_layers=100, n_estimators = 10, max_depth=1, n_iter_no_change = None, validation_fraction = 0.1, learning_rate = 0.1,hidden_size = 8,verbose=1, n_trees=1,batch_size = 500,max_features=0.5)
#model = GradientBoostingClassifier(loss = "log_loss", n_estimators=100,  max_depth=1, verbose = 1)
model.fit(X_train, y_train) 

accuracy = model.score(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")