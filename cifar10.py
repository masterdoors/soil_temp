import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import pickle
import os
from boosted_forest import CascadeBoostingClassifier

def load_cifar_batch(file):
    with open(file, 'rb') as f:
        data_dict = pickle.load(f, encoding='latin1')
        return data_dict['data'], np.array(data_dict['labels'])

batches = ['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5']

bs = [load_cifar_batch(os.path.join('cifar-10-batches-py', b)) for b in batches]
X_train = np.vstack([b[0] for b in bs])
y_train = np.hstack([b[1] for b in bs])
X_test, y_test = load_cifar_batch(os.path.join('cifar-10-batches-py', 'test_batch'))

X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

model = CascadeBoostingClassifier(loss = "multinomial", n_layers=100, n_estimators = 30, max_depth=1, n_iter_no_change = None, validation_fraction = 0.1, learning_rate = 0.1,hidden_size = 8,verbose=1, n_trees=7,batch_size = 500,max_features=0.5)
model.fit(X_train, y_train) 

accuracy = model.score(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")