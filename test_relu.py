from scnn.optimize import optimize_model, sample_gate_vectors
from scnn.models import ConvexGatedReLU, NonConvexGatedReLU
from scnn.solvers import RFISTA
from scnn.metrics import Metrics
import nlopt

from sklearn.metrics import mean_squared_error


from sklearn import datasets, metrics, preprocessing
import numpy as np
from sklearn.model_selection import train_test_split

x, Y = datasets.load_breast_cancer(return_X_y = True)

Y = 2 * Y - 1

x = preprocessing.normalize(x, copy=False, axis = 0)

X_train, X_test, y_train, y_test = train_test_split(
    x, Y, test_size=0.5, shuffle=True
)

# create convex reformulation
max_neurons = 8
G = sample_gate_vectors(np.random.default_rng(123), X_train.shape[1], max_neurons)
model = ConvexGatedReLU(G)

D = model.compute_activations(X_train)

def data_mvp(v, X, D):
    w = v.reshape(-1, D.shape[1], X.shape[1])
    return np.einsum("ij, lkj, ik->il", X, w, D)

def gradient(
    v,
    X,
    y,
    D,
):
    w = v.reshape(-1, D.shape[1], X.shape[1])
    return np.einsum("ij, il, ik->ljk", D, data_mvp(w, X, D) - y.reshape(-1,w.shape[0]), X).reshape(*v.shape)

def myfunc(v, grad):
    if grad.size > 0:
        grad[:] = gradient(v,X_train,y_train,D)
    res = mean_squared_error(y_train.flatten(),data_mvp(v, X_train, D).flatten())
    #print(res)
    return res

x0 = np.random.rand(model.parameters[0].flatten().shape[0])
opt = nlopt.opt(nlopt.LD_TNEWTON, x0.shape[0])

opt.set_min_objective(myfunc)


opt.set_maxeval(300)
x = opt.optimize(x0)
minf = opt.last_optimum_value()
print("minimum value = ", minf)
print("result code = ", opt.last_optimize_result())

U = x.reshape(D.shape[1], X_train.shape[1])

model = NonConvexGatedReLU(G,1,False)
model.parameters[0] = U
model.parameters[1] = np.ones((1,U.shape[0]))

#training accuracy
train_acc = np.sum(np.sign(model(X_train)).flatten() == y_train.flatten()) / len(y_train)
test_acc = np.sum(np.sign(model(X_test)).flatten() == y_test.flatten()) / len(y_test)

print(train_acc,test_acc)

model = ConvexGatedReLU(G)
solver = RFISTA(model, tol=1e-6)
# choose metrics to collect during training
metrics = Metrics()#model_loss=True,
                  #train_accuracy=True,
                  #test_accuracy=True,
                  #neuron_sparsity=True)
# train model!
model, metrics = optimize_model(model,
                                solver,
                                metrics,
                                X_train,
                                y_train,
                                X_test,
                                y_test,
                                None,
                                device="cpu")

print(metrics.objective[-1]*2)
#training accuracy
train_acc = np.sum(np.sign(model(X_train)).flatten() == y_train.flatten()) / len(y_train)
test_acc = np.sum(np.sign(model(X_test)).flatten() == y_test.flatten()) / len(y_test)
print(train_acc,test_acc)