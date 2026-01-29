from scnn.optimize import optimize_model, sample_gate_vectors
from scnn.models import ConvexGatedReLU
from scnn.solvers import RFISTA
from scnn.metrics import Metrics


"""Convex formulations of neural networks."""

from typing import Optional, List, Tuple, Dict
from math import ceil

from scipy.sparse.linalg import LinearOperator  # type: ignore

import lab

from scnn.private.models.model import Model
from scnn.private.models.regularizers import Regularizer
from . import operators
from scnn.private.utils import MatVecOperator
from scnn.private.loss_functions import squared_error, relu

# two-layer MLPs with ReLU activations.

import numpy as np

import lab


def data_mvp(v: lab.Tensor, X: lab.Tensor, D: lab.Tensor) -> lab.Tensor:
    w = v.reshape(-1, D.shape[1], D.shape[2], X.shape[1])
    return lab.einsum("inj, lnkj, ink->inl", X, w, D)


def data_t_mvp(v: lab.Tensor, X: lab.Tensor, D: lab.Tensor) -> lab.Tensor:
    return lab.einsum("inj, in, inl->jnl", D, v, X).reshape(-1,X.shape[1])


def gradient(
    v: lab.Tensor,
    X: lab.Tensor,
    y: lab.Tensor,
    D: lab.Tensor,
    mask: lab.Tensor,
) -> lab.Tensor:
    w = v.reshape(-1, D.shape[1], X.shape[1])
    return lab.einsum("inj, inl, ink->lnjk", D, data_mvp(w, X, D)[:,mask] - y[:,mask], X).reshape(*v.shape)


def hessian_mvp(v: lab.Tensor, X: lab.Tensor, D: lab.Tensor, mask: lab.Tensor) -> lab.Tensor:
    w = v.reshape(-1, D.shape[1],D.shape[2], X.shape[1])
    return lab.einsum("irj, nrkj, irk, irl, irm ->nrlm", X, w, D, D, X).reshape(*v.shape)[:,mask]


def bd_hessian_mvp(v: lab.Tensor, X: lab.Tensor, D: lab.Tensor, mask: lab.Tensor) -> lab.Tensor:
    w = v.reshape(-1, D.shape[1], D.shape[2], X.shape[1])
    return lab.einsum("ij, nkj, ik, il->nkl", X, w, D, X).reshape(*v.shape)[:,mask]


# builders
def data(X: lab.Tensor, D: lab.Tensor) -> lab.Tensor:
    return np.concatenate(lab.einsum("inj, ink->jnik", D, X), axis=2)


def hessian(X: lab.Tensor, D: lab.Tensor, mask: lab.Tensor) -> lab.Tensor:
    return lab.einsum("inj, ink, inl, inm -> jnkml", D, D, X, X)[:,mask]


def bd_hessian(X: lab.Tensor, D: lab.Tensor, mask: lab.Tensor) -> lab.Tensor:
    return lab.einsum("inj, inl, inm -> jnml", D, X, X)[:,mask]



class ConvexMLP(Model):
    """Convex formulation of a two-layer neural network (multi-layer
    perceptron) with ReLU activations."""

    def __init__(
        self,
        d: int,
        D: lab.Tensor,
        U: lab.Tensor,
        kernel: str = "einsum",
        regularizer: Optional[Regularizer] = None,
        c: int = 1,
    ) -> None:
        """
        :param d: the dimensionality of the dataset (ie. number of features).
        :param D: array of possible sign patterns.
        :param U: array of hyperplanes creating the sign patterns.
        :param kernel: the kernel to drive the matrix-vector operations.
        """
        super().__init__(regularizer)

        self.d = d
        self.p = D.shape[1]  # each column is a unique sign pattern
        self.c = c
        self.weights = lab.zeros(
            (c, self.p, self.d)
        )  # one linear model per sign pattern

        self.D = D
        self.U = U
        self.kernel = kernel

        (
            self._data_mvp,
            self._data_t_mvp,
            self._gradient,
            self._hessian_mvp,
            self._bd_hessian_mvp,
        ) = data_mvp, data_t_mvp, gradient, hessian_mvp, bd_hessian_mvp

        (
            self._data_builder,
            self._hessian_builder,
            self._bd_hessian_builder,
        ) = data, hessian, bd_hessian

        self._train = True

    def _weights(self, w: Optional[lab.Tensor]) -> lab.Tensor:
        return self.weights if w is None else w

    def get_reduced_weights(self) -> lab.Tensor:
        return self.weights

    def _signs(self, X: lab.Tensor, D: Optional[lab.Tensor] = None):
        local_D = self.D
        if D is not None:
            return D
        elif not self._train:
            local_D = relu(X @ self.U)
            local_D[local_D > 0] = 1

        return local_D

    def _forward(
        self,
        X: lab.Tensor,
        w: lab.Tensor,
        D: Optional[lab.Tensor] = None,
        mask: lab.Tensor = None,
        **kwargs,
    ) -> lab.Tensor:
        """Compute forward pass.

        :param X: (n,m,d) array containing the data examples.
        :param w: parameter at which to compute the forward pass.
        :param D: (optional) specific activation matrix at which to compute the forward pass.
        Defaults to self.D or manual computation depending on the value of self._train.
        :returns: predictions for X.
        """
        if mask is not None:
            return self._data_mvp(w, X, self._signs(X, D))[:,mask,:]
        else:
            return self._data_mvp(w, X, self._signs(X, D))    

    def _objective(
        self,
        X: lab.Tensor,
        y: lab.Tensor,
        w: lab.Tensor,
        D: Optional[lab.Tensor] = None,
        scaling: Optional[float] = None,
        mask: lab.Tensor = None,
        **kwargs,
    ) -> float:
        return self.__objective(X,y,w,D,scaling,mask = self.mask,**kwargs)
    
    def __objective(
        self,
        X: lab.Tensor,
        y: lab.Tensor,
        w: lab.Tensor,
        D: Optional[lab.Tensor] = None,
        scaling: Optional[float] = None,
        mask: lab.Tensor = None,
        **kwargs,
    ) -> float:
        """Compute the l2 objective with respect to the model weights.

        :param X: (n,d) array containing the data examples.
        :param y: (n,d) array containing the data targets.
        :param w: parameter at which to compute the objective.
        :param D: (optional) specific activation matrix at which to compute the forward pass.
        Defaults to self.D or manual computation depending on the value of self._train.
        :param scaling: (optional) scaling parameter for the objective. Defaults to `n * c`.
        :returns: the objective
        """

        return squared_error(self._forward(X, w, D, mask), y) / (
            2 * self._scaling(y, scaling)
        )

    def _grad(
        self,
        X: lab.Tensor,
        y: lab.Tensor,
        w: lab.Tensor,
        D: Optional[lab.Tensor] = None,
        flatten: bool = False,
        scaling: Optional[float] = None,
        **kwargs,
    ) -> lab.Tensor:
        """Compute the gradient of the l2 objective with respect to the model
        weights. As in 'self.__call__' above, we could optimize this by
        implementing it in a faster low-level language.

        :param X: (n,d) array containing the data examples.
        :param y: (n,d) array containing the data targets.
        :param w: parameter at which to compute the gradient.
        :param D: (optional) specific activation matrix at which to compute the forward pass.
        Defaults to self.D or manual computation depending on the value of self._train.
        :param flatten: whether or not to flatten the blocks of the gradient into a single vector.
        :param scaling: (optional) scaling parameter for the objective. Defaults to `n * c`.
        :returns: the gradient
        """
        grad = self._gradient(w, X, y, self._signs(X, D),self.mask) / self._scaling(
            y[self.mask], scaling
        )

        if flatten:
            grad = grad.reshape(-1)

        return grad

    def data_operator(
        self,
        X: lab.Tensor,
        D: Optional[lab.Tensor] = None,
    ) -> LinearOperator:
        """Construct a matrix operator to evaluate the matrix-vector equivalent
        to the sum,

        .. math::

            \\sum_i D_i X v_i

        where v_i is the i'th block of the input vector 'v'. This is equivalent to
        constructing the expanded matrix :math:`A = [D_1 X, D_2 X, ..., D_P X]` and then evaluating :math:`Av`.
        Use 'data_matrix' to directly compute $A$.
        :param X: (n,d) array containing the data examples.
        :param D: (optional) specific activation matrix at which to compute the forward pass.
        Defaults to self.D or manual computation depending on the value of self._train.
        :returns: LinearOperator which computes the desired product.
        """
        n, _ = X.shape
        pd = self.d * self.p
        local_D = self._signs(X, D)

        # pre-compute extended matrix
        if self.kernel == operators.DIRECT:
            expanded_X = self._data_builder(X, local_D)

            def forward(v):
                return lab.squeeze(
                    self._data_mvp(v, X=X, D=local_D, expanded_X=expanded_X)[:,self.mask]
                )

            def transpose(v):
                return lab.squeeze(
                    self._data_t_mvp(v, X=X, D=local_D, expanded_X=expanded_X)[:,self.mask]
                )

        else:

            def forward(v):
                return lab.squeeze(self._data_mvp(v, X=X, D=local_D)[:,self.mask])

            def transpose(v):
                return lab.squeeze(self._data_t_mvp(v, X=X, D=local_D)[:,self.mask])

        op = MatVecOperator(
            shape=(n, pd), forward=forward, transpose=transpose
        )

        return op

    def add_new_patterns(
        self, patterns: lab.Tensor, weights: lab.Tensor, remove_zero=False
    ) -> lab.Tensor:
        """Attempt to augment the current model with additional sign patterns.

        :param patterns: the tensor of sign patterns to add to the current set.
        :param weights: the tensor of weights which induced the new patterns.
        :param remove_zero: whether or not to remove the zero vector.
        :returns: None
        """

        if lab.size(patterns) > 0:
            # update neuron count.
            self.D = lab.concatenate([self.D, patterns], axis=1)
            self.U = lab.concatenate([self.U, weights.T], axis=1)

            # initialize new model components at 0.
            added_weights = lab.zeros((self.c, weights.shape[0], self.d))
            self.weights = lab.concatenate(
                [self.weights, added_weights], axis=1
            )

            # filter out the zero column.
            if remove_zero:
                non_zero_cols = lab.logical_not(
                    lab.all(self.D == lab.zeros((self.D.shape[0], 1)), axis=0)
                )
                self.D = self.D[:, non_zero_cols]
                self.U = self.U[:, non_zero_cols]
                self.weights = self.weights[:, non_zero_cols]

            self.p = self.D.shape[1]

        return added_weights

    def batch_X(
        self, batch_size: Optional[int], X: lab.Tensor
    ) -> List[Dict[str, lab.Tensor]]:
        D = self._signs(X)

        if batch_size is None:
            return [{"X": X, "D": D}]

        n = X.shape[0]
        n_batches = ceil(n / batch_size)

        return [
            {
                "X": X[i * batch_size : (i + 1) * batch_size],
                "D": D[i * batch_size : (i + 1) * batch_size],
            }
            for i in range(n_batches)
        ]

    def batch_Xy(
        self, batch_size: Optional[int], X: lab.Tensor, y: lab.Tensor
    ) -> List[Dict[str, lab.Tensor]]:
        D = self._signs(X)

        if batch_size is None:
            return [{"X": X, "y": y, "D": D}]

        n = X.shape[0]
        n_batches = ceil(n / batch_size)

        return [
            {
                "X": X[i * batch_size : (i + 1) * batch_size],
                "y": y[i * batch_size : (i + 1) * batch_size],
                "D": D[i * batch_size : (i + 1) * batch_size],
            }
            for i in range(n_batches)
        ]

    def sign_patterns(
        self,
        X: lab.Tensor,
        y: lab.Tensor,
        w: Optional[lab.Tensor] = None,
        **kwargs,
    ) -> Tuple[lab.Tensor, lab.Tensor]:
        """Compute the gradient of the l2 objective with respect to the model
        parameters.

        :param X: (n,d) array containing the data examples.
        :param y: (n) array containing the data targets.
        :param w: (optional) specific parameter at which to compute the sign patterns.
        :returns: the set of sign patterns active at w or the current models parameters if w is None.
        """

        return lab.sign(relu(X @ self.U)), self.U.T


def compute_activation_patterns(
    X: np.ndarray,
    G: np.ndarray,
    filter_duplicates: bool = True,
    filter_zero: bool = True,
    bias: bool = False,
    active_proportion: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute activation patterns corresponding to a set of gate vectors.

    Args:
        X: an (n x d) data matrix, where the examples are stored as rows.
        G: an (d x m) matrix of "gate vectors" used to generate the activation
            patterns.
        filter_duplicates: whether or not to remove duplicate activation
            patterns and the corresponding
        filter_zero: whether or not to filter the zero activation pattern and
            corresponding gates. Defaults to `True`.
        bias: whether or not a bias should be included in the gate vectors.
        active_proportion: force each gate to be active for
            `active_proportion`*n of the training examples using a bias
            term. This feature is only supported when `bias == True`.

    Returns:
        - `D`, an (n x p) matrix of (possibly unique) activation patterns where
            :math:`p \\leq m`

        - `G`, a (d x b) matrix of gate vectors generating `D`.
    """
    n, d = X.shape

    # need to extend the gates with zeros.
    if bias and G.shape[0] + 1 == X.shape[1]:
        G = np.concatenate([G, np.zeros((1, G.shape[1]))], axis=0)

    XG = np.matmul(X, G)

    if active_proportion is not None:
        # Gates must be augmented with a row of zeros to be valid.
        assert np.all(G[-1] == 0)
        # X must be augmented with a column of ones to be valid.
        assert np.all(X[:, -1] == X[0, -1])

        # set bias terms in G
        quantiles = np.quantile(XG, q=1 - active_proportion, axis=0, keepdims=True)
        XG = XG - quantiles
        G = G.copy()
        G[-1] = -np.ravel(quantiles)

    XG = np.maximum(XG, 0)
    XG[XG > 0] = 1

    if filter_duplicates:
        D, indices = np.unique(XG, axis=1, return_index=True)
        G = G[:, indices]

    # filter out the zero column.
    if filter_zero:
        non_zero_cols = np.logical_not(np.all(D == np.zeros((n, 1)), axis=0))
        D = D[:, non_zero_cols]
        G = G[:, non_zero_cols]

    return D, G


D, G = lab.all_to_tensor(
    compute_activation_patterns(
        lab.to_np(X_train),
        G_input,
        bias=model.bias,
    ),
    dtype=lab.get_dtype(),
)


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
max_neurons = 16
G = sample_gate_vectors(np.random.default_rng(123), X_train.shape[1], max_neurons)
model = ConvexGatedReLU(G)
# specify regularizer and solver
#regularizer = NeuronGL1(lam=0.001)
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

# training accuracy
train_acc = np.sum(np.sign(model(X_train)).flatten() == y_train.flatten()) / len(y_train)
test_acc = np.sum(np.sign(model(X_test)).flatten() == y_test.flatten()) / len(y_test)

print(train_acc,test_acc)