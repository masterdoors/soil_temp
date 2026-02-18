# This is modified code from Scnn project https://github.com/pilancilab/scnn, which is under MIT License:
# MIT License

# Copyright (c) 2022 Pilanci Research Group

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
from scipy.special import expit, softmax
from typing import List, Optional
from scipy import linalg


class Model:
    """Base class for convex and non-convex models.

    Attributes:
        c: the output dimension.
        d: the input dimension.
        p: the number of neurons.
        bias: whether or not the model uses a bias term.
        parameters: a list of NumPy arrays comprising the model parameters.
    """

    d: int
    p: int
    c: int
    bias: bool
    parameters: List[np.ndarray]

class GatedModel(Model):
    """Abstract class for models with fixed gate vectors.

    Attributes:
        c: the output dimension.
        d: the input dimension.
        p: the number of neurons. This is is always `1` for a linear model.
        bias: whether or not the model uses a bias term.
        G: the gate vectors for the Gated ReLU activation stored as a
            (d x p) matrix.
        G_bias: an optional vector of biases for the gates.
    """

    def __init__(
        self,
        G: np.ndarray,
        c: int,
        bias: bool = False,
        G_bias: Optional[np.ndarray] = None,
    ):
        """Construct a new convex Gated ReLU model.

        Args:
            G: a (d x p) matrix of get vectors, where p is the
                number neurons.
            c: the output dimension.
            bias: whether or not to include a bias term.
            G_bias: a vector of bias parameters for the gates.
                Note that `bias` must be True for this to be supported.
        """
        self.G = G
        self.d, self.p = G.shape
        self.c = c
        self.bias = bias

        if bias is None:
            assert G_bias is None
        self.G_bias = G_bias

        if self.G_bias is None:
            self.G_bias = np.zeros(self.p)

    def compute_activations(self, X: np.ndarray) -> np.ndarray:
        """Compute activations for models with fixed gate vectors.

        Args:
            X: (n x d) matrix of input examples.

        Returns:
            D: (n x p) matrix of activation patterns.
        """
        D = np.maximum(X @ self.G + self.G_bias, 0)
        D[D > 0] = 1

        return D

class ConvexGatedReLU(GatedModel):
    """Convex reformulation of a Gated ReLU Network with two-layers.

    This model has the prediction function

    .. math::

        g(X) = \\sum_{i=1}^m \\text{diag}(X g_i > 0) X U_{1i}.

    A one-vs-all strategy is used to extend the model to multi-dimensional
    targets.

    Attributes:
        c: the output dimension.
        d: the input dimension.
        p: the number of neurons.
        bias: whether or not the model uses a bias term.
        G: the gate vectors for the Gated ReLU activation stored as a
            (d x p) matrix.
        G_bias: an optional vector of biases for the gates.
        parameters: the parameters of the model stored as a list of tensors.
    """

    def __init__(
        self,
        G: np.ndarray,
        c: int = 1,
        bias: bool = False,
        G_bias: Optional[np.ndarray] = None,
    ) -> None:
        """Construct a new convex Gated ReLU model.

        Args:
            G: a (d x p) matrix of get vectors, where p is the
                number neurons.
            c: the output dimension.
            bias: whether or not to include a bias term.
            G_bias: a vector of bias parameters for the gates.
                Note that `bias` must be True for this to be supported.
        """

        super().__init__(G, c, bias, G_bias)

        # one linear model per gate vector
        if self.bias:
            self.parameters = [
                np.zeros((c, self.p, self.d)),
                np.zeros((c, self.p)),
            ]
        else:
            self.parameters = [np.zeros((c, self.p, self.d))]

    def get_parameters(self) -> List[np.ndarray]:
        """Get the model parameters."""
        return self.parameters

    def set_parameters(self, parameters: List[np.ndarray]):
        """Set the model parameters.

        This method safety checks the dimensionality of the new parameters.

        Args:
            parameters: the new model parameters.
        """
        assert parameters[0].shape == (self.c, self.p, self.d)

        if self.bias:
            assert parameters[1].shape == (self.c, self.p)

        self.parameters = parameters

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Compute the model predictions for a given dataset.

        Args:
            X: an (n  d) array containing the data examples on
                which to predict.

        Returns:
            - g(X): the model predictions for X.
        """
        D = super().compute_activations(X)

        if self.bias:
            Z = (
                np.einsum("ij, lkj->lik", X, self.parameters[0])
                + self.parameters[1]
            )

            return np.einsum("lik, ik->il", Z, D)
        else:
            return np.einsum("ij, lkj, ik->il", X, self.parameters[0], D)


def data_mvp(v, X, D, bias):
    w = v.reshape(-1, D.shape[1], X.shape[1])
    return np.einsum("ij, lkj, ik->il", X, w, D) + bias.reshape(X.shape[0],w.shape[0])

def data_mvp_soft(v, X, D, bias):
    return softmax(data_mvp(v, X, D, bias), axis = 1)

def data_mvp_expit(v, X, D, bias):
    return expit(data_mvp(v, X, D, bias))

def inv_hessian(U,
    X,
    v,
    D,
    C):
    inv_H = np.identity(U.shape[0]) / C
    #print("H det: ", linalg.det(inv_H))

    for i in range(X.shape[0]):
        M = (X[i].reshape(-1,1) @ D[i].reshape(1,-1)).reshape(-1,1)
        mt = np.transpose(M)
        inv_H = inv_H - (inv_H @ M @ mt @ inv_H) / (1. + mt @ inv_H @ M).sum()
        #print("H det: ",linalg.det(inv_H))
    return inv_H #inv_H @ X @ D @ v    

     
def gradient_l2(
    v,
    X,
    y,
    D,
    bias
):
    inv_hessian(v,X,y,D,0.1)
    w = v.reshape(-1, D.shape[1], X.shape[1])
    return np.einsum("ij, il, ik->ljk", D, data_mvp(w, X, D, bias) - y.reshape(-1,w.shape[0]), X).reshape(*v.shape) + v * 0.00001

def gradient_hb(
    v,
    X,
    y,
    D,
    bias    
):
    w = v.reshape(-1, D.shape[1], X.shape[1])
    return np.einsum("ij, il, ik->ljk", D, expit(data_mvp(w, X, D, bias)) - y.reshape(-1,w.shape[0]), X).reshape(*v.shape)

def gradient_hm(
    v,
    X,
    y,
    D,
    bias    
):
    w = v.reshape(-1, D.shape[1], X.shape[1])
    n_classes = w.shape[0]
    y_ =  y.flatten().astype(int)
    out = softmax(data_mvp(w, X, D, bias), axis = 1)
    grad = np.zeros((n_classes,D.shape[1],X.shape[1]))
    for k in range(n_classes):
        grad[k] =  np.einsum("ij, i, ik->jk", D, out[:,k] - (y_ == k).astype(float), X)
    return grad.reshape(*v.shape)    