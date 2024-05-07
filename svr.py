from sklearn.svm import LinearSVR


class LinearSVRB(LinearSVR):
    def __init__(self,
                *,
                epsilon=0.0,
                tol=1e-4,
                C=1.0,
                loss="epsilon_insensitive",
                fit_intercept=True,
                intercept_scaling=1.0,
                dual="warn",
                verbose=0,
                random_state=None,
                max_iter=1000                 
                ):
        super().__init__(epsilon = epsilon,
                         tol = tol,
                         C = C,
                         loss = loss,
                         fit_intercept = fit_intercept,
                         intercept_scaling = intercept_scaling,
                         dual = dual,
                         verbose = verbose,
                         random_state = random_state,
                         max_iter = max_iter)

    def fit(self, X, y, bias = None,sample_weight = None):    
        super().fit(X, y - bias,sample_weight)

    def decision_function(self,X):
        return self.predict(X)
