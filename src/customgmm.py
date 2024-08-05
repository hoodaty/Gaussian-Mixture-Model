"""Custom Gaussian Mixture model

"""

class custom_GMM:
    def __init__(self):
        self._gmm_list = None
        self._log_prior = None
        self._log_likelihoods = []

    def fit(self, X, y, n_components, max_iter=100):
        self._log_prior = np.log(np.bincount(y.astype(int)) / len(y))
        shape = (len(self._log_prior), X.shape[1])
        self._gmm_list = np.empty(shape, dtype=object)
        for i in range(shape[0]):
            for j in range(shape[1]):
                print('fit model ({0},{1})'.format(i, j))
                model = GaussianMixture(n_components)
                a = X[y == i, j:j + 1]
                model.init_model(a)
                model.fit(a, max_iter)
                self._gmm_list[i, j] = model
                print('n_iter_: {0}'.format(model.n_iter_))

    def predict_proba(self, X):
        assert self._gmm_list is not None, 'gmm list is none'
        assert self._log_prior is not None, 'log prior is none'
        shape = (len(self._log_prior), X.shape[1], X.shape[0])
        ll = [[self._gmm_list[i][j].score_samples(X[:, j:j + 1])
            for j in range(shape[1])]
            for i in range(shape[0])]

        log_likelihood = np.sum(ll, axis=1).T
        log_joint = self._log_prior + log_likelihood.flatten()
        print(f"log_joint shape: {log_joint.shape}")
        predicts = np.exp(log_joint - logsumexp(log_joint, axis=0, keepdims=True))
        print(f"predicts shape: {predicts.shape}")

        return predicts