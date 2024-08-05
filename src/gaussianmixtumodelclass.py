"""Gaussian Mixture Model Class

"""

# Commented out IPython magic to ensure Python compatibility.
class GaussianMixture:

    def __init__(self, n_components):
        # list of gaussian components
        self.g = None

        # weights of the gaussian components
        self.mix = None

        # the number of mixture components.
        self.n_components = n_components

        # we define convergence threshold, E-M iterations will stop when the lower bound average gain is below this threshold.
        self.tol = 0.001

        # steps taken by the best fit of E-M to reach convergence.
        self.n_iter_ = None

        # the method used to initialize the weights, the means and the precisions.
        self.init_params = 'kmeans'

    def _initialize_parameters(self, X, random_state=42):

        n_samples, _ = X.shape

        if self.init_params == 'kmeans':#We initialise using KMeans which we import from the sk-learn package since coding KMeans from scratch is not our goal
            resp = np.zeros((n_samples, self.n_components))
            label = KMeans(n_clusters=self.n_components, n_init=1,random_state=random_state).fit(X).labels_
            resp[np.arange(n_samples), label] = 1
        else:
            raise ValueError("Unimplemented initialization method '%s'"
#                              % self.init_params)

        self.m_step(X, resp)

    def init_model(self, X):#initialising the model
        mu_min = np.min(X, axis=0)
        mu_max = np.max(X, axis=0)
        sigma_min = 1
        sigma_max = 1
        g = []
        mix = []

        for i in range(self.n_components):
           g.append(Gaussian(uniform(mu_min, mu_max), uniform(sigma_min, sigma_max)))
           mix.append(1 / self.n_components)

        self.g = g
        self.mix = mix

        return self

    def e_step(self, X):#expectation step to calculate calculate responsibilities

        assert X is not None and len(X) > 0, 'X is none or empty'
        assert self.g is not None and len(self.g) > 0, 'g is none or empty'
        assert self.mix is not None and len(self.mix) > 0, 'mix is none or empty'
        assert len(self.g) == len(self.mix), 'length of g and mix is not equal'

        log_prob_norm, log_resp = self.estimate_log_prob_resp(X)
        return np.mean(log_prob_norm), log_resp

    def estimate_log_prob_resp(self, X):

        weighted_log_prob = self.estimate_weighted_log_prob_np(X)
        log_prob_norm = logsumexp(weighted_log_prob, axis=1)
        assert len(log_prob_norm) == len(X), 'length of log_prob_norm error'
        with np.errstate(under='ignore'):# We ignore underflow
            log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]

        return log_prob_norm, log_resp

    def estimate_weighted_log_prob_np(self, X):#We calculate the weighted log probablility array

        X = np.array(X)
        X = X.flatten()
        weighted_log_prob = np.zeros((len(X), len(self.g)))
        for i in range(len(self.g)):
            print(f'g[{i}] log_pdf_np shape: {self.g[i].log_pdf_np(X).shape}')
            print(f'mix[{i}] log shape: {np.log(self.mix[i]).shape}')
            weighted_log_prob[:, i] = self.g[i].log_pdf_np(X) + np.log(self.mix[i])

        return weighted_log_prob

    def m_step(self, X, resp):

        nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
        resp = resp.T
        # compute new means
        for i in range(len(self.g)):
            self.g[i].mu = np.dot(resp[i], np.array(X)) / nk[i]
        # compute new sigmas
        for i in range(len(self.g)):
            self.g[i].sigma = np.sqrt(np.dot(resp[i], (np.array(X) - self.g[i].mu) ** 2) / nk[i])
        # compute new weights
        for i in range(len(self.g)):
            self.mix[i] = nk[i] / len(X)

    def pdf(self, x):
        v = 0
        for i in range(len(self.g)):
            v += self.g[i].pdf(x) * self.mix[i]
        return v

    def fit(self, X, max_iter):

        self.init_model(X)
        self._initialize_parameters(X)
        lower_bound = None
        for i in range(max_iter):
            self.n_iter_ = i
            prev_lower_bound = lower_bound
            log_prob_norm, log_resp = self.e_step(X)
            self.m_step(X, np.exp(log_resp))
            lower_bound = log_prob_norm
            if prev_lower_bound is not None:
                change = lower_bound - prev_lower_bound
                if abs(change) < self.tol:
                    break

    def score_samples(self, X):

        weighted_log_prob = self.estimate_weighted_log_prob_np(X)
        log_prob_max = np.max(weighted_log_prob, axis=1)
        log_prob_normalized = weighted_log_prob - log_prob_max[:, np.newaxis]
        log_sum_exp = log_prob_max + np.log(np.sum(np.exp(log_prob_normalized), axis=1))

        return log_sum_exp