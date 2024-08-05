"""#### Gaussian Mixture Model definition

Gaussian Class
"""

class Gaussian:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def pdf(self, datum):
        u = (datum - self.mu) / abs(self.sigma)
        y = (1 / (np.sqrt(2 * np.pi) * abs(self.sigma))) * np.exp(-u * u / 2)
        return y

    def log_pdf_np(self, X):
        Y = (X - self.mu) / abs(self.sigma)
        Y = np.log((1 / (np.sqrt(2 * np.pi) * abs(self.sigma)))) + (-Y ** 2 / 2)
        return Y