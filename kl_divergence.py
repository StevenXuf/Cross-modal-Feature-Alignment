import numpy as np
from scipy.stats import gaussian_kde

def compute_kl_divergence(P_samples,Q_samples):
    assert(P_samples.shape==Q_samples.shape)

# Estimate densities using KDE
    kde_P = gaussian_kde(P_samples.T)
    kde_Q = gaussian_kde(Q_samples.T)

# Monte Carlo approximation of KL divergence
# Evaluate log(p(x) / q(x)) for each point x in P_samples
    log_p = np.log(kde_P(P_samples.T))
    log_q = np.log(kde_Q(P_samples.T))
    kl_divergence = np.mean(log_p - log_q)

    return kl_divergence

if __name__=='__main__':
    num_samples=1000
    dim=126
    P_samples = np.random.normal(loc=0, scale=1, size=(num_samples, dim))
    Q_samples = np.random.normal(loc=0.5, scale=1.2, size=(num_samples, dim))

    compute_kl_divergence(P_samples,Q_samples)

