import numpy as np
import matplotlib.pyplot as plt

def linear_model(x, theta):
    return theta[0] + theta[1]*x

def likelihood(y, x, theta, sigma):
    return np.prod(np.exp(-0.5 * (y - linear_model(x, theta))**2 / sigma**2) / np.sqrt(2 * np.pi * sigma**2))

def prior(theta):
    if all(theta >= 0) and all(theta <= 10):
        return 1.0
    return 0.0

def posterior(y, x, theta, sigma):
    return likelihood(y, x, theta, sigma) * prior(theta)

def mcmc(y, x, sigma, n_steps=200000):
    theta = np.zeros((n_steps, 2))
    theta[0] = np.random.uniform(0, 1, 2)
    for i in range(1, n_steps):
        new_theta = theta[i-1] + np.random.normal(0, 1, 2)
        acceptance_prob = posterior(y, x, new_theta, sigma) / posterior(y, x, theta[i-1], sigma)
        if acceptance_prob > np.random.uniform(0, 1):
            theta[i] = new_theta
        else:
            theta[i] = theta[i-1]
    return theta

# Generate some synthetic data
np.random.seed(0)
x = np.sort(10 * np.random.rand(50))
y = 3 + 0.5 * x + np.random.normal(0, 2, x.shape)
sigma = 2

# Perform MCMC
theta = mcmc(y, x, sigma)

# Plot posterior distribution of the coefficients
plt.figure(figsize=(10,5))
plt.subplot(121)
plt.hist(theta[:,0], bins=20, label='Intercept', density=True)
plt.legend()
plt.subplot(122)
plt.hist(theta[:,1], bins=20, label='Slope', density=True)
plt.legend()
plt.savefig('./Posterior.png')
plt.close()

