import numpy as np
from scipy.stats import gamma

def generate_gm(mu_list, sigma):
    def gauss(x):
        return 1.0/(2.0*np.pi)**0.5*np.exp(-0.5*x**2)
    def gm(x):
        x = np.array(x)
        ratio = np.array([1/len(mu_list)]*len(mu_list))
        gauss_list = [gauss(x-mu) for mu in mu_list]
        gauss_list = np.array(gauss_list)
        return np.dot(ratio, gauss_list)
    return gm

def f_gamma(theta):
    lam = 1
    alpha = 11
    return np.exp(-lam*theta)*theta**(alpha-1)

def h_gamma(theta):
    lam = 1
    alpha = 11
    return lam*theta - (alpha-1)*np.log(theta)

def dhdtheta_gamma(theta):
    lam = 1
    alpha = 11
    return lam - (alpha-1)/theta