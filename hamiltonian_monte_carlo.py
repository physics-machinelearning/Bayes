import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma

class MetropolisHastings:
    def __init__(self, epsilon, theta, NMCS, f):
        self.epsilon = epsilon
        self.theta = theta
        self.NMCS = NMCS
        self.f = f

    def judgeacceptornot(self, theta, new_theta):
        judge = self.f(new_theta) > self.f(theta)
        r = self.f(new_theta) / self.f(theta)
        return judge, r

    def loop(self):
        theta = self.theta
        self.theta_list = [theta]
        for i in range(self.NMCS):
            theta_new = theta + self.epsilon*np.random.randn()
            judge, r = self.judgeacceptornot(theta, theta_new)
            if judge:
                theta = theta_new
            else:
                if np.random.rand() < r:
                    theta = theta_new

            self.theta_list.append(theta)
    
    def plot(self):
        theta = np.linspace(min(self.theta_list), max(self.theta_list), num=100)
        fig, ax1 = plt.subplots()
        ax1.hist(self.theta_list, density=True, bins=40)
        ax2 = ax1.twinx()
        ax2.plot(theta, self.f(theta))
        plt.show()
    
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

class ReplicaExchangeMethod:
    """
    This class is for generation of samples from 
    one dimentional gaussian mixture distribution
    """
    def __init__(self, epsilon, mu_list, sigma, NMCS, e_freq, num_chain):
        self.epsilon = epsilon
        self.mu_list = mu_list
        self.sigma = sigma
        self.NMCS = NMCS
        self.e_freq = e_freq
        self.num_chain = num_chain

    def judgeacceptornot(self, x, new_x, f):
        judge = f(new_x) > f(x)
        r = f(new_x) / f(x)
        return judge, r

    def loop(self):
        mu_min = min(self.mu_list)
        mu_max = max(self.mu_list)
        self.x_list = [[np.random.uniform(mu_min, mu_max)] for i in range(self.num_chain)]
        center = (mu_min+mu_max)/2
        dis = (mu_max - center)/self.num_chain
        temp = [np.linspace(center-dis*i, center+dis*i, len(self.mu_list)) for i in range(1,self.num_chain+1)]
        distributions = [generate_gm(mu_list_temp, self.sigma) for mu_list_temp in temp]
        
        for i in range(self.NMCS):
            for i, x in enumerate(self.x_list):
                x = x[-1]
                x_new = x + self.epsilon*np.random.randn()
                f = distributions[i]
                judge, r = self.judgeacceptornot(x, x_new, f)
                if judge:
                    x = x_new
                else:
                    if np.random.rand() < r:
                        x = x_new

                self.x_list[i].append(x)
            
            if i %self.e_freq == 0:
                index = int(np.random.uniform(0,len(self.mu_list)-1))
                x0 = self.x_list[index][-1]
                x1 = self.x_list[index+1][-1]
                dist0 = distributions[index]
                dist1 = distributions[index+1]

                if np.random.uniform() < (dist0(x1)*dist1(x0))/(dist0(x0)*dist1(x1)):
                    self.x_list[index][-1], self.x_list[index+1][-1] = np.copy(x1),np.copy(x0)
    
    def plot(self):
        theta = np.linspace(min(self.x_list[-1]), max(self.x_list[-1]), num=100)
        fig, ax1 = plt.subplots()
        ax1.hist(self.x_list[-1], density=True, bins=40)
        ax2 = ax1.twinx()
        ax2.plot(theta, generate_gm(self.mu_list, self.sigma)(theta))
        plt.show()

        # for i in range(len(self.x_list)):
        #     plt.figure()
        #     plt.hist(self.x_list[i], alpha=0.5, label=str(i))
        #     plt.legend()
        #     plt.show()

class HamiltonianMonteCarlo:
    def __init__(self, epsilon, T, L, theta, f, h, dhdtheta):
        self.epsilon = epsilon
        self.T = T
        self.L = L
        self.theta = theta
        self.f = f
        self.h = h
        self.dhdtheta = dhdtheta

    def leapfrog(self, p, theta):
        p = p - (self.epsilon*self.dhdtheta(theta)/2)
        theta = theta + self.epsilon*p
        p = p - (self.epsilon*self.dhdtheta(theta)/2)

        return p, theta

    def calchamiltonian(self, theta, p):
        return self.h(theta) + p**2/2

    def judgeacceptornot(self, new_theta, new_p, old_theta, old_p):
        r = np.exp(self.calchamiltonian(new_theta, new_p)-self.calchamiltonian(old_theta, old_p))
        return np.random.rand() < r

    def loop(self):
        theta = self.theta
        
        self.p_list = []
        self.theta_list = []
        accept_list = []

        for i in range(self.T):
            p = np.random.randn()
            
            old_p = p
            old_theta = theta
            
            for j in range(self.L):
                p, theta = self.leapfrog(p, theta)

            new_p = p
            new_theta = theta

            if self.judgeacceptornot(new_theta, new_p, old_theta, old_p):
                self.p_list.append(p)
                self.theta_list.append(theta)
                accept_list.append(True)
            else:
                p = old_p
                theta = old_theta
                accept_list.append(False)
        
        print('accept ratio: ', sum(accept_list)/len(accept_list))

    def plot_transition(self):
        plt.figure()
        plt.plot(self.p_list, self.theta_list)
        plt.show()

    def plot(self):
        theta = np.linspace(min(self.theta_list), max(self.theta_list), num=100)
        fig, ax1 = plt.subplots()
        ax1.hist(self.theta_list, density=True, bins=40)
        ax2 = ax1.twinx()
        ax2.plot(theta, self.f(theta))
        plt.show()

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

if __name__ == '__main__':
#     f = f_gamma
#     h = h_gamma
#     dhdtheta = dhdtheta_gamma

#     # HMC
#     epsilon = 0.01
#     T = 10000
#     L = 100
#     theta = 3.0

#     temp = HamiltonianMonteCarlo(epsilon=epsilon, T=T, L=L, theta=theta, \
#     f=f, h=h, dhdtheta=dhdtheta)
#     temp.loop()
#     temp.plot()

#     # MH
#     theta = 10
#     epsilon = 0.3

#     temp = MetropolisHastings(epsilon=epsilon, theta=theta, NMCS=T*L, f=f)
#     temp.loop()
#     temp.plot()
    epsilon = 0.3
    mu_list = [-3,3]
    e_freq = 100
    sigma = 1
    NMCS = 100000
    num_chain = 5
    re_ex = ReplicaExchangeMethod(epsilon, mu_list, sigma, NMCS, e_freq, num_chain)
    re_ex.loop()
    re_ex.plot()


            
