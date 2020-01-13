import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import math

class DataSet:
    def __init__(self, xmin, xmax, num_data, noise_level):
        self.xmin = xmin
        self.xmax = xmax
        self.x = (xmax - xmin) * np.random.rand(num_data) + xmin
        self.x = np.sort(self.x)

        self.y = np.empty(num_data)
        for i in range(num_data):
            self.y[i] = self.make_y(self.x[i], noise_level)

    @staticmethod
    def make_y(x, noise_level):
            if x > 0:
                return 1.0/2.0*x**(1.2) + math.cos(x) + np.random.normal(0, noise_level)
            else:
                return math.sin(x) + 1.1**x + np.random.normal(0, noise_level)

    def plot(self): 
        plt.figure(figsize=(15, 10))
        plt.xlabel("x", fontsize=16)
        plt.ylabel("y", fontsize=16)
        plt.xlim(self.xmin, self.xmax)
        plt.tick_params(labelsize=16)
        plt.scatter(self.x, self.y)
        plt.show()

def generate_data(noise_level):
    def func(x):
        return 1.0/2.0*x**(1.2) + np.cos(x) + np.random.normal(0, noise_level)
    
    x = np.linspace(0,4,num=1000) + np.random.rand(1000)
    y = func(x) 

    # x = x[:,np.newaxis]
    # y = y[:,np.newaxis]

    x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.05)

    index = np.argsort(x_train)
    x_train = x_train[index]
    y_train = y_train[index]

    index = np.argsort(x_test)[::-1]
    x_test = x_test[index]
    y_test = y_test[index]

    return x_train, x_test, x, y_train, y_test

class GPRegression:
    def __init__(self, x_train, x_test, y_train):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.n = y_train.shape[0]

        # Param for gaussian kernel
        self.theta1 = 1
        self.theta2 = 0.4
        self.theta3 = 0.1

    def linear_kenel(self, x0, x1):
        kernel = np.dot(x0, x1.T)
        return kernel

    def gaussian_kernel(self, x0, x1):
        kernel = np.zeros((x0.shape[0], x1.shape[0]))

        for i in range(x0.shape[0]):
            for j in range(x1.shape[0]):
                # distance = np.abs(x0[i]-x1[j])**2
                # distance = np.sum(distance)
                distance = (x0[i]-x1[j])**2
                kernel[i,j] = self.theta1*np.exp(-distance/self.theta2)

        return kernel

    def gaussian_kernel_temp(self, x0, x1):
        if x0 == x1:
            delta = 1
        else:
            delta = 0

        return self.theta1*np.exp(-(x0-x1)**2/self.theta2)+self.theta3*delta

    def predict(self):
        k_train_train = self.gaussian_kernel(self.x_train, self.x_train)

        plt.figure()
        plt.imshow(k_train_train)
        plt.colorbar()
        plt.show()

        k_train_train_inv = np.linalg.inv(k_train_train)
        yy = np.dot(k_train_train_inv, self.y_train)

        k_train_test = self.gaussian_kernel(self.x_train, self.x_test)
        k_test_test = self.gaussian_kernel(self.x_test, self.x_test)
        mu = np.dot(k_train_test.T, yy)
        sigma_temp = np.dot(k_train_test.T, k_train_train_inv)
        sigma_temp = np.dot(sigma_temp, k_train_test)
        sigma = k_test_test - sigma_temp

        return mu, sigma

    def predict_temp(self):
        k_train_train = np.empty((self.x_train.shape[0], self.x_train.shape[0]))

        for i in range(self.x_train.shape[0]):
            for j in range(self.x_train.shape[0]):
                k_train_train[i][j] = self.gaussian_kernel_temp(self.x_train[i], self.x_train[j])

        yy = np.dot(np.linalg.inv(k_train_train), self.y_train)

        mu = []
        var = []
        for i in range(self.x_test.shape[0]):
            k_train_test = np.empty((self.x_train.shape[0],))
            for j in range(self.x_train.shape[0]):
                k_train_test[j] = self.gaussian_kernel_temp(self.x_train[j], self.x_test[i])
            
            k_test_test = self.gaussian_kernel_temp(self.x_test[i], self.x_test[i])
            
            mu.append(np.dot(k_train_test.T, yy))
            temp = np.dot(k_train_test.T, np.linalg.inv(k_train_train))
            var.append(k_test_test - np.dot(temp, k_train_test))
        
        return mu, var

    def plot(self):
        # mu, sigma = self.predict()
        # sigma_diag = np.diag(sigma)

        mu, var = self.predict_temp()

        mu = np.array(mu)
        var = np.array(var)

        y_test_predict_max = np.empty(mu.shape)
        y_test_predict_min = np.empty(mu.shape)

        y_test_predict_max = mu+var
        y_test_predict_min = mu-var

        plt.figure()
        plt.scatter(self.x_train, self.y_train, label='train')
        plt.plot(self.x_test, mu, label='test')
        plt.fill_between(self.x_test, y_test_predict_max, y_test_predict_min, alpha=0.5)
        plt.show()

if __name__ == '__main__':
    xmin = 0
    xmax = 4
    noise_level = 0.2
    train_data = DataSet(xmin, xmax, num_data=50, noise_level=noise_level)
    test_data = DataSet(xmin, xmax, num_data=1000, noise_level=noise_level)

    # x_train, x_test, x, y_train, y_test = generate_data()
    # gp = GPRegression(x_train, x, y_train)
    gp = GPRegression(train_data.x, test_data.x, train_data.y)
    gp.plot()


