import numpy as np


class LinearRegressionUniVariateIID:
    """
    Model Name: Linear Regression
    Assumption: Independent and identically distributed random variables, y~N(0, sigma^2)
    """
    def __init__(self):
        pass
        # self.x = x
        # self.y = y

    def fit(self, x, y):
        """
        Take x and y as input and return weight of linear regression equation
        :param x: x_train
        :param y: y_train
        :return: weight
        """
        numerator = np.zeros((1, 2))
        denominator = 0
        for i in range(len(x)):
            numerator += x[i] * y[i]
            denominator += x[i] ** 2
        weights = numerator / denominator
        return weights


class LinearRegressionMultivariateIID:
    """
    Model Name: Linear Regression
    Assumption: Independent and identically distributed random variables, y~N(0, sigma^2)
    """
    def __init__(self):
        pass
        # self.x = x
        # self.y = y

    def fit(self, x, y):
        """
        Take x and y as input and return weight of linear regression equation
        :param x: x_train
        :param y: y_train
        :return: weight
        """
        return np.dot(np.linalg.inv(np.dot(np.transpose(x), x)), np.dot(np.transpose(x), y))


class LinearRegressionMultivariateNonIID:
    """
    Model Name: Linear Regression
    Assumption: Independent distributed random variables (Non Identical), y~N(0, sigma^2)
    """
    def __init__(self):
        pass
        # self.x = x
        # self.y = y

    def fit(self, x, y):
        """
        Take x and y as input and return weight of linear regression equation
        :param x: x_train
        :param y: y_train
        :return: weight
        """
        distance = np.exp(np.abs((x - np.mean(x))))
        mean = np.mean(distance, axis=1).reshape(-1, 1)
        weights = np.diagflat(np.matrix(mean))
        w = np.matmul(
            np.linalg.inv(np.matmul(np.matmul(np.transpose(x), np.linalg.inv(weights)), x)),
            np.matmul(np.matmul(np.transpose(x), np.linalg.inv(weights)), y)
        )
        return w
