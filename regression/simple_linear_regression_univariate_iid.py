# import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from linear_model.LinearRegression import LinearRegressionUniVariateIID

# set numpy random state
np.random.seed(42)

# generate y (1 dimension) - shape is (row=1000, column=1)
N = 1000
x = np.random.randn(N, 1)
y = 1 + 2*x + 0.1*np.random.randn(N, 1)
# print(y.shape)

# concatenate x and y and create pandas dataframe - shape is (row=1000, column=2)
df = pd.DataFrame(data=np.concatenate((x, y), axis=1), columns=['x', 'y'])
# print(df.shape)

# see mean and standard deviation of x and y
print('X: mean: ', np.mean(x), ' standard deviation: ', np.std(x))
print('y: mean: ', np.mean(y), ' standard deviation: ', np.std(y))

print('Descriptive statistics: \n', df.describe())

# Visualize scatter plot of our df
sns.scatterplot(data=df, x='x', y='y').set_title('distribution of dataset')
plt.show()

# Train Test Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
train_ones = np.ones((len(x_train), 1))
x_train = np.concatenate((train_ones, x_train), axis=1)


# def simple_linear_regression_1d_parameter(input_x, input_y):
#     """
#     This function is simple linear regression with assume of Independent Identical Distribution(IID) and x~N(1, sigma^2)
#      with single-variate input
#     :param input_x: x
#     :param input_y: y
#     :return: weights
#     """
#     numerator = np.zeros((1, 2))
#     denominator = 0
#     for i in range(len(input_x)):
#         numerator += input_x[i] * input_y[i]
#         denominator += input_x[i] ** 2
#     weights = numerator / denominator
#     return weights


# Modeling
rg = LinearRegressionUniVariateIID()
w = rg.fit(x_train, y_train)
print('Model weights is: ', w)

# Evaluate
y_pre = w[0][0] + w[0][1] * x_test
df_plot = pd.DataFrame(np.concatenate((x_test, y_test, y_pre), axis=1), columns=['x_test', 'y_test', 'y_pre'])
sns.lmplot(x='x_test', y='y_test', palette='y_pre', data=df_plot).fig.suptitle('Test vs model prediction')
plt.show()

# Evaluation of Error
error = y_pre - y_test
print('Mean of error: ', np.mean(error))
print('Standard deviation of error: ', np.std(error))
sns.lineplot(data=error).set_title('Error')
plt.show()
