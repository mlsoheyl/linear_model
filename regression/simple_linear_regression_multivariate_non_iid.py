# Initialization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from linear_model.LinearRegression import LinearRegressionMultivariateNonIID

# Generate data
N = 1000
x1 = np.random.randn(N, 1)
x2 = 2 + np.random.randn(N, 1)
x3 = 2 * np.random.randn(N, 1)
x4 = 2 + 2 * np.random.randn(N, 1)
X = np.concatenate((x1, x2, x3, x4), axis=1)

y = np.empty((N, 1))

for entry in range(N):
    sigma = np.random.randint(low=0.1, high=4)
    y[entry] = 0.5 - 3 * x1[entry] + 4 * x2[entry] + 3 * x3[entry] - 0.05 * x4[entry] - sigma * np.random.randn(1, 1)

# Convert data to pandas dataframe
df = pd.DataFrame(data=np.concatenate((X, y), axis=1), columns=['x1', 'x2', 'x3', 'x4', 'y'])
print('Descriptive Statistics')
print(df.head().to_string())

# Plot dataset
sns.scatterplot(data=df, x='x1', y='y')
sns.scatterplot(data=df, x='x2', y='y')
sns.scatterplot(data=df, x='x3', y='y')
sns.scatterplot(data=df, x='x4', y='y')
plt.title('Scatter 2d(input and output)')
plt.xlabel('x1, x2, x3, x4')
plt.show()

# Train Test Split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Preparing Data
one_train = np.ones((x_train.shape[0], 1))
one_test = np.ones((x_test.shape[0], 1))
x_train = np.concatenate((one_train, x_train), axis=1)
x_test = np.concatenate((one_test, x_test), axis=1)

# distance = np.exp(np.abs((x_train - np.mean(x_train))))
# mean = np.mean(distance, axis=1).reshape(-1, 1)
# weights = np.diagflat(np.matrix(mean))
# print(weights.shape)


# def linear_regression_wls(x, y):
#     distance = np.exp(np.abs((x - np.mean(x))))
#     mean = np.mean(distance, axis=1).reshape(-1, 1)
#     weights = np.diagflat(np.matrix(mean))
#     w = np.matmul(
#         np.linalg.inv(np.matmul(np.matmul(np.transpose(x), np.linalg.inv(weights)), x)),
#         np.matmul(np.matmul(np.transpose(x), np.linalg.inv(weights)), y)
#     )
#     return w

# Modeling
rg = LinearRegressionMultivariateNonIID()
weight = rg.fit(x_train, y_train)
print(weight)

# Evaluation
y_pre = np.matmul(x_test, weight)
error = y_test - y_pre
df_plot = pd.DataFrame(np.concatenate((y_test, y_pre), axis=1), columns=['y_test', 'y_pre'])
# print(df_plot.shape)
sns.lineplot(data=df_plot, palette=['r', 'b']).set_title('Evaluation Model (y_test vs y_train)')
plt.show()

sns.lineplot(data=error).set_title('Error')
plt.show()

print('mean(error)')
print(np.mean(error))
print('Standard Deviation(error)')
print(np.std(error))
