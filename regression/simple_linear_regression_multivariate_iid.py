# Initialization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from linear_model.LinearRegression import LinearRegressionMultivariateIID
# from sklearn.linear_model import LinearRegression


# Set random state
np.random.seed(42)

# Create dataset
N = 1000
x1 = np.random.randn(N, 1)
x2 = 2 + np.random.randn(N, 1)
x3 = 2 * np.random.randn(N, 1)
x4 = 2 + 2 * np.random.randn(N, 1)
X = np.concatenate((x1, x2, x3, x4), axis=1)
y = 0.5 - 3*x1 + 4*x2 + 3*x3 - 0.05*x4 + 0.5*np.random.randn(N, 1)
# print(X.shape)
# print(y.shape)

pdf = pd.DataFrame(data=np.concatenate((X, y), axis=1), columns=['x1', 'x2', 'x3', 'x4', 'y'])
print(pdf.head().to_string())
print(pdf.shape)

sns.scatterplot(data=pdf, x='x1', y='y')
sns.scatterplot(data=pdf, x='x2', y='y')
sns.scatterplot(data=pdf, x='x3', y='y')
sns.scatterplot(data=pdf, x='x4', y='y')
plt.legend(labels=['x1', 'x2', 'x3', 'x4'])
plt.title('Scatter 2d(input and output)')
plt.xlabel('x1, x2, x3, x4')
plt.show()

# sns.scatterplot(data=pdf, x='x1', y='x2')
# sns.scatterplot(data=pdf, x='x1', y='x3')
# sns.scatterplot(data=pdf, x='x1', y='x4')
# sns.scatterplot(data=pdf, x='x2', y='x3')
# sns.scatterplot(data=pdf, x='x2', y='x4')
# sns.scatterplot(data=pdf, x='x3', y='x4')
# plt.show()

# Heatmap Analysis (using correlation)
sns.heatmap(pdf.corr(), annot=True).set_title('Correlation Analysis')
plt.show()

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Preparing
one_train = np.ones((X_train.shape[0], 1))
X_train = np.concatenate((one_train, X_train), axis=1)
one_test = np.ones((X_test.shape[0], 1))
X_test = np.concatenate((one_test, X_test), axis=1)


# Modeling
# def simple_linear_regression(x, y):
#     return np.dot(np.linalg.inv(np.dot(np.transpose(x), x)), np.dot(np.transpose(x), y))
rg = LinearRegressionMultivariateIID()
weights = rg.fit(X_train, y_train)
print(weights)

# Evaluation
y_pre = np.matmul(X_test, weights)
error = y_pre - y_test
print(np.mean(y_pre))
print(np.std(y_pre))
plot_pdf = pd.DataFrame(data=np.concatenate((y_test, y_pre), axis=1), columns=['y_test', 'y_pre'])
sns.lineplot(data=plot_pdf, legend=True).set_title('Evaluate Model(y_test vs y_pre)')
plt.show()
