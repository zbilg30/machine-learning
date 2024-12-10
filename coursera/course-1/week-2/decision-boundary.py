# def compute_cost_logistic(X, y, w, b):
#     """
#     Computes cost

#     Args:
#       X (ndarray (m,n)): Data, m examples with n features
#       y (ndarray (m,)) : target values
#       w (ndarray (n,)) : model parameters  
#       b (scalar)       : model parameter
      
#     Returns:
#       cost (scalar): cost
#     """

#     m = X.shape[0]
#     cost = 0.0
#     for i in range(m):
#         z_i = np.dot(X[i],w) + b
#         f_wb_i = sigmoid(z_i)
#         cost +=  -y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i)
             
#     cost = cost / m
#     return cost


import numpy as np
import matplotlib.pyplot as plt

# Sample data
X = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])  # Feature set
y = np.array([0, 1, 1, 0])  # Labels

# Logistic regression parameters
w = np.array([1, -1])  # Weights
b = -0.5  # Bias

# Function to calculate the decision boundary
def decision_boundary(x):
    return -(w[0] * x + b) / w[1]

# Plotting the data points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolor='k')

# Plotting the decision boundary
x_values = np.linspace(-1, 2, 100)
plt.plot(x_values, decision_boundary(x_values), color='green', label='Decision Boundary')

# Adding labels and title
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Logistic Regression Decision Boundary')
plt.legend()
plt.grid()
plt.show()