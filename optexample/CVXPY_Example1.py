import numpy as np
import cvxpy as cp

m, n = 50, 5                         # m = number of samples, n = number of variables
A = np.random.randn(m, n)            # random data matrix A (m x n)
b = np.random.randn(m)               # random vector b (m x 1)

x = cp.Variable(n)                   # optimization variable x in R^n
obj = 0.5 * cp.sum_squares(A @ x - b)  # objective: (1/2)||Ax - b||^2
prob = cp.Problem(cp.Minimize(obj))  # define problem: minimize obj
prob.solve()                         # solve using default solver

print("status:", prob.status)        # solver status
print("optimal value:", prob.value)  # optimal cost
print("x* (solution):", x.value)     # optimal variable

