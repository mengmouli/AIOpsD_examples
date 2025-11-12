import numpy as np
import cvxpy as cp

Q = np.array([[4.0, 0.0],
              [0.0, 1.0]])
c = np.array([-8.0, 3.0])
l = np.array([0.0, -1.0])
u   = np.array([2.0,  1.0])

x = cp.Variable(2)                        # decision variable x in R^2
obj = 0.5 * cp.quad_form(x, Q) + c @ x    # objective: (1/2)x^T Q x + c^T x
constraints = [x >= l, x <= u]            # box constraints: l <= x <= u
prob = cp.Problem(cp.Minimize(obj), constraints)  # define problem
prob.solve()                              # solve with default solver

print("status:", prob.status)
print("optimal value:", prob.value)
print("x* (cvxpy):", x.value)