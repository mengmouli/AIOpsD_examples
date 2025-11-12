import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

m, c, k = 1.0, 2.0, 5.0
A = np.array([[0.0, 1.0],
              [-k/m, -c/m]])
B = np.array([[0.0],
              [1.0/m]])
C = np.array([[1.0, 0.0]])

def u_of_t(t):
    return 1.0  # unit step

def f(t, x):
    # x: shape (2,), return dx/dt with same shape
    u = u_of_t(t)
    return (A @ x + B.flatten()*u)

t_span = (0.0, 10.0)                   # simulation time interval [0, 10] seconds
t_eval = np.linspace(*t_span, 1001)    # 1001 evaluation points evenly spaced
x0 = np.zeros(2)                       # initial state x(0) = [0, 0]

sol = solve_ivp(f, t_span, x0,
                t_eval=t_eval,
                rtol=1e-8, atol=1e-10) # solve ODE with high accuracy tolerances

y = (C @ sol.y).flatten()              # compute output y(t) = C x(t) as 1D array

plt.figure()
plt.plot(sol.t, y)
plt.xlabel("Time [s]")
plt.ylabel("Displacement [m]")
plt.title("SISO MSD via solve_ivp (unit step input)")
plt.grid(True); plt.show()