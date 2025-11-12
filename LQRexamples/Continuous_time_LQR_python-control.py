# --- Requirements: numpy, scipy, matplotlib, control ---
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are
from scipy.integrate import solve_ivp
import control as ctrl

# System matrices (A, B)
A = np.array([[0., 1.],
              [0., 0.]])
B = np.array([[0.],
              [1.]])

# Weights
Q = np.diag([1.0, 1.0])    # state cost
R = np.array([[0.1]])      # input cost

K, P, eigs = ctrl.lqr(A, B, Q, R)  # K matches above (within numerics)

print("P =\n", P)
print("K =", K)

# Closed-loop dynamics: xdot = (A - B K) x
Acl = A - B @ K
def f(t, x): return Acl @ x

# Simulate
x0 = np.array([1.0, 0.0])              # initial state
T, N = 8.0, 2001                       # time horizon and samples
t_eval = np.linspace(0.0, T, N)
sol = solve_ivp(f, (0.0, T), x0, t_eval=t_eval, rtol=1e-8, atol=1e-10)

# Recover control u(t) = -K x(t)
X = sol.y.T                            # states (N,2)
U = -(X @ K.T).ravel()                 # control (N,)

# Plots
plt.figure(figsize=(6.5, 4.0))
plt.plot(sol.t, X[:, 0], label="x1 (position)")
plt.plot(sol.t, X[:, 1], label="x2 (velocity)")
plt.xlabel("Time [s]"); plt.ylabel("States")
plt.title("Continuous-time LQR: states")
plt.grid(True, alpha=0.4); plt.legend(loc="best")

plt.figure(figsize=(6.5, 3.6))
plt.plot(sol.t, U)
plt.xlabel("Time [s]"); plt.ylabel("u(t)")
plt.title("Continuous-time LQR: control input")
plt.grid(True, alpha=0.4)

plt.show()
