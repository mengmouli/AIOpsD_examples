import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import cont2discrete
from scipy.linalg import solve_discrete_are

# Continuous model (double integrator)
A = np.array([[0., 1.],
              [0., 0.]])
B = np.array([[0.],
              [1.]])

# Discretize with sampling time h (ZOH)
h = 0.02  # [s]
Ad, Bd, _, _, _ = cont2discrete((A, B, np.eye(2), np.zeros((2, 1))), h, method='zoh')

# Weights (Q >= 0, R > 0)
Qd = np.diag([1.0, 1.0])
Rd = np.array([[0.1]])

# DARE -> P
Pd = solve_discrete_are(Ad, Bd, Qd, Rd)

# LQR gain K = (R + B^T P B)^{-1} B^T P A  (use solve for numerical stability)
K = np.linalg.solve(Rd + Bd.T @ Pd @ Bd, Bd.T @ Pd @ Ad)   # shape (1,2)

# Closed-loop iteration: x_{k+1} = (Ad - Bd K) x_k,  u_k = -K x_k
Acl = Ad - Bd @ K
N = 400
x = np.array([1.0, 0.0])
X = [x.copy()]
U = []
for k in range(N):
    u = -K @ x
    x = Acl @ x
    X.append(x.copy())
    U.append(u.item())  # avoid deprecation: extract scalar from (1,1) array

X = np.array(X)                  # (N+1, 2)
U = np.array(U)                  # (N,)
t = h * np.arange(N + 1)         # time axis for states
tk = h * np.arange(N)            # time axis for inputs

# Plots
plt.figure(figsize=(6.5, 4.0))
plt.plot(t, X[:, 0], label="x1[k]")
plt.plot(t, X[:, 1], label="x2[k]")
plt.xlabel("Time [s]"); plt.ylabel("States")
plt.title("Discrete-time LQR (ZOH h=0.02 s): states")
plt.grid(True, alpha=0.4); plt.legend(loc="best")

plt.figure(figsize=(6.5, 3.6))
plt.step(tk, U, where='post')
plt.xlabel("Time [s]"); plt.ylabel("u[k]")
plt.title("Discrete-time LQR: control input")
plt.grid(True, alpha=0.4)

plt.show()
