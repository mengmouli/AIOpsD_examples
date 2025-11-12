import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

A = np.array([[0, 1],
              [-2, -3]], dtype=float)
B = np.array([[0, 1],
              [1, 1]], dtype=float)   # n x m (2 x 2)
C = np.eye(2)

def u_of_t(t):
    return np.array([1.0 if t >= 1.0 else 0.0,
                     0.5*np.sin(1.5*t)])

def f(t, x):
    return (A @ x + B @ u_of_t(t))

t_span = (0.0, 10.0)
t_eval = np.linspace(*t_span, 1001)
x0 = np.zeros(2)

sol = solve_ivp(f, t_span, x0, t_eval=t_eval, rtol=1e-7, atol=1e-9)
Y = (C @ sol.y)   # shape: (2, len(t))
y1, y2 = Y[0, :], Y[1, :]

plt.figure()
plt.plot(sol.t, y1, label="y1")
plt.plot(sol.t, y2, label="y2")
plt.xlabel("Time [s]"); plt.ylabel("Output")
plt.title("MIMO state response via solve_ivp")
plt.grid(True); plt.legend(); plt.show()