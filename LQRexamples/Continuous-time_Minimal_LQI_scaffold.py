import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are
from scipy.integrate import solve_ivp

# Continuous-time double integrator
A = np.array([[0., 1.],
              [0., 0.]])
B = np.array([[0.],
              [1.]])
C = np.array([[1., 0.]])   # output y = x1 (position)

# Augmented system: [x; z], where z integrates error
A_aug = np.block([[A,            np.zeros((2,1))],
                  [-C,           np.zeros((1,1))]])
B_aug = np.vstack([B, np.zeros((1,1))])

# LQI weights (penalize integral state to enforce tracking)
Q_aug = np.diag([1.0, 1.0, 5.0])
R     = np.array([[0.1]])

# CARE on augmented system -> feedback gain
P_aug = solve_continuous_are(A_aug, B_aug, Q_aug, R)
K_aug = np.linalg.solve(R, B_aug.T @ P_aug)   # shape (1,3)
Kx, Ki = K_aug[:, :2], K_aug[:, 2:3]

print("Kx =", Kx, " Ki =", Ki)

# Reference to track
r = 1.0   # unit step

def f_aug(t, xtilde):
    x = xtilde[:2]
    z = xtilde[2:]
    u = -(Kx @ x + Ki @ z).item()
    y = (C @ x).item()
    xdot = (A @ x + B.flatten() * u)
    zdot = r - y                     # integral of error
    return np.hstack([xdot, zdot])

# Simulate augmented system
x0 = np.array([0.0, 0.0])    # initial state
z0 = np.array([0.0])         # integral state
xtilde0 = np.hstack([x0, z0])

T, N = 8.0, 2001
t_eval = np.linspace(0.0, T, N)
sol = solve_ivp(f_aug, (0.0, T), xtilde0, t_eval=t_eval, rtol=1e-8, atol=1e-10)

# Recover signals
X = sol.y[:2, :].T                     # plant states
Z = sol.y[2, :]                        # integral state
Y = (X @ C.T).ravel()                  # output
U = -(X @ Kx.T + Z.reshape(-1,1) @ Ki.T).ravel()

# Plots
plt.figure(figsize=(6.5, 4.0))
plt.plot(sol.t, Y, label="output y(t)")
plt.axhline(r, color="k", linestyle="--", label="reference")
plt.xlabel("Time [s]"); plt.ylabel("Output")
plt.title("LQI: tracking performance")
plt.grid(True, alpha=0.4); plt.legend(loc="best")

plt.figure(figsize=(6.5, 3.6))
plt.plot(sol.t, U)
plt.xlabel("Time [s]"); plt.ylabel("u(t)")
plt.title("LQI: control input")
plt.grid(True, alpha=0.4)

plt.show()