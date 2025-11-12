import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# System matrices (discrete-time double integrator)
Ts = 0.1
A = np.array([[1, Ts],  # A = [[1, Ts], [0, 1]]
              [0, 1]])
B = np.array([[0.5 * Ts ** 2],  # B = [[0.5*Ts^2], [Ts]]
              [Ts]])

# Weights and constraints
Q = np.diag([10, 1])  # state weighting matrix
R = np.array([[0.1]])  # input weighting matrix
umax = 2.0  # input constraint: |u| <= 2

# Horizon and simulation settings
N = 15  # prediction horizon
Tsim = 50  # number of simulation steps
x = np.array([[2.0], [0.0]])  # initial state [position=2; velocity=0]
xref = np.zeros((2, 1))  # reference state = [0;0]
uref = np.zeros((1, 1))  # reference input = 0

X_log, U_log = [x.copy()], []  # lists to store state and input trajectories

for k in range(Tsim):  # loop over simulation steps
    X = cp.Variable((2, N + 1))  # predicted states over horizon
    U = cp.Variable((1, N))  # predicted inputs over horizon
    cost = 0  # initialize cost
    constr = [X[:, [0]] == x]  # initial condition constraint

    for i in range(N):  # build cost and constraints along horizon
        dx = X[:, [i]] - xref  # deviation from reference state
        du = U[:, [i]] - uref  # deviation from reference input
        cost += cp.quad_form(dx, Q) + cp.quad_form(du, R)  # stage cost
        constr += [X[:, [i + 1]] == A @ X[:, [i]] + B @ U[:, [i]]]  # dynamics
        constr += [cp.abs(U[:, i]) <= umax]  # input constraint

    cost += cp.quad_form(X[:, [N]] - xref, Q)  # terminal cost (reuse Q)

    prob = cp.Problem(cp.Minimize(cost), constr)  # define optimization problem
    prob.solve(solver=cp.OSQP, warm_start=True)  # solve QP with OSQP

    u = U.value[:, [0]]  # first optimal input (MPC principle)
    x = A @ x + B @ u  # apply input to update system
    U_log.append(u)  # log applied input
    X_log.append(x.copy())  # log new state

X_log = np.hstack(X_log)  # convert state log to array
U_log = np.hstack(U_log)  # convert input log to array

# Plot results
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(X_log[0, :], label="Position")  # plot position
plt.plot(X_log[1, :], label="Velocity")  # plot velocity
plt.legend();
plt.grid();
plt.ylabel("States")

plt.subplot(2, 1, 2)
plt.step(range(Tsim), U_log.flatten(), label="Control input")  # plot control
plt.grid();
plt.ylabel("u");
plt.xlabel("Time step")
plt.show()