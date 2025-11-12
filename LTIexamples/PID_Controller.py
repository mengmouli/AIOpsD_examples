# Import required libraries
import numpy as np                   # numerical operations and arrays
import matplotlib.pyplot as plt      # plotting results
import control as ctrl               # control system modeling and simulation

# Define the plant: G(s) = 1 / (tau s + 1)
tau = 1.0                            # time constant (seconds)
num = [1]                            # numerator coefficients
den = [tau, 1]                       # denominator coefficients (tau*s + 1)
G = ctrl.TransferFunction(num, den)  # transfer function object

# PID controller parameters
Kp, Ki, Kd = 5.0, 1.0, 0.1     # proportional, integral, derivative gains

# Define the PID transfer function:
# C(s) = Kd*s + Kp + Ki/s
# In Laplace form: (Kd s^2 + Kp s + Ki) / s
C_pid = ctrl.TransferFunction([Kd, Kp, Ki], [1, 0])

# Closed-loop system with unity feedback
closed_loop_pid = ctrl.feedback(C_pid * G, 1)

# Step response of the closed-loop system
t, y = ctrl.step_response(closed_loop_pid)

# Plot the response
plt.figure()
plt.plot(t, y)
plt.xlabel('Time [s]')                # time axis
plt.ylabel('Output')                  # system output
plt.title('Closed-loop with PID Controller')
plt.grid(True)
plt.show()