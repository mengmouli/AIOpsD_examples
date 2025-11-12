# Import required libraries
import numpy as np                   # numerical operations and arrays
import matplotlib.pyplot as plt      # plotting results
import control as ctrl               # control system modeling and simulation

# Define the plant: G(s) = 1 / (tau s + 1)
tau = 1.0                            # time constant (seconds)
num = [1]                            # numerator coefficients
den = [tau, 1]                       # denominator coefficients (tau*s + 1)
G = ctrl.TransferFunction(num, den)  # transfer function object

# Proportional control (unity feedback) for plant 'G'
Kp = 5.0
C = ctrl.TransferFunction([Kp], [1])     # C(s) = Kp

# Open loop and closed loop
L = C * G                                 # open-loop TF
closed_loop = ctrl.feedback(L, 1)         # unity feedback: L / (1 + L)

# Step response of closed-loop system
t, y = ctrl.step_response(closed_loop)

# Plot
plt.figure()
plt.plot(t, y)
plt.xlabel('Time [s]')
plt.ylabel('Output')
plt.title('Closed-loop with P Controller (Kp=5)')
plt.grid(True)
plt.show()
