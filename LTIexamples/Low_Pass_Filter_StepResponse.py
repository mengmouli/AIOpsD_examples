# Import required libraries
import numpy as np                   # numerical operations and arrays
import matplotlib.pyplot as plt      # plotting results
import control as ctrl               # control system modeling and simulation

# Define the system: G(s) = 1 / (tau s + 1)
tau = 1.0                            # time constant (in seconds)
num = [1]                            # numerator coefficients
den = [tau, 1]                       # denominator coefficients ( tau s + 1)
G = ctrl.TransferFunction(num, den)  # create transfer function object

# Compute step response of the system
t, y = ctrl.step_response(G)         # t: time points, y: system output

# Plot the response
plt.figure()                         # open new figure
plt.plot(t, y)                       # plot output vs. time
plt.xlabel('Time [s]')                # label x-axis
plt.ylabel('Output')                  # label y-axis
plt.title('Step Response of First-Order System')  # figure title
plt.grid(True)                        # add grid for readability
plt.show()                            # display the plot