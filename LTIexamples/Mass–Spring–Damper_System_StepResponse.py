# Import required libraries
import numpy as np                   # numerical operations and arrays
import matplotlib.pyplot as plt      # plotting results
import control as ctrl               # control system modeling and simulation

# Parameters of the mass–spring–damper system
m = 1.0    # mass (kg)
c = 2.0    # damping coefficient (N·s/m)
k = 5.0    # spring constant (N/m)

# Define transfer function: X(s)/F(s) = 1 / (m s^2 + c s + k)
num = [1]                       # numerator coefficients
den = [m, c, k]                 # denominator coefficients (ms^2 + cs + k)
sys = ctrl.TransferFunction(num, den)   # system transfer function object

# Compute the step response (displacement under unit step force input)
t, y = ctrl.step_response(sys)  # t: time points, y: displacement output

# Plot the response
plt.figure()
plt.plot(t, y)                   # plot displacement vs. time
plt.xlabel('Time [s]')           # label x-axis
plt.ylabel('Displacement [m]')   # label y-axis
plt.title('Mass–Spring–Damper Step Response')  # figure title
plt.grid(True)                   # add grid for readability
plt.show()                       # display the plot