# Import required libraries
import numpy as np                   # numerical operations and arrays
import matplotlib.pyplot as plt      # plotting results
import control as ctrl               # control system modeling and simulation

# Define the plant: G(s) = 1 / (tau s + 1)
tau = 1.0                            # time constant (seconds)
num = [1]                            # numerator coefficients
den = [tau, 1]                       # denominator coefficients (tau*s + 1)
G = ctrl.TransferFunction(num, den)  # transfer function object

# Compare closed-loop responses for different proportional gains
Kp_values = [1, 5, 10]  # list of proportional gain values to test

# Define a common simulation horizon for all cases
T_end = 6 * tau  # simulate for about 6 time constants
T = np.linspace(0, T_end, 600)  # 600 points between 0 and T_end

plt.figure()
for Kp in Kp_values:
    # Define proportional controller for this Kp
    C = ctrl.TransferFunction([Kp], [1])

    # Closed-loop system with unity feedback
    closed_loop = ctrl.feedback(C * G, 1)

    # Step response using the same time vector T
    t, y = ctrl.step_response(closed_loop, T=T)

    # Plot response with label for the current Kp
    plt.plot(t, y, label=f"Kp={Kp}")

# Plot formatting
plt.xlabel('Time [s]')
plt.ylabel('Output')
plt.title('Effect of Kp on Step Response (common time grid)')
plt.legend()  # show legend for Kp values
plt.grid(True)
plt.show()