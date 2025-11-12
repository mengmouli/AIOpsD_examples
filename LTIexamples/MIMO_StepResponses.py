import numpy as np
import matplotlib.pyplot as plt
import control as ctrl

A = np.array([[0, 1],
              [-2, -3]])
B = np.array([[0, 1],
              [1, 1]])          # n x m (n=2 states, m=2 inputs)
C = np.eye(2)                    # p x n (p=2 outputs)
D = np.zeros((2, 2))

G = ctrl.ss(A, B, C, D)

T = np.linspace(0, 10, 1001)                 # common time vector for all channels

t, y = ctrl.step_response(G, T=T)            # step response: y has shape (p, m, len(T))

fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharex=True)   # 2x2 grid of subplots
for i in range(2):                           # loop over outputs
    for j in range(2):                       # loop over inputs
        axes[i, j].plot(t, y[i, j, :])       # plot response of channel (j → i)
        axes[i, j].set_title(f"Output y{i+1} due to step in u{j+1}")  # title
        axes[i, j].grid(True)                # add grid

for ax in axes[-1, :]:
    ax.set_xlabel("Time [s]")
for ax in axes[:, 0]:
    ax.set_ylabel("Amplitude")

plt.tight_layout()                           # adjust layout to prevent overlap
plt.show()                                   # display the plot